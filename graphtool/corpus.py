import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from graphtool.chunking.markdown import chunk_markdown
from graphtool.chunking.types import Chunk
from graphtool.corpus_stores import SqliteCorpusStores
from graphtool.graph.embedding_store import (
    NodeEmbeddingRecord,
    SqliteGraphEmbeddingStore,
)
from graphtool.graph.extraction_store import JsonChunkExtractionStore
from graphtool.graph.combiner import combine_knowledge_graphs
from graphtool.graph.generator import generate_knowledge_graph
from graphtool.graph.knowledge_base_store import (
    KnowledgeBaseDelta,
)
from graphtool.graph.resolution_embeddings import EmbeddingStore
from graphtool.graph.resolver import (
    DEFAULT_MIN_CANDIDATE_SIMILARITY,
    SemanticEntityResolver,
)
from graphtool.graph.taxonomy_types import (
    TaxonomySuggestionRecord,
)
from graphtool.graph.types import KnowledgeGraph
from graphtool.llm.base import LLMClient
from graphtool.run_logging import LOGGER_NAME
from graphtool.source import document_content_hash

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class CorpusSyncResult:
    added_sources: list[str]
    changed_sources: list[str]
    deleted_sources: list[str]
    unchanged_sources: list[str]


@dataclass(frozen=True)
class _PreparedDocument:
    source: str
    chunks: list[Chunk]
    graph: KnowledgeGraph
    embedding_records: dict[str, NodeEmbeddingRecord]
    taxonomy_suggestions: list[TaxonomySuggestionRecord]


@dataclass
class _TaxonomySuggestionBuffer:
    records: list[TaxonomySuggestionRecord] = field(default_factory=list)

    def append_many(self, records: Sequence[TaxonomySuggestionRecord]) -> None:
        self.records.extend(records)


def synchronize_documents(
    documents: Mapping[str, str],
    stores: SqliteCorpusStores,
    llm: LLMClient,
    *,
    dropped_edges_path: Path | None = None,
    chunk_extraction_store: JsonChunkExtractionStore | None = None,
    min_candidate_similarity: float = DEFAULT_MIN_CANDIDATE_SIMILARITY,
    chunk_generation_workers: int = 4,
) -> CorpusSyncResult:
    if chunk_generation_workers < 1:
        raise ValueError("chunk_generation_workers must be positive")

    graph_store = stores.graphs
    knowledge_base_store = stores.knowledge_base
    graph_embedding_store = stores.graph_embeddings
    knowledge_base_embedding_store = stores.knowledge_base_embeddings
    chunk_store = stores.chunks
    chunk_embedding_store = stores.chunk_embeddings
    taxonomy_suggestion_store = stores.taxonomy_suggestions

    existing_by_source = {
        metadata.source: metadata
        for metadata in graph_store.load_metadata()
    }

    content_hashes = {
        source: document_content_hash(markdown)
        for source, markdown in documents.items()
    }
    current_sources = set(documents)
    existing_sources = set(existing_by_source)
    added_sources = sorted(current_sources - existing_sources)
    deleted_sources = sorted(existing_sources - current_sources)
    changed_sources = sorted(
        source
        for source in current_sources & existing_sources
        if existing_by_source[source].content_hash != content_hashes[source]
    )
    unchanged_sources = sorted(
        current_sources - set(added_sources) - set(changed_sources)
    )
    RUN_LOGGER.info(
        "Knowledge graph changes: %s added, %s changed, %s removed, %s unchanged",
        len(added_sources),
        len(changed_sources),
        len(deleted_sources),
        len(unchanged_sources),
    )

    prepared = []
    sources_to_prepare = [*added_sources, *changed_sources]
    for index, source in enumerate(sources_to_prepare, start=1):
        markdown = documents[source]
        chunks = chunk_markdown(markdown, source)
        RUN_LOGGER.info(
            "[%s/%s] Building knowledge graph: %s (%s %s)",
            index,
            len(sources_to_prepare),
            source,
            len(chunks),
            "chunk" if len(chunks) == 1 else "chunks",
        )
        resolver = _make_semantic_resolver(
            llm,
            graph_embedding_store.for_source(source),
            min_candidate_similarity=min_candidate_similarity,
        )
        suggestion_buffer = _TaxonomySuggestionBuffer()
        graph = generate_knowledge_graph(
            chunks,
            source,
            llm,
            content_hash=content_hashes[source],
            resolver=resolver,
            dropped_edges_path=dropped_edges_path,
            taxonomy_suggestion_store=suggestion_buffer,
            extraction_store=chunk_extraction_store,
            max_workers=chunk_generation_workers,
        )
        RUN_LOGGER.info(
            "Built knowledge graph: %s (%s %s, %s %s)",
            source,
            len(graph.nodes),
            "entity" if len(graph.nodes) == 1 else "entities",
            len(graph.edges),
            "relationship" if len(graph.edges) == 1 else "relationships",
        )
        prepared.append(
            _PreparedDocument(
                source=source,
                chunks=chunks,
                graph=graph,
                embedding_records=(
                    resolver.embedding_records if resolver is not None else {}
                ),
                taxonomy_suggestions=list(suggestion_buffer.records),
            )
        )

    result = CorpusSyncResult(
        added_sources=added_sources,
        changed_sources=changed_sources,
        deleted_sources=deleted_sources,
        unchanged_sources=unchanged_sources,
    )
    if not prepared and not deleted_sources and knowledge_base_store.exists():
        return result

    removed_sources = [*changed_sources, *deleted_sources]
    old_chunk_ids = [
        chunk.id
        for source in removed_sources
        for chunk in chunk_store.load(source)
    ]

    knowledge_base_exists = knowledge_base_store.exists()
    reusable_embedding_sources = (
        sources_to_prepare
        if knowledge_base_exists
        else sorted(current_sources)
    )
    resolver = _make_semantic_resolver(
        llm,
        knowledge_base_embedding_store,
        min_candidate_similarity=min_candidate_similarity,
        reusable_embedding_records=[
            *(
                record
                for item in prepared
                for record in item.embedding_records.values()
            ),
            *_load_document_embedding_records(
                graph_embedding_store,
                [
                    source
                    for source in reusable_embedding_sources
                    if source not in sources_to_prepare
                ],
            ),
        ],
    )
    if knowledge_base_exists:
        old_node_ids, old_edge_ids = knowledge_base_store.affected_ids(
            removed_sources
        )
        knowledge_base = knowledge_base_store.load_excluding_sources(
            removed_sources
        )
        if resolver is not None:
            knowledge_base = resolver.combine_into(
                knowledge_base,
                [item.graph for item in prepared],
            )
        else:
            knowledge_base = combine_knowledge_graphs(
                [knowledge_base, *(item.graph for item in prepared)]
            )
    else:
        old_node_ids, old_edge_ids = set(), set()
        final_graphs = [
            graph_store.load(source)
            for source in existing_by_source
            if source not in removed_sources
        ]
        final_graphs.extend(item.graph for item in prepared)
        knowledge_base = _combine_knowledge_graphs(final_graphs, resolver)

    if knowledge_base_exists:
        changed_source_set = set(sources_to_prepare)
        new_node_ids = {
            node.id
            for node in knowledge_base.nodes
            if any(
                provenance.source in changed_source_set
                for provenance in node.provenance
            )
        }
        new_edge_ids = {
            edge.id
            for edge in knowledge_base.edges
            if any(
                provenance.source in changed_source_set
                for provenance in edge.provenance
            )
        }
        affected_node_ids = old_node_ids | new_node_ids
        affected_edge_ids = old_edge_ids | new_edge_ids
        nodes_by_id = {node.id: node for node in knowledge_base.nodes}
        edges_by_id = {edge.id: edge for edge in knowledge_base.edges}
        knowledge_base_delta = KnowledgeBaseDelta(
            upserted_nodes=[
                nodes_by_id[node_id]
                for node_id in affected_node_ids
                if node_id in nodes_by_id
            ],
            deleted_node_ids=affected_node_ids - set(nodes_by_id),
            upserted_edges=[
                edges_by_id[edge_id]
                for edge_id in affected_edge_ids
                if edge_id in edges_by_id
            ],
            deleted_edge_ids=affected_edge_ids - set(edges_by_id),
        )
    else:
        affected_node_ids = {node.id for node in knowledge_base.nodes}
        knowledge_base_delta = None

    knowledge_embedding_records = (
        resolver.embedding_records if resolver is not None else {}
    )
    with stores.transaction():
        for source in deleted_sources:
            graph_store.delete(source)
            chunk_store.delete(source)
            graph_embedding_store.delete(source)
            if chunk_extraction_store is not None:
                chunk_extraction_store.delete(source)
            taxonomy_suggestion_store.delete_source(source)

        for item in prepared:
            chunk_store.save(item.source, item.chunks)
            graph_store.save(item.graph)
            graph_embedding_store.replace_source(
                item.source,
                item.embedding_records,
            )
            taxonomy_suggestion_store.replace_source(
                item.source,
                item.taxonomy_suggestions,
            )

        if old_chunk_ids:
            chunk_embedding_store.delete(old_chunk_ids)

        if knowledge_base_delta is None:
            knowledge_base_store.replace_all(knowledge_base)
            knowledge_base_embedding_store.replace_all(
                knowledge_embedding_records
            )
        else:
            knowledge_base_store.apply_delta(knowledge_base_delta)
            knowledge_base_embedding_store.delete(
                affected_node_ids - set(knowledge_embedding_records)
            )
            knowledge_base_embedding_store.upsert(
                {
                    node_id: knowledge_embedding_records[node_id]
                    for node_id in affected_node_ids
                    if node_id in knowledge_embedding_records
                }
            )

    return result


def rebuild_knowledge_base(
    stores: SqliteCorpusStores,
    *,
    resolver: SemanticEntityResolver | None = None,
) -> KnowledgeGraph:
    graph = _combine_knowledge_graphs(stores.graphs.load_all(), resolver)
    with stores.transaction():
        stores.knowledge_base.replace_all(graph)
        stores.knowledge_base_embeddings.replace_all(
            resolver.embedding_records if resolver is not None else {}
        )
    return graph


def _combine_knowledge_graphs(
    graphs: list[KnowledgeGraph],
    resolver: SemanticEntityResolver | None,
) -> KnowledgeGraph:
    if resolver is None:
        return combine_knowledge_graphs(graphs)
    return resolver.combine_into(None, graphs)


def _make_semantic_resolver(
    llm: LLMClient,
    embedding_store: EmbeddingStore | None,
    *,
    min_candidate_similarity: float = DEFAULT_MIN_CANDIDATE_SIMILARITY,
    reusable_embedding_records: Sequence[NodeEmbeddingRecord] = (),
) -> SemanticEntityResolver | None:
    if (
        not hasattr(llm, "embed_texts")
        or not hasattr(llm, "embedding_model")
    ):
        return None

    return SemanticEntityResolver(
        llm,
        llm,
        embedding_store,
        min_candidate_similarity=min_candidate_similarity,
        reusable_embedding_records=reusable_embedding_records,
    )


def _load_document_embedding_records(
    store: SqliteGraphEmbeddingStore | None,
    sources: Sequence[str],
) -> list[NodeEmbeddingRecord]:
    if store is None:
        return []
    return [
        record
        for source in sources
        for record in store.load(source).values()
    ]
