from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from graphtool.chunking.types import Chunk
from graphtool.graph.chunk_graph import build_chunk_graph, log_document_graph
from graphtool.graph.combiner import combine_knowledge_graphs
from graphtool.graph.extraction import extract_chunks
from graphtool.graph.extraction_store import ChunkExtractionStore
from graphtool.graph.taxonomy import make_taxonomy_suggestion_records
from graphtool.graph.taxonomy_types import TaxonomySuggestionStore
from graphtool.graph.types import GraphMetadata, KnowledgeGraph
from graphtool.llm.base import LLMClient


class GraphResolver(Protocol):
    def combine(self, graphs: Sequence[KnowledgeGraph]) -> KnowledgeGraph:
        ...

    def combine_into(
        self,
        existing: KnowledgeGraph | None,
        graphs: Sequence[KnowledgeGraph],
    ) -> KnowledgeGraph:
        ...


def generate_knowledge_graph(
    chunks: Sequence[Chunk],
    source: str,
    llm: LLMClient,
    *,
    content_hash: str,
    resolver: GraphResolver | None = None,
    dropped_edges_path: Path | None = None,
    taxonomy_suggestion_store: TaxonomySuggestionStore | None = None,
    extraction_store: ChunkExtractionStore | None = None,
    max_workers: int = 4,
) -> KnowledgeGraph:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    extractions = extract_chunks(
        chunks,
        source,
        llm,
        extraction_store,
        max_workers=max_workers,
    )
    generated_chunks = [
        build_chunk_graph(
            chunk,
            extracted,
            dropped_edges_path=dropped_edges_path,
        )
        for chunk, extracted in zip(chunks, extractions.graphs, strict=True)
    ]

    taxonomy_suggestion_records = []
    if taxonomy_suggestion_store is not None:
        for chunk, generated in zip(chunks, generated_chunks, strict=True):
            taxonomy_suggestion_records.extend(
                make_taxonomy_suggestion_records(
                    nodes=generated.graph.nodes,
                    source=chunk.source,
                    chunk_id=chunk.id,
                )
            )
        if taxonomy_suggestion_records:
            taxonomy_suggestion_store.append_many(taxonomy_suggestion_records)

    graphs = [generated.graph for generated in generated_chunks]
    graph = (
        resolver.combine(graphs)
        if resolver is not None
        else combine_knowledge_graphs(graphs)
    )
    if extraction_store is not None:
        extraction_store.replace(source, extractions.records)
    log_document_graph(
        source,
        len(chunks),
        generated_chunks,
        graph,
        cached_chunks=extractions.cached_chunks,
        generated_chunk_count=extractions.generated_chunks,
        extraction_requests=extractions.extraction_requests,
    )
    return graph.model_copy(
        update={
            "metadata": GraphMetadata(
                source=source,
                content_hash=content_hash,
                model=None,
                created_at=datetime.now(timezone.utc),
            )
        }
    )
