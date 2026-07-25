from collections.abc import Sequence
from dataclasses import dataclass

from graphtool.chunking.types import Chunk
from graphtool.graph.types import KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.bm25 import BM25Index
from graphtool.retrieval.chunk_index import (
    prepare_chunk_vectors,
    search_fields_by_chunk,
    searchable_text_by_chunk,
)
from graphtool.retrieval.context import (
    ChunkGraphIndex,
    attach_graph_annotations,
    build_chunk_graph_index,
    format_context,
    source_references,
)
from graphtool.retrieval.embedding_store import ChunkEmbeddingStore
from graphtool.retrieval.scoring import (
    bm25_index,
    bm25_scores,
    semantic_similarity_scores,
)
from graphtool.retrieval.types import RetrievalResult
from graphtool.sequences import unique_ordered

PRIMARY_LABEL_BM25_WEIGHT = 2.0
ALIAS_BM25_WEIGHT = 1.5
CONTENT_BM25_WEIGHT = 1.0
METADATA_BM25_WEIGHT = 0.5
SEMANTIC_CHUNK_WEIGHT = 1.0


@dataclass(frozen=True)
class PreparedChunkRetriever:
    chunks_by_id: dict[str, Chunk]
    nodes_by_id: dict[str, Node]
    graph_index: ChunkGraphIndex
    primary_label_index: BM25Index
    alias_index: BM25Index
    content_index: BM25Index
    metadata_index: BM25Index
    chunk_vectors: dict[str, list[float]]

    def retrieve(
        self,
        query: str,
        *,
        top_chunks: int = 5,
        query_vector: Sequence[float] | None = None,
    ) -> RetrievalResult:
        ranked_chunks = _rank_chunks(query, self, query_vector)[:top_chunks]
        chunk_hits = attach_graph_annotations(
            ranked_chunks,
            self.graph_index,
            self.nodes_by_id,
        )
        sources = unique_ordered(hit.chunk.source for hit in chunk_hits)
        return RetrievalResult(
            query=query,
            sources=sources,
            references=source_references(chunk_hits),
            chunks=chunk_hits,
            context_text=format_context(query, chunk_hits),
        )


def retrieve_context(
    query: str,
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    top_chunks: int = 5,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
    query_vector: Sequence[float] | None = None,
) -> RetrievalResult:
    prepared = prepare_chunk_retriever(
        graph,
        chunks,
        embedding_client=embedding_client,
        chunk_embedding_store=chunk_embedding_store,
    )
    if query_vector is None and embedding_client is not None and prepared.chunk_vectors:
        query_vector = embedding_client.embed_texts([query])[0]
    return prepared.retrieve(
        query,
        top_chunks=top_chunks,
        query_vector=query_vector,
    )


def prepare_chunk_retriever(
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
) -> PreparedChunkRetriever:
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    nodes_by_id = {node.id: node for node in graph.nodes}
    graph_index = build_chunk_graph_index(graph, chunks_by_id, nodes_by_id)
    fields_by_chunk = search_fields_by_chunk(chunks_by_id, graph_index)
    searchable_text = searchable_text_by_chunk(
        chunks_by_id,
        graph_index,
        nodes_by_id,
    )
    return PreparedChunkRetriever(
        chunks_by_id=chunks_by_id,
        nodes_by_id=nodes_by_id,
        graph_index=graph_index,
        primary_label_index=bm25_index(
            {
                chunk_id: fields.primary_labels
                for chunk_id, fields in fields_by_chunk.items()
            }
        ),
        alias_index=bm25_index(
            {
                chunk_id: fields.aliases
                for chunk_id, fields in fields_by_chunk.items()
            }
        ),
        content_index=bm25_index(
            {
                chunk_id: fields.content
                for chunk_id, fields in fields_by_chunk.items()
            }
        ),
        metadata_index=bm25_index(
            {
                chunk_id: fields.metadata
                for chunk_id, fields in fields_by_chunk.items()
            }
        ),
        chunk_vectors=prepare_chunk_vectors(
            searchable_text,
            embedding_client,
            chunk_embedding_store,
        ),
    )


def _rank_chunks(
    query: str,
    prepared: PreparedChunkRetriever,
    query_vector: Sequence[float] | None,
) -> list[tuple[Chunk, float]]:
    primary_label_scores = bm25_scores(query, prepared.primary_label_index)
    alias_scores = bm25_scores(query, prepared.alias_index)
    content_scores = bm25_scores(query, prepared.content_index)
    metadata_scores = bm25_scores(query, prepared.metadata_index)
    semantic_scores = semantic_similarity_scores(
        query_vector,
        prepared.chunk_vectors,
    )

    ranked = []
    for chunk in prepared.chunks_by_id.values():
        score = (
            primary_label_scores.get(chunk.id, 0.0) * PRIMARY_LABEL_BM25_WEIGHT
            + alias_scores.get(chunk.id, 0.0) * ALIAS_BM25_WEIGHT
            + content_scores.get(chunk.id, 0.0) * CONTENT_BM25_WEIGHT
            + metadata_scores.get(chunk.id, 0.0) * METADATA_BM25_WEIGHT
            + semantic_scores.get(chunk.id, 0.0) * SEMANTIC_CHUNK_WEIGHT
        )
        if score > 0:
            ranked.append((chunk, score))
    ranked.sort(key=lambda item: (-item[1], item[0].index, item[0].id))
    return ranked
