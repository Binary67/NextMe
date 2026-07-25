import logging
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from graphtool.chunking.types import Chunk
from graphtool.graph.types import KnowledgeGraph
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.embedding_store import ChunkEmbeddingStore
from graphtool.retrieval.context import (
    format_context,
    source_references,
)
from graphtool.retrieval.graph_retriever import (
    DEFAULT_MAX_HOPS,
    DEFAULT_TOP_PATHS,
    NodeEmbeddingStore,
    PreparedGraphRetriever,
    prepare_graph_retriever,
    retrieve_graph_context,
)
from graphtool.retrieval.retriever import (
    PreparedChunkRetriever,
    prepare_chunk_retriever,
    retrieve_context,
)
from graphtool.retrieval.types import ChunkHit, RetrievalResult
from graphtool.run_logging import LOGGER_NAME
from graphtool.sequences import unique_ordered

RECIPROCAL_RANK_CONSTANT = 60
RUN_LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class PreparedHybridRetriever:
    direct: PreparedChunkRetriever
    graph: PreparedGraphRetriever
    embedding_client: EmbeddingClient | None

    def retrieve(
        self,
        query: str,
        *,
        max_hops: int = DEFAULT_MAX_HOPS,
        top_paths: int = DEFAULT_TOP_PATHS,
        top_chunks: int = 5,
    ) -> RetrievalResult:
        retrieval_started_at = perf_counter()
        query_vector = None
        if self.embedding_client is not None and (
            self.direct.chunk_vectors or self.graph.node_vectors
        ):
            started_at = perf_counter()
            query_vector = self.embedding_client.embed_texts([query])[0]
            RUN_LOGGER.info(
                "Query embedding completed in %.2fs",
                perf_counter() - started_at,
            )
        started_at = perf_counter()
        direct_result = self.direct.retrieve(
            query,
            top_chunks=top_chunks,
            query_vector=query_vector,
        )
        RUN_LOGGER.info(
            "Direct retrieval completed in %.2fs",
            perf_counter() - started_at,
        )
        started_at = perf_counter()
        graph_result = self.graph.retrieve(
            query,
            max_hops=max_hops,
            top_paths=top_paths,
            top_chunks=top_chunks,
            query_vector=query_vector,
        )
        RUN_LOGGER.info(
            "Graph retrieval completed in %.2fs",
            perf_counter() - started_at,
        )
        result = _combine_results(query, direct_result, graph_result, top_chunks)
        RUN_LOGGER.info(
            "Retrieval completed in %.2fs: chunks=%d, sources=%d, graph paths=%d",
            perf_counter() - retrieval_started_at,
            len(result.chunks),
            len(result.sources),
            len(result.graph_paths),
        )
        return result


def prepare_hybrid_retriever(
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
    node_embedding_store: NodeEmbeddingStore | None = None,
) -> PreparedHybridRetriever:
    return PreparedHybridRetriever(
        direct=prepare_chunk_retriever(
            graph,
            chunks,
            embedding_client=embedding_client,
            chunk_embedding_store=chunk_embedding_store,
        ),
        graph=prepare_graph_retriever(
            graph,
            chunks,
            embedding_client=embedding_client,
            node_embedding_store=node_embedding_store,
        ),
        embedding_client=embedding_client,
    )


def retrieve_hybrid_context(
    query: str,
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    top_chunks: int = 5,
    embedding_client: EmbeddingClient | None = None,
    chunk_embedding_store: ChunkEmbeddingStore | None = None,
    node_embedding_store: NodeEmbeddingStore | None = None,
) -> RetrievalResult:
    query_vector = (
        embedding_client.embed_texts([query])[0]
        if embedding_client is not None and chunks
        else None
    )
    direct_result = retrieve_context(
        query,
        graph,
        chunks,
        top_chunks=top_chunks,
        embedding_client=embedding_client,
        chunk_embedding_store=chunk_embedding_store,
        query_vector=query_vector,
    )
    graph_result = retrieve_graph_context(
        query,
        graph,
        chunks,
        max_hops=max_hops,
        top_paths=top_paths,
        top_chunks=top_chunks,
        embedding_client=embedding_client,
        node_embedding_store=node_embedding_store,
        query_vector=query_vector,
    )
    return _combine_results(query, direct_result, graph_result, top_chunks)


def _combine_results(
    query: str,
    direct_result: RetrievalResult,
    graph_result: RetrievalResult,
    top_chunks: int,
) -> RetrievalResult:
    chunk_hits = _fuse_chunk_hits(
        direct_result.chunks,
        graph_result.chunks,
    )[:top_chunks]
    sources = unique_ordered(hit.chunk.source for hit in chunk_hits)

    return RetrievalResult(
        query=query,
        sources=sources,
        references=source_references(chunk_hits),
        chunks=chunk_hits,
        graph_paths=graph_result.graph_paths,
        context_text=format_context(
            query,
            chunk_hits,
            graph_result.graph_paths,
        ),
    )


def _fuse_chunk_hits(
    direct_hits: Sequence[ChunkHit],
    graph_hits: Sequence[ChunkHit],
) -> list[ChunkHit]:
    hits_by_id = {
        hit.chunk.id: hit
        for hit in [*direct_hits, *graph_hits]
    }
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for hits in (direct_hits, graph_hits):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + (
                1.0 / (RECIPROCAL_RANK_CONSTANT + rank)
            )
            best_rank[hit.chunk.id] = min(best_rank.get(hit.chunk.id, rank), rank)

    ranked = [
        hit.model_copy(update={"score": scores[chunk_id]})
        for chunk_id, hit in hits_by_id.items()
    ]
    ranked.sort(
        key=lambda hit: (
            -hit.score,
            best_rank[hit.chunk.id],
            hit.chunk.index,
            hit.chunk.id,
        )
    )
    return ranked
