from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from graphtool.chunking.types import Chunk
from graphtool.graph.embedding_store import NodeEmbeddingRecord
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.bm25 import BM25Index
from graphtool.retrieval.context import (
    ChunkGraphIndex,
    attach_graph_annotations,
    build_chunk_graph_index,
    format_context,
    source_references,
)
from graphtool.retrieval.graph_paths import (
    DEFAULT_BEAM_WIDTH,
    build_adjacency,
    candidate_chunk_ids,
    mentioned_node_scores,
    node_search_text,
    rank_evidence_chunks,
    rank_path_candidates,
    seed_node_scores,
    traverse_paths,
)
from graphtool.retrieval.scoring import bm25_index
from graphtool.retrieval.types import GraphPathHit, RetrievalResult
from graphtool.sequences import unique_ordered

DEFAULT_MAX_HOPS = 2
DEFAULT_TOP_PATHS = 5


class NodeEmbeddingStore(Protocol):
    def load(self) -> dict[str, NodeEmbeddingRecord]:
        ...


@dataclass(frozen=True)
class PreparedGraphRetriever:
    graph: KnowledgeGraph
    chunks_by_id: dict[str, Chunk]
    nodes_by_id: dict[str, Node]
    edges_by_id: dict[str, Edge]
    chunk_graph_index: ChunkGraphIndex
    label_index: BM25Index
    metadata_index: BM25Index
    node_vectors: dict[str, list[float]]
    adjacency: dict[str, list[tuple[Edge, str]]]

    def retrieve(
        self,
        query: str,
        *,
        max_hops: int = DEFAULT_MAX_HOPS,
        top_paths: int = DEFAULT_TOP_PATHS,
        top_chunks: int = 5,
        query_vector: Sequence[float] | None = None,
    ) -> RetrievalResult:
        _validate_limits(max_hops, top_paths, top_chunks)
        mention_scores = mentioned_node_scores(query, self.graph.nodes)
        seed_scores = seed_node_scores(
            query,
            self.graph.nodes,
            mention_scores,
            self.label_index,
            self.metadata_index,
            query_vector,
            self.node_vectors,
        )
        candidates = traverse_paths(
            query,
            self.adjacency,
            self.chunks_by_id,
            self.nodes_by_id,
            self.edges_by_id,
            seed_scores,
            mention_scores,
            max_hops=max_hops,
            beam_width=max(DEFAULT_BEAM_WIDTH, top_paths * 5),
        )
        ranked_paths = rank_path_candidates(
            query,
            candidates,
            self.chunks_by_id,
            self.nodes_by_id,
            self.edges_by_id,
            seed_scores,
            mention_scores,
        )

        graph_paths = []
        for candidate, score in ranked_paths:
            chunk_ids = candidate_chunk_ids(
                candidate,
                self.chunks_by_id,
                self.nodes_by_id,
                self.edges_by_id,
            )
            if not chunk_ids:
                continue
            graph_paths.append(
                GraphPathHit(
                    score=score,
                    nodes=[
                        self.nodes_by_id[node_id]
                        for node_id in candidate.node_ids
                    ],
                    edges=[
                        self.edges_by_id[edge_id]
                        for edge_id in candidate.edge_ids
                    ],
                    chunk_ids=chunk_ids,
                )
            )
            if len(graph_paths) == top_paths:
                break

        ranked_chunks = rank_evidence_chunks(
            graph_paths,
            self.chunks_by_id,
        )[:top_chunks]
        chunk_hits = attach_graph_annotations(
            ranked_chunks,
            self.chunk_graph_index,
            self.nodes_by_id,
        )
        sources = unique_ordered(hit.chunk.source for hit in chunk_hits)
        return RetrievalResult(
            query=query,
            sources=sources,
            references=source_references(chunk_hits),
            chunks=chunk_hits,
            graph_paths=graph_paths,
            context_text=format_context(query, chunk_hits, graph_paths),
        )


def retrieve_graph_context(
    query: str,
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    top_chunks: int = 5,
    embedding_client: EmbeddingClient | None = None,
    node_embedding_store: NodeEmbeddingStore | None = None,
    query_vector: Sequence[float] | None = None,
) -> RetrievalResult:
    _validate_limits(max_hops, top_paths, top_chunks)
    prepared = prepare_graph_retriever(
        graph,
        chunks,
        embedding_client=embedding_client,
        node_embedding_store=node_embedding_store,
    )
    if query_vector is None and embedding_client is not None and prepared.node_vectors:
        query_vector = embedding_client.embed_texts([query])[0]
    return prepared.retrieve(
        query,
        max_hops=max_hops,
        top_paths=top_paths,
        top_chunks=top_chunks,
        query_vector=query_vector,
    )


def prepare_graph_retriever(
    graph: KnowledgeGraph,
    chunks: Sequence[Chunk],
    *,
    embedding_client: EmbeddingClient | None = None,
    node_embedding_store: NodeEmbeddingStore | None = None,
) -> PreparedGraphRetriever:
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    nodes_by_id = {node.id: node for node in graph.nodes}
    edges_by_id = {edge.id: edge for edge in graph.edges}
    return PreparedGraphRetriever(
        graph=graph,
        chunks_by_id=chunks_by_id,
        nodes_by_id=nodes_by_id,
        edges_by_id=edges_by_id,
        chunk_graph_index=build_chunk_graph_index(
            graph,
            chunks_by_id,
            nodes_by_id,
        ),
        label_index=bm25_index(
            {node.id: node.label for node in graph.nodes}
        ),
        metadata_index=bm25_index(
            {node.id: node_search_text(node) for node in graph.nodes}
        ),
        node_vectors=_load_node_vectors(
            graph.nodes,
            embedding_client,
            node_embedding_store,
        ),
        adjacency=build_adjacency(graph),
    )


def _validate_limits(max_hops: int, top_paths: int, top_chunks: int) -> None:
    if max_hops < 1:
        raise ValueError("max_hops must be positive")
    if top_paths < 1:
        raise ValueError("top_paths must be positive")
    if top_chunks < 1:
        raise ValueError("top_chunks must be positive")


def _load_node_vectors(
    nodes: Sequence[Node],
    embedding_client: EmbeddingClient | None,
    node_embedding_store: NodeEmbeddingStore | None,
) -> dict[str, list[float]]:
    if embedding_client is None or node_embedding_store is None:
        return {}
    records = node_embedding_store.load()
    return {
        node.id: records[node.id].vector
        for node in nodes
        if node.id in records
        and records[node.id].embedding_model == embedding_client.embedding_model
    }
