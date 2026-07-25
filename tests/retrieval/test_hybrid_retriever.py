import pytest

import graphtool.retrieval.hybrid_retriever as hybrid_retriever
from graphtool.chunking.types import Chunk
from graphtool.graph.embedding_store import NodeEmbeddingRecord
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.retrieval import (
    ChunkHit,
    GraphPathHit,
    RetrievalResult,
)
from graphtool.retrieval.hybrid_retriever import retrieve_hybrid_context
from graphtool.retrieval.hybrid_retriever import prepare_hybrid_retriever
from graphtool.retrieval.graph_retriever import PreparedGraphRetriever
from graphtool.retrieval.retriever import (
    PreparedChunkRetriever,
    retrieve_context,
)


class FakeEmbeddingClient:
    embedding_model = "current-model"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        batch = list(texts)
        self.calls.extend(batch)
        return [[1.0, 0.0] for _ in batch]


class MemoryChunkEmbeddingStore:
    def __init__(self):
        self.records = {}

    def load(self):
        return self.records

    def upsert(self, records):
        self.records = dict(records)


class MemoryNodeEmbeddingStore:
    def load(self):
        return {
            "alpha": NodeEmbeddingRecord(
                node_id="alpha",
                embedding_model="current-model",
                embedding_input_hash="alpha-hash",
                vector=[1.0, 0.0],
            )
        }


def _corpus() -> tuple[KnowledgeGraph, list[Chunk]]:
    chunks = [
        Chunk(
            id="alpha-beta",
            source="doc.md",
            index=0,
            text="Alpha uses Beta.",
        ),
        Chunk(
            id="beta-gamma",
            source="doc.md",
            index=1,
            text="Beta depends on Gamma.",
        ),
        Chunk(
            id="operations",
            source="operations.md",
            index=0,
            text="Retry throttled deployment operations.",
        ),
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="alpha",
                label="Alpha",
                type="System",
                chunk_ids=["alpha-beta"],
            ),
            Node(
                id="beta",
                label="Beta",
                type="Component",
                chunk_ids=["alpha-beta", "beta-gamma"],
            ),
            Node(
                id="gamma",
                label="Gamma",
                type="Service",
                chunk_ids=["beta-gamma"],
            ),
        ],
        edges=[
            Edge(
                id="alpha-beta-edge",
                source="alpha",
                target="beta",
                label="uses",
                chunk_ids=["alpha-beta"],
            ),
            Edge(
                id="beta-gamma-edge",
                source="beta",
                target="gamma",
                label="depends on",
                chunk_ids=["beta-gamma"],
            ),
        ],
    )
    return graph, chunks


def _hit(chunk_id: str, source: str, index: int) -> ChunkHit:
    return ChunkHit(
        chunk=Chunk(
            id=chunk_id,
            source=source,
            index=index,
            text=f"Evidence for {chunk_id}.",
        ),
        score=1.0,
        relevance=1.0,
        linked_nodes=[],
        linked_relationships=[],
    )


def test_hybrid_search_fuses_unique_chunks_and_preserves_graph_paths():
    graph, chunks = _corpus()

    result = retrieve_hybrid_context(
        "How is Alpha related to Gamma?",
        graph,
        chunks,
        max_hops=2,
        top_paths=1,
        top_chunks=3,
    )

    chunk_ids = [hit.chunk.id for hit in result.chunks]
    assert chunk_ids == ["alpha-beta", "beta-gamma"]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(result.graph_paths) == 1
    assert [edge.id for edge in result.graph_paths[0].edges] == [
        "alpha-beta-edge",
        "beta-gamma-edge",
    ]
    assert "Graph paths:" in result.context_text


def test_hybrid_search_fuses_distinct_rankings_and_limits_output(monkeypatch):
    graph, chunks = _corpus()
    overlap = _hit("overlap", "shared.md", 0)
    direct_only = _hit("direct-only", "direct.md", 1)
    graph_only = _hit("graph-only", "graph.md", 2)
    graph_path = GraphPathHit(
        score=1.0,
        nodes=graph.nodes[:2],
        edges=graph.edges[:1],
        chunk_ids=["graph-only"],
    )

    def direct_search(self, query, scores, **kwargs):
        return RetrievalResult(
            query=query,
            sources=["shared.md", "direct.md"],
            chunks=[overlap, direct_only],
            context_text="Direct context.",
        )

    def graph_search(self, query, **kwargs):
        return RetrievalResult(
            query=query,
            sources=["graph.md", "shared.md"],
            chunks=[graph_only, overlap],
            graph_paths=[graph_path],
            context_text="Graph context.",
        )

    def chunk_scores(self, query, *, query_vector=None):
        return {"overlap": 1.0, "direct-only": 1.0, "graph-only": 1.0}

    monkeypatch.setattr(
        PreparedChunkRetriever,
        "result_for_scores",
        direct_search,
    )
    monkeypatch.setattr(PreparedChunkRetriever, "chunk_scores", chunk_scores)
    monkeypatch.setattr(PreparedGraphRetriever, "retrieve", graph_search)

    result = retrieve_hybrid_context("query", graph, chunks, top_chunks=3)
    limited = retrieve_hybrid_context("query", graph, chunks, top_chunks=2)

    assert [hit.chunk.id for hit in result.chunks] == [
        "overlap",
        "graph-only",
        "direct-only",
    ]
    assert result.chunks[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert result.chunks[1].score == pytest.approx(1 / 61)
    assert result.chunks[2].score == pytest.approx(1 / 62)
    assert all(hit.relevance == 1.0 for hit in result.chunks)
    assert result.sources == ["shared.md", "graph.md", "direct.md"]
    assert [reference.source for reference in result.references] == [
        "shared.md",
        "graph.md",
        "direct.md",
    ]
    assert result.graph_paths == [graph_path]
    assert [hit.chunk.id for hit in limited.chunks] == ["overlap", "graph-only"]


def test_hybrid_search_falls_back_to_direct_chunks_without_graph_match():
    graph, chunks = _corpus()

    direct = retrieve_context("throttled deployment", graph, chunks)
    hybrid = retrieve_hybrid_context("throttled deployment", graph, chunks)

    assert [hit.chunk.id for hit in hybrid.chunks] == [
        hit.chunk.id for hit in direct.chunks
    ]
    assert hybrid.graph_paths == []
    assert hybrid.sources == ["operations.md"]


def test_hybrid_search_embeds_query_once_for_chunk_and_graph_search():
    graph, chunks = _corpus()
    embedding = FakeEmbeddingClient()

    retrieve_hybrid_context(
        "How is Alpha related to Gamma?",
        graph,
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=MemoryChunkEmbeddingStore(),
        node_embedding_store=MemoryNodeEmbeddingStore(),
    )

    assert embedding.calls.count("How is Alpha related to Gamma?") == 1


def test_prepared_hybrid_search_logs_stage_timings_and_result_counts(monkeypatch):
    graph, chunks = _corpus()
    logger = type(
        "FakeLogger",
        (),
        {"info": lambda self, *args: calls.append(args)},
    )()
    calls = []
    monkeypatch.setattr(hybrid_retriever, "RUN_LOGGER", logger)
    prepared = prepare_hybrid_retriever(graph, chunks)

    prepared.retrieve("How is Alpha related to Gamma?")

    messages = [call[0] for call in calls]
    assert messages == [
        "Direct retrieval completed in %.2fs",
        "Graph retrieval completed in %.2fs",
        "Retrieval completed in %.2fs: chunks=%d, sources=%d, graph paths=%d",
    ]
    assert calls[-1][2:] == (2, 1, 3)
