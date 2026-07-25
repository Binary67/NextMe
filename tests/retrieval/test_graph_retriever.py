import pytest

from graphtool.chunking.types import Chunk
from graphtool.graph.embedding_store import NodeEmbeddingRecord, SqliteEmbeddingStore
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.retrieval.graph_paths import PATH_HOP_DECAY
from graphtool.retrieval.graph_retriever import retrieve_graph_context
from graphtool.storage import open_database


class FakeEmbeddingClient:
    embedding_model = "current-model"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        return [[1.0, 0.0] for _ in batch]


def _chunks() -> list[Chunk]:
    return [
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
            id="beta-company",
            source="doc.md",
            index=2,
            text="Beta was created by Example Corp.",
        ),
    ]


def _graph(*, include_cycle: bool = False) -> KnowledgeGraph:
    edges = [
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
        Edge(
            id="beta-company-edge",
            source="beta",
            target="company",
            label="created by",
            chunk_ids=["beta-company"],
        ),
    ]
    if include_cycle:
        edges.append(
            Edge(
                id="gamma-alpha-edge",
                source="gamma",
                target="alpha",
                label="feeds back to",
                chunk_ids=["beta-gamma"],
            )
        )

    return KnowledgeGraph(
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
                chunk_ids=["alpha-beta", "beta-gamma", "beta-company"],
            ),
            Node(
                id="gamma",
                label="Gamma",
                type="Service",
                chunk_ids=["beta-gamma"],
            ),
            Node(
                id="company",
                label="Example Corp",
                type="Organization",
                chunk_ids=["beta-company"],
            ),
        ],
        edges=edges,
    )


def test_graph_search_returns_relevant_two_hop_path_and_evidence():
    result = retrieve_graph_context(
        "How is Alpha related to Gamma?",
        _graph(),
        _chunks(),
        max_hops=2,
        top_paths=1,
        top_chunks=3,
    )

    path = result.graph_paths[0]
    assert [node.id for node in path.nodes] == ["alpha", "beta", "gamma"]
    assert [edge.id for edge in path.edges] == [
        "alpha-beta-edge",
        "beta-gamma-edge",
    ]
    assert path.chunk_ids == ["alpha-beta", "beta-gamma"]
    assert [hit.chunk.id for hit in result.chunks] == [
        "alpha-beta",
        "beta-gamma",
    ]
    assert "Alpha --uses--> Beta --depends on--> Gamma" in result.context_text
    assert "Example Corp" not in result.context_text


def test_graph_search_uses_cached_node_embeddings_for_semantic_seed(tmp_path):
    store = SqliteEmbeddingStore(open_database(tmp_path / "node_embeddings.db"))
    store.upsert(
        {
            "alpha": NodeEmbeddingRecord(
                node_id="alpha",
                embedding_model="current-model",
                embedding_input_hash="alpha-hash",
                vector=[1.0, 0.0],
            )
        }
    )
    embedding_client = FakeEmbeddingClient()

    result = retrieve_graph_context(
        "installation stalls",
        _graph(),
        _chunks(),
        max_hops=1,
        top_paths=1,
        embedding_client=embedding_client,
        node_embedding_store=store,
    )

    assert embedding_client.calls == [["installation stalls"]]
    assert [node.id for node in result.graph_paths[0].nodes] == ["alpha", "beta"]
    assert [hit.chunk.id for hit in result.chunks] == ["alpha-beta"]


def test_graph_search_ignores_cached_embeddings_from_another_model(tmp_path):
    store = SqliteEmbeddingStore(open_database(tmp_path / "node_embeddings.db"))
    store.upsert(
        {
            "alpha": NodeEmbeddingRecord(
                node_id="alpha",
                embedding_model="stale-model",
                embedding_input_hash="alpha-hash",
                vector=[1.0, 0.0],
            )
        }
    )
    embedding_client = FakeEmbeddingClient()

    result = retrieve_graph_context(
        "installation stalls",
        _graph(),
        _chunks(),
        embedding_client=embedding_client,
        node_embedding_store=store,
    )

    assert embedding_client.calls == []
    assert result.graph_paths == []
    assert result.chunks == []


def test_graph_path_scores_stay_below_one_when_query_names_path_nodes():
    result = retrieve_graph_context(
        "How is Alpha related to Gamma?",
        _graph(),
        _chunks(),
        max_hops=2,
        top_paths=5,
    )

    assert result.graph_paths[0].score < PATH_HOP_DECAY
    assert len({path.score for path in result.graph_paths}) == len(
        result.graph_paths
    )


def test_graph_search_respects_max_hops():
    result = retrieve_graph_context(
        "How is Alpha related to Gamma?",
        _graph(),
        _chunks(),
        max_hops=1,
        top_paths=5,
    )

    assert result.graph_paths
    assert all(len(path.edges) == 1 for path in result.graph_paths)
    assert all(
        {"alpha", "gamma"} != {node.id for node in path.nodes}
        for path in result.graph_paths
    )


def test_graph_search_respects_top_path_and_chunk_limits():
    unrestricted = retrieve_graph_context(
        "How is Alpha related to Gamma?",
        _graph(),
        _chunks(),
        top_paths=5,
        top_chunks=3,
    )
    limited = retrieve_graph_context(
        "How is Alpha related to Gamma?",
        _graph(),
        _chunks(),
        top_paths=1,
        top_chunks=1,
    )

    assert len(unrestricted.graph_paths) > 1
    assert len(unrestricted.chunks) > 1
    assert len(limited.graph_paths) == 1
    assert len(limited.chunks) == 1


def test_graph_search_does_not_revisit_nodes_in_cycles():
    result = retrieve_graph_context(
        "Alpha Gamma feedback",
        _graph(include_cycle=True),
        _chunks(),
        max_hops=2,
        top_paths=10,
    )

    assert result.graph_paths
    for path in result.graph_paths:
        node_ids = [node.id for node in path.nodes]
        assert len(node_ids) == len(set(node_ids))


def test_graph_search_returns_empty_result_without_relevant_seed():
    result = retrieve_graph_context(
        "xylophone",
        _graph(),
        _chunks(),
    )

    assert result.graph_paths == []
    assert result.chunks == []
    assert result.sources == []
    assert result.references == []
    assert result.context_text == "Query: xylophone\n\nEvidence:\n- None"


@pytest.mark.parametrize(
    ("limit", "message"),
    [
        ({"max_hops": 0}, "max_hops must be positive"),
        ({"top_paths": 0}, "top_paths must be positive"),
        ({"top_chunks": 0}, "top_chunks must be positive"),
    ],
)
def test_graph_search_requires_positive_limits(limit, message):
    with pytest.raises(ValueError, match=message):
        retrieve_graph_context("Alpha", _graph(), _chunks(), **limit)
