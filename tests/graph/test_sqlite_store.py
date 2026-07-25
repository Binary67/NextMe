from datetime import datetime, timezone

import pytest

from graphtool.graph.embedding_store import (
    NodeEmbeddingRecord,
    SqliteEmbeddingStore,
    SqliteGraphEmbeddingStore,
)
from graphtool.graph.provenance import add_edge_provenance, add_node_provenance
from graphtool.graph.document_store import SqliteGraphStore
from graphtool.graph.knowledge_base_store import (
    KnowledgeBaseDelta,
    SqliteKnowledgeBaseStore,
)
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.chunking.store import SqliteChunkStore
from graphtool.chunking.types import Chunk
from graphtool.storage import open_database


def _metadata(source: str) -> GraphMetadata:
    return GraphMetadata(
        source=source,
        content_hash=f"hash:{source}",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _sample_graph(source: str = "doc.md") -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id="a",
                label="A",
                type="Concept",
                aliases=["Concept A"],
                chunk_ids=["doc-chunk-0000"],
            )
        ],
        edges=[
            Edge(
                id="e1",
                source="a",
                target="a",
                label="relates_to",
                chunk_ids=["doc-chunk-0000"],
            )
        ],
        metadata=_metadata(source),
    )


def test_document_graph_store_roundtrips_and_lists_by_source(tmp_path):
    store = SqliteGraphStore(tmp_path / "graphs.db")
    first = _sample_graph("docs/user/guide.md")
    second = _sample_graph("docs/api/guide.md")

    store.save(first)
    store.save(second)

    assert store.load(first.metadata.source) == first
    assert [item.source for item in store.load_metadata()] == [
        "docs/api/guide.md",
        "docs/user/guide.md",
    ]
    assert [
        graph.metadata.source
        for graph in store.load_all()
        if graph.metadata is not None
    ] == [
        "docs/api/guide.md",
        "docs/user/guide.md",
    ]


def test_document_graph_store_replaces_only_one_source(tmp_path):
    store = SqliteGraphStore(tmp_path / "graphs.db")
    store.save(_sample_graph("first.md"))
    store.save(_sample_graph("second.md"))
    replacement = KnowledgeGraph(
        nodes=[Node(id="b", label="B", type="Concept")],
        edges=[],
        metadata=_metadata("first.md"),
    )

    store.save(replacement)

    assert store.load("first.md") == replacement
    assert store.load("second.md") == _sample_graph("second.md")


def test_document_graph_store_exists_delete_and_validation(tmp_path):
    store = SqliteGraphStore(tmp_path / "graphs.db")
    store.save(_sample_graph())

    assert store.exists("doc.md") is True
    assert store.exists("missing.md") is False
    with pytest.raises(ValueError, match="metadata.source"):
        store.save(KnowledgeGraph(nodes=[], edges=[]))

    store.delete("doc.md")

    assert store.exists("doc.md") is False
    with pytest.raises(FileNotFoundError):
        store.load("doc.md")


def test_knowledge_base_store_roundtrips_canonical_provenance(tmp_path):
    store = SqliteKnowledgeBaseStore(tmp_path / "knowledge.db")
    metadata = _metadata("docs/a.md")
    node = add_node_provenance(
        Node(id="a", label="A", type="Concept", aliases=["Concept A"]),
        metadata,
    )
    edge = add_edge_provenance(
        Edge(id="e1", source="a", target="a", label="relates_to"),
        metadata,
    )
    graph = KnowledgeGraph(nodes=[node], edges=[edge])

    store.replace_all(graph)

    assert store.exists() is True
    assert store.load() == graph


def test_knowledge_base_store_removes_only_requested_source_contributions(tmp_path):
    store = SqliteKnowledgeBaseStore(tmp_path / "knowledge.db")
    first = _metadata("docs/a.md")
    second = _metadata("docs/b.md")
    node = add_node_provenance(
        Node(id="a", label="A", type="Concept"),
        first,
    )
    node = node.model_copy(
        update={
            "provenance": [
                *node.provenance,
                add_node_provenance(
                    Node(id="b-a", label="A concept", type="Concept"),
                    second,
                ).provenance[0],
            ]
        }
    )
    from graphtool.graph.provenance import materialize_node

    node = materialize_node("a", node.provenance)
    store.replace_all(KnowledgeGraph(nodes=[node], edges=[]))

    node_ids, edge_ids = store.affected_ids(["docs/a.md"])
    remaining = store.load_excluding_sources(["docs/a.md"])
    store.apply_delta(
        KnowledgeBaseDelta(
            upserted_nodes=remaining.nodes,
            deleted_node_ids=node_ids - {node.id for node in remaining.nodes},
            upserted_edges=remaining.edges,
            deleted_edge_ids=edge_ids - {edge.id for edge in remaining.edges},
        )
    )

    loaded = store.load()
    assert len(loaded.nodes) == 1
    assert [item.source for item in loaded.nodes[0].provenance] == ["docs/b.md"]

    node_ids, edge_ids = store.affected_ids(["docs/b.md"])
    store.apply_delta(
        KnowledgeBaseDelta(
            upserted_nodes=[],
            deleted_node_ids=node_ids,
            upserted_edges=[],
            deleted_edge_ids=edge_ids,
        )
    )

    assert store.load().nodes == []


def test_knowledge_base_store_does_not_rewrite_unchanged_rows(tmp_path):
    conn = open_database(tmp_path / "knowledge.db")
    store = SqliteKnowledgeBaseStore(conn)
    graph = KnowledgeGraph(
        nodes=[Node(id="a", label="A", type="Concept")],
        edges=[],
    )
    store.replace_all(graph)
    changes_before = conn.total_changes

    store.replace_all(graph)

    assert conn.total_changes == changes_before


def test_knowledge_base_delta_does_not_scan_complete_tables(tmp_path):
    conn = open_database(tmp_path / "knowledge.db")
    store = SqliteKnowledgeBaseStore(conn)
    store.replace_all(
        KnowledgeGraph(
            nodes=[
                Node(id="a", label="A", type="Concept"),
                Node(id="b", label="B", type="Concept"),
            ],
            edges=[],
        )
    )
    statements = []
    conn.set_trace_callback(statements.append)

    store.apply_delta(
        KnowledgeBaseDelta(
            upserted_nodes=[Node(id="a", label="Updated A", type="Concept")],
            deleted_node_ids=set(),
            upserted_edges=[],
            deleted_edge_ids=set(),
        )
    )

    assert not any(
        statement.lstrip().upper().startswith("SELECT")
        for statement in statements
    )
    assert {node.id: node.label for node in store.load().nodes} == {
        "a": "Updated A",
        "b": "B",
    }


def test_shared_sqlite_store_transaction_rolls_back_all_graph_writes(tmp_path):
    conn = open_database(tmp_path / "knowledge.db")
    graph_store = SqliteGraphStore(conn)
    knowledge_store = SqliteKnowledgeBaseStore(conn)
    chunk_store = SqliteChunkStore(conn)

    with pytest.raises(RuntimeError, match="stop"):
        with graph_store.transaction():
            graph_store.save(_sample_graph())
            knowledge_store.replace_all(KnowledgeGraph(nodes=[], edges=[]))
            chunk_store.save(
                "doc.md",
                [
                    Chunk(
                        id="doc-chunk-0000",
                        source="doc.md",
                        index=0,
                        text="Text",
                    )
                ],
            )
            raise RuntimeError("stop")

    assert graph_store.exists("doc.md") is False
    assert knowledge_store.exists() is False
    assert chunk_store.load_all() == []


def test_embedding_store_roundtrips_and_updates_incrementally(tmp_path):
    conn = open_database(tmp_path / "embeddings.db")
    store = SqliteEmbeddingStore(conn)
    record = NodeEmbeddingRecord(
        node_id="openai",
        embedding_model="embedding-model",
        embedding_input_hash="hash",
        vector=[0.1, 0.2],
    )

    store.upsert({"openai": record})
    changes_before = conn.total_changes
    store.upsert({"openai": record})

    assert conn.total_changes == changes_before
    assert store.exists() is True
    loaded = store.load()["openai"]
    assert loaded.vector == pytest.approx(record.vector, abs=1e-6)


def test_graph_embedding_store_roundtrips_per_source(tmp_path):
    store = SqliteGraphEmbeddingStore(open_database(tmp_path / "embeddings.db"))
    record = NodeEmbeddingRecord(
        node_id="openai",
        embedding_model="embedding-model",
        embedding_input_hash="hash",
        vector=[0.1],
    )

    store.replace_source("docs/openai.md", {"openai": record})

    assert store.exists("docs/openai.md") is True
    assert store.exists("docs/missing.md") is False
    assert store.load("docs/openai.md")["openai"].vector == pytest.approx(
        record.vector,
        abs=1e-6,
    )

    store.delete("docs/openai.md")

    assert store.exists("docs/openai.md") is False
