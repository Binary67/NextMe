from datetime import datetime, timezone

import pytest

from graphtool.graph.document_store import SqliteGraphStore as JsonGraphStore
from graphtool.graph.knowledge_base_store import (
    SqliteKnowledgeBaseStore as JsonKnowledgeBaseStore,
)
from graphtool.graph.types import GraphMetadata, KnowledgeGraph, Node
from graphtool.source import source_key
from graphtool.visualization import export_knowledge_base_visualizations


class FakeGraphStore:
    def __init__(self, graphs: list[KnowledgeGraph]) -> None:
        self.graphs = graphs

    def load_all(self) -> list[KnowledgeGraph]:
        return self.graphs


def _graph(source: str, node_id: str, label: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id=node_id,
                label=label,
                type="Concept",
                chunk_ids=[f"{node_id}-chunk-0000"],
            )
        ],
        edges=[],
        metadata=GraphMetadata(
            source=source,
            content_hash="hash",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_export_knowledge_base_visualizations_writes_documents_and_combined_graph(
    tmp_path,
):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    graph_store.save(_graph("documents/a.md", "alpha", "Alpha"))
    graph_store.save(_graph("documents/b.md", "beta", "Beta"))
    output_dir = tmp_path / "visualizations"

    paths = export_knowledge_base_visualizations(graph_store, output_dir)

    expected_paths = [
        (
            output_dir
            / "documents"
            / f"{source_key('documents/a.md')}.html"
        ).resolve(),
        (
            output_dir
            / "documents"
            / f"{source_key('documents/b.md')}.html"
        ).resolve(),
        (output_dir / "knowledge_graph.html").resolve(),
    ]
    assert paths == expected_paths
    for path in expected_paths:
        assert path.exists()

    combined_html = expected_paths[-1].read_text()
    assert "Alpha" in combined_html
    assert "Beta" in combined_html


def test_export_knowledge_base_visualizations_uses_cached_combined_graph(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    graph_store.save(_graph("documents/a.md", "alpha", "Alpha"))
    knowledge_base_store = JsonKnowledgeBaseStore(tmp_path / "knowledge_base.json")
    knowledge_base_store.replace_all(
        KnowledgeGraph(
            nodes=[Node(id="cached", label="Cached Only", type="Concept")],
            edges=[],
        )
    )
    output_dir = tmp_path / "visualizations"

    paths = export_knowledge_base_visualizations(
        graph_store,
        output_dir,
        knowledge_base_store=knowledge_base_store,
    )

    combined_html = paths[-1].read_text()
    assert "Cached Only" in combined_html
    assert '"id": "alpha"' not in combined_html


def test_export_knowledge_base_visualizations_removes_deleted_document_html(tmp_path):
    graph_store = JsonGraphStore(tmp_path / "graphs")
    deleted_source = "documents/deleted.md"
    graph_store.save(_graph(deleted_source, "deleted", "Deleted"))
    output_dir = tmp_path / "visualizations"
    paths = export_knowledge_base_visualizations(graph_store, output_dir)
    deleted_path = paths[0]
    assert deleted_path.exists()

    graph_store.delete(deleted_source)
    export_knowledge_base_visualizations(graph_store, output_dir)

    assert deleted_path.exists() is False


def test_export_knowledge_base_visualizations_raises_for_missing_metadata(tmp_path):
    graph_store = FakeGraphStore([KnowledgeGraph(nodes=[], edges=[])])

    with pytest.raises(
        ValueError,
        match="Cannot visualize graph without metadata.source.",
    ):
        export_knowledge_base_visualizations(graph_store, tmp_path)
