from datetime import datetime, timezone

from graphtool.graph.taxonomy import evolve_taxonomy, promote_suggestions
from graphtool.graph.taxonomy_stores import (
    JsonNodeTypeRegistryStore,
    JsonTaxonomyPromotionAuditStore,
    SqliteTaxonomySuggestionStore,
)
from graphtool.graph.taxonomy_types import (
    TaxonomySuggestionRecord,
    default_node_type_registry,
)
from graphtool.graph.types import GraphMetadata, KnowledgeGraph, Node
from graphtool.storage import open_database


def _record(
    source: str,
    node_id: str,
    suggested_type: str = "distribution channel",
) -> TaxonomySuggestionRecord:
    return TaxonomySuggestionRecord(
        suggested_type=suggested_type,
        normalized_suggested_type="distribution_channel",
        node_id=node_id,
        node_label=node_id.title(),
        current_type="unclassified",
        source=source,
        chunk_id=f"{node_id}-chunk-0000",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _graph(source: str, node_ids: list[str]) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id=node_id,
                label=node_id.title(),
                type="unclassified",
                suggested_type="distribution channel",
            )
            for node_id in node_ids
        ],
        edges=[],
        metadata=GraphMetadata(
            source=source,
            content_hash="hash",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_promote_suggestions_updates_registry_and_migrates_unclassified_nodes():
    records = [
        _record("docs/one.md", "marketplace"),
        _record("docs/one.md", "registry"),
        _record("docs/one.md", "directory"),
        _record("docs/two.md", "plugin-store", "distribution-channel"),
        _record("docs/two.md", "package-index"),
    ]
    graphs = [
        _graph("docs/one.md", ["marketplace", "registry", "directory"]),
        _graph("docs/two.md", ["plugin-store", "package-index"]),
    ]

    result = promote_suggestions(
        default_node_type_registry(),
        records,
        graphs,
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert "distribution_channel" in result.registry.types
    assert all(
        node.type == "distribution_channel" and node.suggested_type is None
        for graph in result.graphs
        for node in graph.nodes
    )
    assert len(result.promotions) == 1
    promotion = result.promotions[0]
    assert promotion.type == "distribution_channel"
    assert promotion.matched_suggestions == [
        "distribution channel",
        "distribution-channel",
    ]
    assert promotion.affected_nodes == [
        "docs/one.md:marketplace",
        "docs/one.md:registry",
        "docs/one.md:directory",
        "docs/two.md:plugin-store",
        "docs/two.md:package-index",
    ]
    assert "5 nodes across 2 source documents" in promotion.reason


def test_promote_suggestions_keeps_small_suggestions_pending():
    result = promote_suggestions(
        default_node_type_registry(),
        [
            _record("docs/one.md", "marketplace"),
            _record("docs/two.md", "plugin-store"),
        ],
        [_graph("docs/one.md", ["marketplace"])],
    )

    assert "distribution_channel" not in result.registry.types
    assert result.graphs[0].nodes[0].type == "unclassified"
    assert result.graphs[0].nodes[0].suggested_type == "distribution channel"
    assert result.promotions == []


def test_evolve_taxonomy_uses_backed_stores(tmp_path):
    registry_store = JsonNodeTypeRegistryStore(tmp_path / "registry.json")
    suggestion_store = SqliteTaxonomySuggestionStore(
        open_database(tmp_path / "suggestions.db")
    )
    audit_store = JsonTaxonomyPromotionAuditStore(tmp_path / "promotions.json")
    suggestion_store.append_many(
        [
            _record("docs/one.md", "marketplace"),
            _record("docs/one.md", "registry"),
            _record("docs/one.md", "directory"),
            _record("docs/two.md", "plugin-store"),
            _record("docs/two.md", "package-index"),
        ]
    )

    result = evolve_taxonomy(
        registry_store,
        suggestion_store,
        audit_store,
        [_graph("docs/one.md", ["marketplace"])],
    )

    assert registry_store.exists() is True
    assert audit_store.exists() is True
    assert "distribution_channel" in registry_store.load().types
    assert audit_store.load() == result.promotions
