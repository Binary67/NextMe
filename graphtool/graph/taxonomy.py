from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from graphtool.graph.taxonomy_stores import (
    JsonNodeTypeRegistryStore,
    JsonTaxonomyPromotionAuditStore,
    SqliteTaxonomySuggestionStore,
)
from graphtool.graph.taxonomy_types import (
    DEFAULT_PROMOTION_MIN_NODES,
    DEFAULT_PROMOTION_MIN_SOURCES,
    UNCLASSIFIED_NODE_TYPE,
    NodeTypeRegistry,
    TaxonomyEvolutionResult,
    TaxonomyPromotionRecord,
    TaxonomySuggestionAggregate,
    TaxonomySuggestionRecord,
    normalize_type_name,
)
from graphtool.graph.types import KnowledgeGraph
from graphtool.sequences import unique_ordered


def make_taxonomy_suggestion_records(
    *,
    nodes: Sequence[object],
    source: str,
    chunk_id: str,
    created_at: datetime | None = None,
) -> list[TaxonomySuggestionRecord]:
    timestamp = created_at or datetime.now(timezone.utc)
    records = []
    for node in nodes:
        suggested_type = getattr(node, "suggested_type", None)
        if suggested_type is None:
            continue
        normalized = normalize_type_name(suggested_type)
        if not normalized:
            continue
        records.append(
            TaxonomySuggestionRecord(
                suggested_type=suggested_type,
                normalized_suggested_type=normalized,
                node_id=getattr(node, "id"),
                node_label=getattr(node, "label"),
                current_type=getattr(node, "type"),
                source=source,
                chunk_id=chunk_id,
                created_at=timestamp,
            )
        )
    return records


def aggregate_suggestions(
    records: Sequence[TaxonomySuggestionRecord],
) -> list[TaxonomySuggestionAggregate]:
    grouped: dict[str, list[TaxonomySuggestionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.normalized_suggested_type].append(record)

    aggregates = []
    for normalized, group in sorted(grouped.items()):
        node_keys = unique_ordered(
            _node_key(record.source, record.node_id)
            for record in group
        )
        aggregates.append(
            TaxonomySuggestionAggregate(
                normalized_suggested_type=normalized,
                suggested_types=unique_ordered(
                    record.suggested_type for record in group
                ),
                node_count=len(node_keys),
                source_count=len({record.source for record in group}),
                node_ids=node_keys,
                sources=unique_ordered(record.source for record in group),
            )
        )
    return aggregates


def evolve_taxonomy(
    registry_store: JsonNodeTypeRegistryStore,
    suggestion_store: SqliteTaxonomySuggestionStore,
    audit_store: JsonTaxonomyPromotionAuditStore,
    graphs: Sequence[KnowledgeGraph],
    *,
    min_nodes: int = DEFAULT_PROMOTION_MIN_NODES,
    min_sources: int = DEFAULT_PROMOTION_MIN_SOURCES,
) -> TaxonomyEvolutionResult:
    result = promote_suggestions(
        registry_store.load_or_default(),
        suggestion_store.load(),
        graphs,
        min_nodes=min_nodes,
        min_sources=min_sources,
    )
    registry_store.save(result.registry)
    audit_store.append_many(result.promotions)
    return result


def promote_suggestions(
    registry: NodeTypeRegistry,
    records: Sequence[TaxonomySuggestionRecord],
    graphs: Sequence[KnowledgeGraph],
    *,
    min_nodes: int = DEFAULT_PROMOTION_MIN_NODES,
    min_sources: int = DEFAULT_PROMOTION_MIN_SOURCES,
    created_at: datetime | None = None,
) -> TaxonomyEvolutionResult:
    timestamp = created_at or datetime.now(timezone.utc)
    existing_types = {normalize_type_name(type_name) for type_name in registry.types}
    promoted_types = []
    promotions = []

    for aggregate in aggregate_suggestions(records):
        promoted_type = aggregate.normalized_suggested_type
        if promoted_type in existing_types:
            continue
        if aggregate.node_count < min_nodes:
            continue
        if aggregate.source_count < min_sources:
            continue

        promoted_types.append(promoted_type)
        existing_types.add(promoted_type)
        promotions.append(
            TaxonomyPromotionRecord(
                type=promoted_type,
                matched_suggestions=aggregate.suggested_types,
                affected_nodes=[],
                reason=(
                    f"Appeared in {aggregate.node_count} nodes across "
                    f"{aggregate.source_count} source documents."
                ),
                created_at=timestamp,
            )
        )

    promoted_lookup = set(promoted_types)
    migrated_graphs = [
        migrate_promoted_types(graph, promoted_lookup)
        for graph in graphs
    ]
    affected_by_type = _affected_nodes_by_type(graphs, migrated_graphs, promoted_lookup)
    promotions = [
        promotion.model_copy(
            update={"affected_nodes": affected_by_type.get(promotion.type, [])}
        )
        for promotion in promotions
    ]

    registry = (
        registry.with_promoted_types(promoted_types)
        if promoted_types
        else registry
    )
    return TaxonomyEvolutionResult(
        registry=registry,
        graphs=migrated_graphs,
        promotions=promotions,
    )


def migrate_promoted_types(
    graph: KnowledgeGraph,
    promoted_types: set[str],
) -> KnowledgeGraph:
    nodes = []
    for node in graph.nodes:
        suggested_type = node.suggested_type
        should_migrate = (
            normalize_type_name(node.type) == UNCLASSIFIED_NODE_TYPE
            and suggested_type is not None
            and normalize_type_name(suggested_type) in promoted_types
        )
        if not should_migrate:
            nodes.append(node)
            continue

        nodes.append(
            node.model_copy(
                update={
                    "type": normalize_type_name(suggested_type),
                    "suggested_type": None,
                }
            )
        )
    return graph.model_copy(update={"nodes": nodes})


def _affected_nodes_by_type(
    original_graphs: Sequence[KnowledgeGraph],
    migrated_graphs: Sequence[KnowledgeGraph],
    promoted_types: set[str],
) -> dict[str, list[str]]:
    affected: dict[str, list[str]] = {type_name: [] for type_name in promoted_types}
    for original, migrated in zip(original_graphs, migrated_graphs, strict=True):
        source = original.metadata.source if original.metadata is not None else ""
        for original_node, migrated_node in zip(
            original.nodes,
            migrated.nodes,
            strict=True,
        ):
            if original_node.type == migrated_node.type:
                continue
            promoted_type = normalize_type_name(migrated_node.type)
            if promoted_type in affected:
                affected[promoted_type].append(
                    _node_key(source, migrated_node.id)
                    if source
                    else migrated_node.id
                )
    return {
        type_name: unique_ordered(nodes)
        for type_name, nodes in affected.items()
    }


def _node_key(source: str, node_id: str) -> str:
    return f"{source}:{node_id}"
