import re
from collections.abc import Sequence

from graphtool.graph.taxonomy_types import (
    UNCLASSIFIED_NODE_TYPE,
    normalize_type_name,
)
from graphtool.graph.types import (
    Edge,
    EdgeProvenance,
    GraphMetadata,
    KnowledgeGraph,
    Node,
    NodeProvenance,
)


def add_node_provenance(
    node: Node,
    metadata: GraphMetadata | None,
    resolution_aliases: Sequence[str] = (),
) -> Node:
    if metadata is None:
        return node
    provenance = NodeProvenance(
        source=metadata.source,
        content_hash=metadata.content_hash,
        node_id=node.id,
        label=node.label,
        type=node.type,
        suggested_type=node.suggested_type,
        aliases=list(node.aliases),
        properties=dict(node.properties),
        chunk_ids=list(node.chunk_ids),
        resolution_aliases=list(resolution_aliases),
    )
    return materialize_node(node.id, [provenance])


def add_edge_provenance(edge: Edge, metadata: GraphMetadata | None) -> Edge:
    if metadata is None:
        return edge
    provenance = EdgeProvenance(
        source=metadata.source,
        content_hash=metadata.content_hash,
        edge_id=edge.id,
        source_node_id=edge.source,
        target_node_id=edge.target,
        label=edge.label,
        properties=dict(edge.properties),
        chunk_ids=list(edge.chunk_ids),
    )
    return materialize_edge(edge.id, edge.source, edge.target, [provenance])


def canonicalize_node(node: Node) -> Node:
    if node.provenance:
        return materialize_node(node.id, node.provenance)
    return node.model_copy(
        update={
            "aliases": _unique_aliases(node.label, node.aliases),
            "chunk_ids": _extend_unique([], node.chunk_ids),
        }
    )


def merge_nodes(
    existing: Node,
    incoming: Node,
    resolution_aliases: Sequence[str] = (),
) -> Node:
    if existing.provenance or incoming.provenance:
        incoming_provenance = [
            provenance.model_copy(
                update={
                    "resolution_aliases": _extend_unique(
                        provenance.resolution_aliases,
                        resolution_aliases,
                    )
                }
            )
            for provenance in incoming.provenance
        ]
        return materialize_node(
            existing.id,
            [*existing.provenance, *incoming_provenance],
        )

    alias_additions = []
    if _normalize_name(incoming.label) != _normalize_name(existing.label):
        alias_additions.append(incoming.label)
    alias_additions.extend(incoming.aliases)
    alias_additions.extend(resolution_aliases)
    return existing.model_copy(
        update={
            "type": _merge_node_type(existing.type, incoming.type),
            "suggested_type": existing.suggested_type or incoming.suggested_type,
            "aliases": _unique_aliases(
                existing.label,
                [*existing.aliases, *alias_additions],
            ),
            "chunk_ids": _extend_unique(existing.chunk_ids, incoming.chunk_ids),
        }
    )


def merge_edges(existing: Edge, incoming: Edge) -> Edge:
    if existing.provenance or incoming.provenance:
        return materialize_edge(
            existing.id,
            existing.source,
            existing.target,
            [*existing.provenance, *incoming.provenance],
        )
    return existing.model_copy(
        update={
            "chunk_ids": _extend_unique(existing.chunk_ids, incoming.chunk_ids),
        }
    )


def remove_source_from_knowledge_graph(
    graph: KnowledgeGraph,
    source: str,
) -> KnowledgeGraph:
    nodes = []
    for node in graph.nodes:
        if all(item.source != source for item in node.provenance):
            nodes.append(node)
            continue
        provenance = [item for item in node.provenance if item.source != source]
        if provenance:
            nodes.append(materialize_node(node.id, provenance))

    node_ids = {node.id for node in nodes}
    edges = []
    for edge in graph.edges:
        if (
            all(item.source != source for item in edge.provenance)
            and edge.source in node_ids
            and edge.target in node_ids
        ):
            edges.append(edge)
            continue
        provenance = [item for item in edge.provenance if item.source != source]
        if provenance and edge.source in node_ids and edge.target in node_ids:
            edges.append(
                materialize_edge(
                    edge.id,
                    edge.source,
                    edge.target,
                    provenance,
                )
            )

    return KnowledgeGraph(nodes=nodes, edges=edges)


def filter_knowledge_graph_by_source(
    graph: KnowledgeGraph,
    source: str,
) -> KnowledgeGraph:
    return filter_knowledge_graph_by_sources(graph, [source])


def filter_knowledge_graph_by_sources(
    graph: KnowledgeGraph,
    sources: Sequence[str],
) -> KnowledgeGraph:
    allowed_sources = set(sources)
    nodes = []
    for node in graph.nodes:
        provenance = [
            item for item in node.provenance if item.source in allowed_sources
        ]
        if provenance:
            nodes.append(materialize_node(node.id, provenance))

    node_ids = {node.id for node in nodes}
    edges = []
    for edge in graph.edges:
        provenance = [
            item for item in edge.provenance if item.source in allowed_sources
        ]
        if provenance and edge.source in node_ids and edge.target in node_ids:
            edges.append(
                materialize_edge(
                    edge.id,
                    edge.source,
                    edge.target,
                    provenance,
                )
            )

    return KnowledgeGraph(nodes=nodes, edges=edges)


def materialize_node(
    node_id: str,
    provenance: Sequence[NodeProvenance],
) -> Node:
    first = provenance[0]
    node_type = first.type
    suggested_type = first.suggested_type
    aliases: list[str] = []
    chunk_ids: list[str] = []
    for item in provenance:
        node_type = _merge_node_type(node_type, item.type)
        suggested_type = suggested_type or item.suggested_type
        if _normalize_name(item.label) != _normalize_name(first.label):
            aliases.append(item.label)
        aliases.extend(item.aliases)
        aliases.extend(item.resolution_aliases)
        chunk_ids = _extend_unique(chunk_ids, item.chunk_ids)

    return Node(
        id=node_id,
        label=first.label,
        type=node_type,
        suggested_type=suggested_type,
        aliases=_unique_aliases(first.label, aliases),
        properties=dict(first.properties),
        chunk_ids=chunk_ids,
        provenance=list(provenance),
    )


def materialize_edge(
    edge_id: str,
    source: str,
    target: str,
    provenance: Sequence[EdgeProvenance],
) -> Edge:
    first = provenance[0]
    chunk_ids: list[str] = []
    for item in provenance:
        chunk_ids = _extend_unique(chunk_ids, item.chunk_ids)
    return Edge(
        id=edge_id,
        source=source,
        target=target,
        label=first.label,
        properties=dict(first.properties),
        chunk_ids=chunk_ids,
        provenance=list(provenance),
    )


def _merge_node_type(existing: str, incoming: str) -> str:
    existing_type = normalize_type_name(existing)
    incoming_type = normalize_type_name(incoming)
    if (
        existing_type == UNCLASSIFIED_NODE_TYPE
        and incoming_type != UNCLASSIFIED_NODE_TYPE
    ):
        return incoming
    return existing


def _unique_aliases(label: str, aliases: Sequence[str]) -> list[str]:
    label_key = _normalize_name(label)
    seen = {label_key}
    unique = []
    for alias in aliases:
        normalized = _normalize_name(alias)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(alias)
    return unique


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(_singularize_word(word) for word in normalized.split())


def _singularize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if (
        len(word) > 4
        and word.endswith(("ches", "shes", "sses", "xes", "zes"))
    ):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _extend_unique(values: Sequence[str], additions: Sequence[str]) -> list[str]:
    merged = list(values)
    for addition in additions:
        if addition not in merged:
            merged.append(addition)
    return merged
