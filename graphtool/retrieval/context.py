import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.retrieval.references import format_source_location
from graphtool.retrieval.types import (
    ChunkHit,
    ChunkRelationship,
    GraphPathHit,
    SourceReference,
)


@dataclass(frozen=True)
class ChunkGraphIndex:
    nodes_by_chunk: dict[str, list[Node]]
    edges_by_chunk: dict[str, list[Edge]]


def build_chunk_graph_index(
    graph: KnowledgeGraph,
    chunks_by_id: dict[str, Chunk],
    nodes_by_id: dict[str, Node],
) -> ChunkGraphIndex:
    nodes_by_chunk: dict[str, list[Node]] = {}
    for node in sorted(graph.nodes, key=lambda item: item.id):
        for chunk_id in node.chunk_ids:
            if chunk_id in chunks_by_id:
                nodes_by_chunk.setdefault(chunk_id, []).append(node)

    edges_by_chunk: dict[str, list[Edge]] = {}
    for edge in sorted(graph.edges, key=lambda item: item.id):
        if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
            continue
        for chunk_id in edge.chunk_ids:
            if chunk_id in chunks_by_id:
                edges_by_chunk.setdefault(chunk_id, []).append(edge)

    return ChunkGraphIndex(
        nodes_by_chunk=nodes_by_chunk,
        edges_by_chunk=edges_by_chunk,
    )


def attach_graph_annotations(
    ranked_chunks: Sequence[tuple[Chunk, float]],
    index: ChunkGraphIndex,
    nodes_by_id: dict[str, Node],
) -> list[ChunkHit]:
    relationships_by_chunk: dict[str, list[ChunkRelationship]] = {}
    for chunk, _ in ranked_chunks:
        for edge in index.edges_by_chunk.get(chunk.id, []):
            relationship = ChunkRelationship(
                edge=edge,
                source_node=nodes_by_id[edge.source],
                target_node=nodes_by_id[edge.target],
            )
            relationships_by_chunk.setdefault(chunk.id, []).append(relationship)

    return [
        ChunkHit(
            chunk=chunk,
            score=score,
            relevance=score,
            linked_nodes=index.nodes_by_chunk.get(chunk.id, []),
            linked_relationships=relationships_by_chunk.get(chunk.id, []),
        )
        for chunk, score in ranked_chunks
    ]


def format_context(
    query: str,
    chunk_hits: Sequence[ChunkHit],
    graph_paths: Sequence[GraphPathHit] = (),
) -> str:
    lines = [f"Query: {query}", "", "Evidence:"]
    if not chunk_hits:
        lines.append("- None")
    else:
        for hit in chunk_hits:
            heading = " > ".join(hit.chunk.heading_path)
            metadata = f"{hit.chunk.id} | {hit.chunk.source}"
            page_reference = format_source_location(
                hit.chunk.source,
                hit.chunk.page_start,
                hit.chunk.page_end,
            )
            if page_reference:
                metadata = f"{metadata} | {page_reference}"
            if heading:
                metadata = f"{metadata} | {heading}"
            lines.extend([f"[{metadata}]", hit.chunk.text])

            if hit.linked_nodes:
                lines.append("Linked entities:")
                lines.extend(f"- {node_text(node)}" for node in hit.linked_nodes)

            if hit.linked_relationships:
                lines.append("Linked relationships:")
                lines.extend(
                    "- " + relationship_text(
                        relationship.edge,
                        relationship.source_node,
                        relationship.target_node,
                    )
                    for relationship in hit.linked_relationships
                )
            lines.append("")

    if graph_paths:
        lines.extend(["", "Graph paths:"])
        for path in graph_paths:
            lines.append(f"- {format_graph_path(path)}")
            if path.chunk_ids:
                lines.append(f"  Evidence chunks: {', '.join(path.chunk_ids)}")
    return "\n".join(lines).rstrip()


def source_references(chunk_hits: Sequence[ChunkHit]) -> list[SourceReference]:
    ranges_by_source: dict[str, list[tuple[int, int]]] = {}
    unpaged_sources = set()
    for hit in chunk_hits:
        source = hit.chunk.source
        ranges_by_source.setdefault(source, [])
        if hit.chunk.page_start is None:
            unpaged_sources.add(source)
        else:
            assert hit.chunk.page_end is not None
            ranges_by_source[source].append(
                (hit.chunk.page_start, hit.chunk.page_end)
            )

    references = []
    for source, ranges in ranges_by_source.items():
        if source in unpaged_sources:
            references.append(SourceReference(source=source))
            continue

        merged = []
        for page_start, page_end in sorted(ranges):
            if merged and page_start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], page_end))
            else:
                merged.append((page_start, page_end))
        references.extend(
            SourceReference(
                source=source,
                page_start=page_start,
                page_end=page_end,
            )
            for page_start, page_end in merged
        )
    return references


def node_text(node: Node) -> str:
    parts = [f"{node.label} [{node.type}]"]
    if node.aliases:
        parts.append(f"aliases: {', '.join(node.aliases)}")
    if node.suggested_type:
        parts.append(f"suggested type: {node.suggested_type}")
    if node.properties:
        parts.append(f"properties: {properties_text(node.properties)}")
    return " | ".join(parts)


def relationship_text(edge: Edge, source: Node, target: Node) -> str:
    text = f"{source.label} --{edge.label}--> {target.label}"
    if edge.properties:
        return f"{text} | properties: {properties_text(edge.properties)}"
    return text


def properties_text(properties: dict[str, Any]) -> str:
    return json.dumps(properties, sort_keys=True)


def format_graph_path(path: GraphPathHit) -> str:
    parts = [path.nodes[0].label]
    for index, edge in enumerate(path.edges):
        left = path.nodes[index]
        right = path.nodes[index + 1]
        if edge.source == left.id and edge.target == right.id:
            parts.append(f"--{edge.label}--> {right.label}")
        else:
            parts.append(f"<--{edge.label}-- {right.label}")
    return " ".join(parts)
