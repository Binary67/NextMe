import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.retrieval.bm25 import BM25Document, BM25Index, tokenize
from graphtool.retrieval.scoring import (
    bm25_scores,
    normalize_scores,
    semantic_similarity_scores,
)
from graphtool.retrieval.types import GraphPathHit
from graphtool.sequences import unique_ordered

DEFAULT_TOP_SEEDS = 5
DEFAULT_BEAM_WIDTH = 50
PATH_HOP_DECAY = 0.85


@dataclass(frozen=True)
class PathCandidate:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    seed_score: float


def seed_node_scores(
    query: str,
    nodes: Sequence[Node],
    mention_scores: Mapping[str, float],
    label_index: BM25Index,
    metadata_index: BM25Index,
    query_vector: Sequence[float] | None,
    node_vectors: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    label_scores = bm25_scores(query, label_index)
    metadata_scores = bm25_scores(query, metadata_index)
    semantic_scores = semantic_similarity_scores(query_vector, node_vectors)
    return normalize_scores(
        {
            node.id: (
                label_scores.get(node.id, 0.0) * 2.0
                + mention_scores.get(node.id, 0.0) * 2.0
                + metadata_scores.get(node.id, 0.0)
                + semantic_scores.get(node.id, 0.0)
            )
            for node in nodes
        }
    )


def mentioned_node_scores(
    query: str,
    nodes: Sequence[Node],
) -> dict[str, float]:
    query_tokens = tokenize(query)
    matches: list[tuple[str, int, int]] = []
    for node in nodes:
        for name in [node.label, *node.aliases]:
            name_tokens = tokenize(name)
            if not name_tokens:
                continue
            for start in range(len(query_tokens) - len(name_tokens) + 1):
                if query_tokens[start : start + len(name_tokens)] == name_tokens:
                    matches.append((node.id, start, len(name_tokens)))

    scores = {}
    for node_id, start, length in matches:
        end = start + length
        contained_by_longer_match = any(
            other_start <= start
            and end <= other_start + other_length
            and other_length > length
            for _, other_start, other_length in matches
        )
        if not contained_by_longer_match:
            scores[node_id] = 1.0
    return scores


def node_search_text(node: Node) -> str:
    parts = [node.label, node.type, *node.aliases]
    if node.suggested_type:
        parts.append(node.suggested_type)
    if node.properties:
        parts.append(json.dumps(node.properties, sort_keys=True))
    return "\n".join(parts)


def traverse_paths(
    query: str,
    adjacency: Mapping[str, Sequence[tuple[Edge, str]]],
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
    seed_scores: Mapping[str, float],
    mention_scores: Mapping[str, float],
    *,
    max_hops: int,
    beam_width: int,
) -> list[PathCandidate]:
    if not seed_scores:
        return []

    seed_ids = sorted(
        seed_scores,
        key=lambda node_id: (-seed_scores[node_id], node_id),
    )[:DEFAULT_TOP_SEEDS]
    frontier = [
        PathCandidate(
            node_ids=(node_id,),
            edge_ids=(),
            seed_score=seed_scores[node_id],
        )
        for node_id in seed_ids
    ]
    candidates = []

    for _ in range(max_hops):
        expanded = []
        for candidate in frontier:
            current_node_id = candidate.node_ids[-1]
            for edge, neighbor_id in adjacency.get(current_node_id, []):
                if neighbor_id in candidate.node_ids:
                    continue
                expanded.append(
                    PathCandidate(
                        node_ids=(*candidate.node_ids, neighbor_id),
                        edge_ids=(*candidate.edge_ids, edge.id),
                        seed_score=candidate.seed_score,
                    )
                )

        if not expanded:
            break
        expanded = _deduplicate_candidates(expanded)
        ranked = rank_path_candidates(
            query,
            expanded,
            chunks_by_id,
            nodes_by_id,
            edges_by_id,
            seed_scores,
            mention_scores,
        )
        frontier = [candidate for candidate, _ in ranked[:beam_width]]
        candidates.extend(frontier)
    return _deduplicate_candidates(candidates)


def build_adjacency(
    graph: KnowledgeGraph,
) -> dict[str, list[tuple[Edge, str]]]:
    adjacency: dict[str, list[tuple[Edge, str]]] = {}
    for edge in sorted(graph.edges, key=lambda item: item.id):
        adjacency.setdefault(edge.source, []).append((edge, edge.target))
        if edge.target != edge.source:
            adjacency.setdefault(edge.target, []).append((edge, edge.source))
    return adjacency


def rank_path_candidates(
    query: str,
    candidates: Sequence[PathCandidate],
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
    node_scores: Mapping[str, float],
    mention_scores: Mapping[str, float],
) -> list[tuple[PathCandidate, float]]:
    if not candidates:
        return []

    path_documents = [
        BM25Document(
            id=str(index),
            text=_candidate_path_text(candidate, nodes_by_id, edges_by_id),
        )
        for index, candidate in enumerate(candidates)
    ]
    path_scores = normalize_scores(
        {
            document.id: score
            for document, score in BM25Index(path_documents).rank(query)
        }
    )
    evidence_documents = [
        BM25Document(
            id=str(index),
            text=_candidate_evidence_text(
                candidate,
                chunks_by_id,
                nodes_by_id,
                edges_by_id,
            ),
        )
        for index, candidate in enumerate(candidates)
    ]
    evidence_scores = normalize_scores(
        {
            document.id: score
            for document, score in BM25Index(evidence_documents).rank(query)
        }
    )
    scores = {
        str(index): (
            path_scores.get(str(index), 0.0)
            + evidence_scores.get(str(index), 0.0) * 0.5
            + _path_node_coverage(candidate, node_scores)
            + sum(
                mention_scores.get(node_id, 0.0)
                for node_id in candidate.node_ids
            )
        )
        * PATH_HOP_DECAY ** (len(candidate.edge_ids) - 1)
        for index, candidate in enumerate(candidates)
    }
    normalized_scores = normalize_scores(scores)
    ranked = [
        (candidate, normalized_scores.get(str(index), 0.0))
        for index, candidate in enumerate(candidates)
    ]
    ranked.sort(
        key=lambda item: (
            -item[1],
            len(item[0].edge_ids),
            item[0].node_ids,
            item[0].edge_ids,
        )
    )
    return ranked


def candidate_chunk_ids(
    candidate: PathCandidate,
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
) -> list[str]:
    edge_chunk_ids = unique_ordered(
        chunk_id
        for edge_id in candidate.edge_ids
        for chunk_id in edges_by_id[edge_id].chunk_ids
        if chunk_id in chunks_by_id
    )
    if edge_chunk_ids:
        return edge_chunk_ids
    return unique_ordered(
        chunk_id
        for node_id in candidate.node_ids
        for chunk_id in nodes_by_id[node_id].chunk_ids
        if chunk_id in chunks_by_id
    )


def rank_evidence_chunks(
    paths: Sequence[GraphPathHit],
    chunks_by_id: Mapping[str, Chunk],
) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    for rank, path in enumerate(paths, start=1):
        for chunk_id in path.chunk_ids:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / rank
    normalized_scores = normalize_scores(scores)
    ranked = [
        (chunks_by_id[chunk_id], score)
        for chunk_id, score in normalized_scores.items()
    ]
    ranked.sort(key=lambda item: (-item[1], item[0].index, item[0].id))
    return ranked


def _candidate_path_text(
    candidate: PathCandidate,
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
) -> str:
    lines = []
    for index, edge_id in enumerate(candidate.edge_ids):
        edge = edges_by_id[edge_id]
        left = nodes_by_id[candidate.node_ids[index]]
        right = nodes_by_id[candidate.node_ids[index + 1]]
        direction = edge.label if edge.source == left.id else f"inverse {edge.label}"
        lines.append(
            f"{node_search_text(left)}\n"
            f"{direction}\n"
            f"{node_search_text(right)}"
        )
    return "\n".join(lines)


def _candidate_evidence_text(
    candidate: PathCandidate,
    chunks_by_id: Mapping[str, Chunk],
    nodes_by_id: Mapping[str, Node],
    edges_by_id: Mapping[str, Edge],
) -> str:
    chunk_ids = candidate_chunk_ids(
        candidate,
        chunks_by_id,
        nodes_by_id,
        edges_by_id,
    )
    return "\n".join(chunks_by_id[chunk_id].text for chunk_id in chunk_ids)


def _path_node_coverage(
    candidate: PathCandidate,
    node_scores: Mapping[str, float],
) -> float:
    strongest_scores = sorted(
        (node_scores.get(node_id, 0.0) for node_id in candidate.node_ids),
        reverse=True,
    )[:2]
    return sum(strongest_scores) / 2.0


def _deduplicate_candidates(
    candidates: Sequence[PathCandidate],
) -> list[PathCandidate]:
    by_path: dict[tuple[str, ...], PathCandidate] = {}
    for candidate in candidates:
        reverse_edge_ids = tuple(reversed(candidate.edge_ids))
        key = min(candidate.edge_ids, reverse_edge_ids)
        existing = by_path.get(key)
        if existing is None or (
            candidate.seed_score,
            tuple(reversed(candidate.node_ids)),
        ) > (
            existing.seed_score,
            tuple(reversed(existing.node_ids)),
        ):
            by_path[key] = candidate
    return list(by_path.values())
