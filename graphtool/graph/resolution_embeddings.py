import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from graphtool.graph.embedding_store import NodeEmbeddingRecord
from graphtool.graph.taxonomy_types import normalize_type_name
from graphtool.graph.types import KnowledgeGraph, Node
from graphtool.llm.base import EmbeddingClient


class EmbeddingStore(Protocol):
    def load(self) -> dict[str, NodeEmbeddingRecord]:
        ...


class ResolutionEmbeddings:
    def __init__(
        self,
        client: EmbeddingClient,
        store: EmbeddingStore | None,
        reusable_records: Sequence[NodeEmbeddingRecord] = (),
    ) -> None:
        self._client = client
        self._store = store
        self._records = store.load() if store is not None else {}
        self._reusable_vectors = {
            (record.embedding_model, record.embedding_input_hash): record.vector
            for record in reusable_records
        }
        self._relationship_contexts: dict[str, list[str]] = {}
        self._prefetched_vectors: dict[tuple[str, str], list[float]] = {}
        self._node_inputs: dict[int, tuple[str, str]] = {}
        self._normalized_vectors: dict[tuple[str, str], np.ndarray] = {}
        self._candidate_matrices: dict[
            str,
            tuple[tuple[tuple[str, str], ...], np.ndarray],
        ] = {}

    def prepare(
        self,
        graphs: Sequence[KnowledgeGraph],
        incoming_nodes: Sequence[Node],
    ) -> None:
        self._relationship_contexts = _build_relationship_contexts(graphs)
        self._prefetched_vectors = dict(self._reusable_vectors)
        self._node_inputs = {}
        self._candidate_matrices = {}
        self._prefetch(incoming_nodes)

    def candidates(
        self,
        node: Node,
        canonical_nodes: Sequence[Node],
        *,
        min_similarity: float,
        top_k: int,
    ) -> list[tuple[Node, float]]:
        if not canonical_nodes:
            return []

        incoming_record = self._ensure([node])[0]
        candidate_records = self._ensure(canonical_nodes)
        incoming_vector = self._normalized_vector(incoming_record)
        matrix_key = tuple(
            (record.node_id, record.embedding_input_hash)
            for record in candidate_records
        )
        bucket = normalize_type_name(node.type)
        cached_matrix = self._candidate_matrices.get(bucket)
        if cached_matrix is None or cached_matrix[0] != matrix_key:
            candidate_vectors = np.stack(
                [
                    self._normalized_vector(record)
                    for record in candidate_records
                ]
            )
            self._candidate_matrices[bucket] = (matrix_key, candidate_vectors)
        else:
            candidate_vectors = cached_matrix[1]
        scores = candidate_vectors @ incoming_vector
        eligible = np.flatnonzero(scores >= min_similarity)
        if eligible.size > top_k:
            local = np.argpartition(scores[eligible], -top_k)[-top_k:]
            cutoff = scores[eligible[local]].min()
            eligible = eligible[scores[eligible] >= cutoff]
        scored = [
            (canonical_nodes[int(index)], float(scores[index]))
            for index in eligible
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0].id))[:top_k]

    def invalidate(self, node_id: str) -> None:
        record = self._records.pop(node_id, None)
        if record is not None:
            self._normalized_vectors.pop(
                (record.embedding_model, record.embedding_input_hash),
                None,
            )
        self._candidate_matrices = {}

    def finalize(self, canonical_nodes: Sequence[Node]) -> None:
        if self._store is None:
            return
        self._ensure(canonical_nodes)
        self._records = {
            node.id: self._records[node.id]
            for node in canonical_nodes
        }

    @property
    def records(self) -> dict[str, NodeEmbeddingRecord]:
        return dict(self._records)

    def _prefetch(self, nodes: Sequence[Node]) -> None:
        embedding_model = self._client.embedding_model
        missing_by_key: dict[tuple[str, str], str] = {}

        for node in nodes:
            text, text_hash = self._node_input(node)
            key = (embedding_model, text_hash)
            if key in self._prefetched_vectors:
                continue

            existing = self._records.get(node.id)
            if (
                existing is not None
                and existing.embedding_model == embedding_model
                and existing.embedding_input_hash == text_hash
            ):
                self._prefetched_vectors[key] = existing.vector
                continue
            missing_by_key.setdefault(key, text)

        if missing_by_key:
            keys = list(missing_by_key)
            vectors = self._client.embed_texts(
                [missing_by_key[key] for key in keys]
            )
            self._prefetched_vectors.update(zip(keys, vectors, strict=True))

    def _ensure(self, nodes: Sequence[Node]) -> list[NodeEmbeddingRecord]:
        records: list[NodeEmbeddingRecord | None] = []
        missing: list[tuple[int, Node, str, str]] = []
        embedding_model = self._client.embedding_model

        for node in nodes:
            text, text_hash = self._node_input(node)
            existing = self._records.get(node.id)
            if (
                existing is not None
                and existing.embedding_model == embedding_model
                and existing.embedding_input_hash == text_hash
            ):
                records.append(existing)
                continue

            prefetched = self._prefetched_vectors.get((embedding_model, text_hash))
            if prefetched is not None:
                record = NodeEmbeddingRecord(
                    node_id=node.id,
                    embedding_model=embedding_model,
                    embedding_input_hash=text_hash,
                    vector=prefetched,
                )
                self._records[node.id] = record
                records.append(record)
                continue

            records.append(None)
            missing.append((len(records) - 1, node, text_hash, text))

        if missing:
            vectors = self._client.embed_texts(
                [text for _, _, _, text in missing]
            )
            for (index, node, text_hash, _), vector in zip(
                missing,
                vectors,
                strict=True,
            ):
                record = NodeEmbeddingRecord(
                    node_id=node.id,
                    embedding_model=embedding_model,
                    embedding_input_hash=text_hash,
                    vector=vector,
                )
                self._records[node.id] = record
                records[index] = record

        return [record for record in records if record is not None]

    def _node_input(self, node: Node) -> tuple[str, str]:
        key = id(node)
        cached = self._node_inputs.get(key)
        if cached is not None:
            return cached
        text = node_embedding_text(
            node,
            self._relationship_contexts.get(node.id, []),
        )
        value = (text, embedding_input_hash(text))
        self._node_inputs[key] = value
        return value

    def _normalized_vector(self, record: NodeEmbeddingRecord) -> np.ndarray:
        key = (record.embedding_model, record.embedding_input_hash)
        cached = self._normalized_vectors.get(key)
        if cached is not None:
            return cached
        vector = np.asarray(record.vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        normalized = vector if norm == 0.0 else vector / norm
        self._normalized_vectors[key] = normalized
        return normalized


def node_embedding_text(
    node: Node,
    relationship_context: Sequence[str] = (),
) -> str:
    parts = [f"label: {node.label}", f"type: {node.type}"]
    if node.suggested_type:
        parts.append(f"suggested_type: {node.suggested_type}")
    if node.aliases:
        parts.append(f"aliases: {', '.join(node.aliases)}")
    if node.properties:
        parts.append(f"properties: {json.dumps(node.properties, sort_keys=True)}")
    if relationship_context:
        relationships = "; ".join(_extend_unique([], relationship_context))
        parts.append(f"relationships: {relationships}")
    return "\n".join(parts)


def embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_relationship_contexts(
    graphs: Sequence[KnowledgeGraph],
) -> dict[str, list[str]]:
    contexts: dict[str, list[str]] = {}
    for graph in graphs:
        nodes_by_id = {node.id: node for node in graph.nodes}
        for edge in graph.edges:
            source = nodes_by_id.get(edge.source)
            target = nodes_by_id.get(edge.target)
            if source is not None and target is not None:
                contexts.setdefault(source.id, []).append(
                    f"outgoing {edge.label} to {target.label} ({target.type})"
                )
                contexts.setdefault(target.id, []).append(
                    f"incoming {edge.label} from {source.label} ({source.type})"
                )
    return contexts


def _extend_unique(
    values: list[str],
    additions: Sequence[str],
) -> list[str]:
    merged = list(values)
    for addition in additions:
        if addition not in merged:
            merged.append(addition)
    return merged
