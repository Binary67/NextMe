import json
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from graphtool.graph.taxonomy_types import (
    UNCLASSIFIED_NODE_TYPE,
    normalize_type_name,
)
from graphtool.graph.types import Node
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage

ENTITY_RESOLUTION_SYSTEM_PROMPT = (
    "You decide whether a new knowledge graph node refers to the same real-world "
    "entity as one of the provided candidate nodes. Merge only when they are the "
    "same entity, not when they are merely related. Keep organizations, products, "
    "services, APIs, and deployments distinct unless the names clearly refer to "
    "the same entity."
)


class EntityResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["merge", "new"]
    target_node_id: str | None = None
    confidence: float = 0.0
    canonical_label: str | None = None
    aliases_to_add: list[str] = Field(default_factory=list)


class EntityCandidateIndex:
    def __init__(self) -> None:
        self._nodes_by_id: dict[str, Node] = {}
        self._rank_by_id: dict[str, int] = {}
        self._ids_by_name: dict[str, set[str]] = {}
        self._ids_by_type: dict[str, set[str]] = {}

    def add(self, node: Node) -> None:
        self._rank_by_id[node.id] = len(self._rank_by_id)
        self._nodes_by_id[node.id] = node
        self._add_to_indexes(node)

    def replace(self, node: Node) -> None:
        existing = self._nodes_by_id[node.id]
        self._remove_from_indexes(existing)
        self._nodes_by_id[node.id] = node
        self._add_to_indexes(node)

    def find_normalized_match(
        self,
        node: Node,
        candidate_ids: set[str] | None,
    ) -> Node | None:
        matching_ids: set[str] = set()
        for name in _normalized_names(node):
            matching_ids.update(self._ids_by_name.get(name, ()))
        if candidate_ids is not None:
            matching_ids.intersection_update(candidate_ids)
        matching_ids = {
            candidate_id
            for candidate_id in matching_ids
            if comparable_node_types(node, self._nodes_by_id[candidate_id])
        }
        if not matching_ids:
            return None
        match_id = min(matching_ids, key=self._rank_by_id.__getitem__)
        return self._nodes_by_id[match_id]

    def same_type_candidates(
        self,
        node: Node,
        candidate_ids: set[str] | None,
    ) -> list[Node]:
        node_type = normalize_type_name(node.type)
        if node_type == UNCLASSIFIED_NODE_TYPE:
            matching_ids = set(self._nodes_by_id)
        else:
            matching_ids = set(self._ids_by_type.get(node_type, ()))
            matching_ids.update(
                self._ids_by_type.get(UNCLASSIFIED_NODE_TYPE, ())
            )
        if candidate_ids is not None:
            matching_ids.intersection_update(candidate_ids)
        return [
            self._nodes_by_id[candidate_id]
            for candidate_id in sorted(
                matching_ids,
                key=self._rank_by_id.__getitem__,
            )
        ]

    def _add_to_indexes(self, node: Node) -> None:
        for name in _normalized_names(node):
            self._ids_by_name.setdefault(name, set()).add(node.id)
        node_type = normalize_type_name(node.type)
        self._ids_by_type.setdefault(node_type, set()).add(node.id)

    def _remove_from_indexes(self, node: Node) -> None:
        for name in _normalized_names(node):
            ids = self._ids_by_name[name]
            ids.remove(node.id)
            if not ids:
                del self._ids_by_name[name]
        node_type = normalize_type_name(node.type)
        ids = self._ids_by_type[node_type]
        ids.remove(node.id)
        if not ids:
            del self._ids_by_type[node_type]


def judge_same_entity(
    llm: LLMClient,
    node: Node,
    candidates: Sequence[tuple[Node, float]],
) -> EntityResolutionDecision:
    payload = {
        "incoming": _node_payload(node),
        "candidates": [
            {
                **_node_payload(candidate),
                "similarity": round(score, 6),
            }
            for candidate, score in candidates
        ],
        "rules": [
            "Return merge only if the incoming node and target candidate are the "
            "same entity.",
            "Do not merge an organization with its product, API, service, "
            "deployment, or partner.",
            "When uncertain, return new.",
        ],
    }
    messages = [
        LLMMessage(role="system", content=ENTITY_RESOLUTION_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=(
                "Resolve the incoming node against the candidates. "
                "Return the structured decision only.\n\n"
                f"{json.dumps(payload, indent=2, sort_keys=True)}"
            ),
        ),
    ]
    return llm.generate_structured(messages, EntityResolutionDecision)


def accepted_target_id(
    decision: EntityResolutionDecision,
    candidate_ids: set[str],
    merge_confidence_threshold: float,
) -> str | None:
    if decision.decision != "merge":
        return None
    if decision.target_node_id not in candidate_ids:
        return None
    if decision.confidence < merge_confidence_threshold:
        return None
    return decision.target_node_id


def comparable_node_types(left: Node, right: Node) -> bool:
    left_type = normalize_type_name(left.type)
    right_type = normalize_type_name(right.type)
    return (
        left_type == right_type
        or left_type == UNCLASSIFIED_NODE_TYPE
        or right_type == UNCLASSIFIED_NODE_TYPE
    )


def _normalized_names(node: Node) -> set[str]:
    return {
        normalized
        for normalized in (
            normalized_entity_name(name) for name in [node.label, *node.aliases]
        )
        if normalized
    }


def normalized_entity_name(value: str) -> str:
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


def _node_payload(node: Node) -> dict[str, object]:
    return {
        "id": node.id,
        "label": node.label,
        "type": node.type,
        "suggested_type": node.suggested_type,
        "aliases": node.aliases,
        "properties": node.properties,
    }
