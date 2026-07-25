import re
from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphtool.graph.types import KnowledgeGraph

UNCLASSIFIED_NODE_TYPE = "unclassified"

CanonicalNodeType = Literal[
    "concept",
    "feature",
    "capability",
    "tool",
    "integration",
    "product",
    "service",
    "organization",
    "person",
    "document",
    "repository",
    "package",
    "plugin",
    "model",
    "process",
    "system",
    "agent",
    "environment",
    "event_trigger",
    "resource",
    "unclassified",
]

CANONICAL_NODE_TYPES: tuple[str, ...] = (
    "concept",
    "feature",
    "capability",
    "tool",
    "integration",
    "product",
    "service",
    "organization",
    "person",
    "document",
    "repository",
    "package",
    "plugin",
    "model",
    "process",
    "system",
    "agent",
    "environment",
    "event_trigger",
    "resource",
    "unclassified",
)

DEFAULT_TAXONOMY_VERSION = 1
DEFAULT_PROMOTION_MIN_NODES = 5
DEFAULT_PROMOTION_MIN_SOURCES = 2


def normalize_type_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold())
    return "_".join(part for part in normalized.split("_") if part)


class TaxonomySuggestionStore(Protocol):
    def append_many(self, records: Sequence["TaxonomySuggestionRecord"]) -> None:
        ...


class NodeTypeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""


class NodeTypeRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = DEFAULT_TAXONOMY_VERSION
    types: dict[str, NodeTypeDefinition] = Field(
        default_factory=lambda: {
            type_name: NodeTypeDefinition()
            for type_name in CANONICAL_NODE_TYPES
        }
    )

    def with_promoted_types(
        self,
        promoted_types: Sequence[str],
    ) -> "NodeTypeRegistry":
        types = dict(self.types)
        for type_name in promoted_types:
            normalized = normalize_type_name(type_name)
            if normalized in types:
                continue
            types[normalized] = NodeTypeDefinition(
                description=f"Promoted from suggested type {normalized}."
            )
        return self.model_copy(update={"version": self.version + 1, "types": types})


class TaxonomySuggestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggested_type: str
    normalized_suggested_type: str
    node_id: str
    node_label: str
    current_type: str
    source: str
    chunk_id: str
    created_at: datetime


class TaxonomySuggestionAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_suggested_type: str
    suggested_types: list[str]
    node_count: int
    source_count: int
    node_ids: list[str]
    sources: list[str]


class TaxonomyPromotionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["promote_type"] = "promote_type"
    type: str
    matched_suggestions: list[str]
    affected_nodes: list[str]
    reason: str
    created_at: datetime


class TaxonomyEvolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry: NodeTypeRegistry
    graphs: list[KnowledgeGraph]
    promotions: list[TaxonomyPromotionRecord]


def default_node_type_registry() -> NodeTypeRegistry:
    return NodeTypeRegistry()


def canonical_node_type_text() -> str:
    return ", ".join(CANONICAL_NODE_TYPES)
