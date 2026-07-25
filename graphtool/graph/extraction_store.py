import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphtool.graph.taxonomy_types import (
    CanonicalNodeType,
    UNCLASSIFIED_NODE_TYPE,
)
from graphtool.source import source_key


class ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    label: str
    type: CanonicalNodeType
    suggested_type: str | None = None

    @model_validator(mode="after")
    def validate_suggested_type(self) -> "ExtractedNode":
        if self.type == UNCLASSIFIED_NODE_TYPE and not self.suggested_type:
            raise ValueError("suggested_type is required for unclassified nodes")
        if self.suggested_type is not None and not self.suggested_type.strip():
            raise ValueError("suggested_type cannot be blank")
        return self


class ExtractedEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_ref: str
    target_ref: str
    label: str


class ExtractedKnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[ExtractedNode]
    edges: list[ExtractedEdge]

    @model_validator(mode="after")
    def validate_unique_node_refs(self) -> "ExtractedKnowledgeGraph":
        seen = set()
        duplicate_refs = []
        for node in self.nodes:
            if node.ref in seen and node.ref not in duplicate_refs:
                duplicate_refs.append(node.ref)
            seen.add(node.ref)

        if duplicate_refs:
            joined = ", ".join(repr(node_ref) for node_ref in duplicate_refs)
            raise ValueError(f"extracted node refs must be unique: {joined}")
        return self


class _ChunkExtractionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: dict[str, ExtractedKnowledgeGraph] = Field(default_factory=dict)


class ChunkExtractionStore(Protocol):
    def load(self, source: str) -> dict[str, ExtractedKnowledgeGraph]:
        ...

    def replace(
        self,
        source: str,
        records: Mapping[str, ExtractedKnowledgeGraph],
    ) -> None:
        ...

    def delete(self, source: str) -> None:
        ...


class JsonChunkExtractionStore:
    """Filesystem-backed raw chunk extraction cache."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def load(self, source: str) -> dict[str, ExtractedKnowledgeGraph]:
        path = self._path_for(source)
        if not path.exists():
            return {}

        manifest = _ChunkExtractionManifest.model_validate_json(path.read_text())
        return dict(manifest.records)

    def replace(
        self,
        source: str,
        records: Mapping[str, ExtractedKnowledgeGraph],
    ) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        data = {
            "records": {
                key: record.model_dump(mode="json")
                for key, record in sorted(records.items())
            }
        }
        self._path_for(source).write_text(json.dumps(data, indent=2))

    def delete(self, source: str) -> None:
        self._path_for(source).unlink(missing_ok=True)

    def _path_for(self, source: str) -> Path:
        return self._directory / f"{source_key(source)}.json"
