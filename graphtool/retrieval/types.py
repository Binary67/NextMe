from pydantic import BaseModel, Field, model_validator

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, Node


class ChunkRelationship(BaseModel):
    edge: Edge
    source_node: Node
    target_node: Node


class ChunkHit(BaseModel):
    # score orders hits within one result and is only comparable inside it.
    # relevance is calibrated against fixed reference points, so it stays
    # comparable across queries and is what absolute cutoffs apply to. Only
    # hybrid retrieval resolves both; single-branch retrieval reports its own
    # ranking value as each.
    chunk: Chunk
    score: float
    relevance: float
    linked_nodes: list[Node]
    linked_relationships: list[ChunkRelationship]


class GraphPathHit(BaseModel):
    score: float
    nodes: list[Node]
    edges: list[Edge]
    chunk_ids: list[str]


class SourceReference(BaseModel):
    source: str
    page_start: int | None = None
    page_end: int | None = None

    @model_validator(mode="after")
    def validate_page_range(self) -> "SourceReference":
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError(
                "page_start and page_end must both be set or both be omitted"
            )
        if self.page_start is not None:
            assert self.page_end is not None
            if self.page_start < 1:
                raise ValueError("page_start must be positive")
            if self.page_end < self.page_start:
                raise ValueError(
                    "page_end must be greater than or equal to page_start"
                )
        return self


class RetrievalResult(BaseModel):
    query: str
    sources: list[str]
    references: list[SourceReference] = Field(default_factory=list)
    chunks: list[ChunkHit]
    graph_paths: list[GraphPathHit] = Field(default_factory=list)
    context_text: str
