from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, ConfigDict, Field

from graphtool.agents.knowledge.state import AgentChunkReference, AgentState
from graphtool.chunking.types import Chunk
from graphtool.retrieval import SourceReference
from graphtool.retrieval.context import format_context, format_graph_path
from graphtool.runtime import GraphToolRuntime


class SearchEvidenceChunk(AgentChunkReference):
    context_text: str


class SearchEvidencePath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_ids: list[str]
    edge_ids: list[str]
    chunk_ids: list[str]
    context_text: str


class KnowledgeSearchArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["search"] = "search"
    query: str
    context_text: str
    references: list[SourceReference] = Field(default_factory=list)
    chunks: list[SearchEvidenceChunk] = Field(default_factory=list)
    graph_paths: list[SearchEvidencePath] = Field(default_factory=list)


class NeighborhoodChunk(AgentChunkReference):
    text: str


class ChunkNeighborhoodArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["chunk_neighborhood"] = "chunk_neighborhood"
    source: str
    chunk_id: str
    context_text: str
    references: list[SourceReference] = Field(default_factory=list)
    previous: NeighborhoodChunk | None = None
    current: NeighborhoodChunk
    next: NeighborhoodChunk | None = None


class ToolErrorArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    tool_name: str
    message: str


def create_knowledge_tools(runtime: GraphToolRuntime) -> list[BaseTool]:
    @tool(response_format="content_and_artifact")
    def search_knowledge_base(
        query: str,
        state: Annotated[AgentState, InjectedState],
    ) -> tuple[str, KnowledgeSearchArtifact]:
        """Search document chunks and knowledge-graph paths for one focused query."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Knowledge base search query must not be empty.")

        scope = state.get("knowledge_scope")
        result = (
            runtime.search(normalized_query, scope=scope)
            if scope is not None
            else runtime.search(normalized_query)
        )
        chunks = [
            SearchEvidenceChunk(
                **_chunk_reference(hit.chunk).model_dump(),
                context_text=format_context(normalized_query, [hit]),
            )
            for hit in result.chunks
        ]
        returned_chunk_ids = {chunk.chunk_id for chunk in chunks}
        artifact = KnowledgeSearchArtifact(
            query=normalized_query,
            context_text=result.context_text,
            references=result.references,
            chunks=chunks,
            graph_paths=[
                SearchEvidencePath(
                    node_ids=[node.id for node in path.nodes],
                    edge_ids=[edge.id for edge in path.edges],
                    chunk_ids=path.chunk_ids,
                    context_text=format_graph_path(path),
                )
                for path in result.graph_paths
                if set(path.chunk_ids).issubset(returned_chunk_ids)
            ],
        )
        available_chunks = "\n".join(
            f"- {item.source} :: {item.chunk_id}" for item in artifact.chunks
        )
        content = (
            f"{artifact.context_text}\n\n"
            "Available chunks for neighborhood lookup:\n"
            f"{available_chunks or '- None'}"
        )
        return content, artifact

    @tool(response_format="content_and_artifact")
    def get_chunk_neighborhood(
        source: str,
        chunk_id: str,
        state: Annotated[AgentState, InjectedState],
    ) -> tuple[str, ChunkNeighborhoodArtifact | ToolErrorArtifact]:
        """Return the previous, current, and next chunks around a prior search hit."""
        key = _chunk_key(source, chunk_id)
        allowed_keys = {
            _chunk_key(item.source, item.chunk_id)
            for item in state["allowed_chunks"]
        }
        if key not in allowed_keys:
            message = (
                f"Unknown chunk_id {chunk_id!r} for source {source!r}. Use a "
                "source and chunk_id pair returned by search_knowledge_base in "
                "this turn."
            )
            return message, ToolErrorArtifact(
                tool_name="get_chunk_neighborhood",
                message=message,
            )
        if key in state["used_neighborhoods"]:
            message = (
                f"The neighborhood for chunk {chunk_id!r} in source {source!r} "
                "was already retrieved in this turn."
            )
            return message, ToolErrorArtifact(
                tool_name="get_chunk_neighborhood",
                message=message,
            )

        previous, current, next_chunk = runtime.chunk_store.load_neighborhood(
            source,
            chunk_id,
        )
        artifact = _neighborhood_artifact(
            source,
            chunk_id,
            previous,
            current,
            next_chunk,
        )
        return artifact.context_text, artifact

    return [search_knowledge_base, get_chunk_neighborhood]


def _chunk_reference(chunk: Chunk) -> AgentChunkReference:
    return AgentChunkReference(
        chunk_id=chunk.id,
        source=chunk.source,
        index=chunk.index,
        heading_path=chunk.heading_path,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
    )


def _neighborhood_chunk(chunk: Chunk) -> NeighborhoodChunk:
    return NeighborhoodChunk(
        **_chunk_reference(chunk).model_dump(),
        text=chunk.text,
    )


def _neighborhood_artifact(
    source: str,
    chunk_id: str,
    previous: Chunk | None,
    current: Chunk,
    next_chunk: Chunk | None,
) -> ChunkNeighborhoodArtifact:
    previous_result = _neighborhood_chunk(previous) if previous is not None else None
    current_result = _neighborhood_chunk(current)
    next_result = _neighborhood_chunk(next_chunk) if next_chunk is not None else None
    chunks = [
        chunk
        for chunk in (previous_result, current_result, next_result)
        if chunk is not None
    ]
    context_text = "\n\n".join(
        _neighborhood_text(label, chunk)
        for label, chunk in (
            ("Previous", previous_result),
            ("Current", current_result),
            ("Next", next_result),
        )
    )
    references = []
    seen_references = set()
    for chunk in chunks:
        reference = SourceReference(
            source=chunk.source,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
        key = (reference.source, reference.page_start, reference.page_end)
        if key not in seen_references:
            references.append(reference)
            seen_references.add(key)
    return ChunkNeighborhoodArtifact(
        source=source,
        chunk_id=chunk_id,
        context_text=context_text,
        references=references,
        previous=previous_result,
        current=current_result,
        next=next_result,
    )


def _neighborhood_text(
    label: str,
    chunk: NeighborhoodChunk | None,
) -> str:
    if chunk is None:
        return f"{label} chunk: [None]"
    heading = " > ".join(chunk.heading_path)
    metadata = f"{chunk.chunk_id} | {chunk.source}"
    if chunk.page_start is not None:
        metadata = f"{metadata} | pages {chunk.page_start}-{chunk.page_end}"
    if heading:
        metadata = f"{metadata} | {heading}"
    return f"{label} chunk [{metadata}]\n{chunk.text}"


def _chunk_key(source: str, chunk_id: str) -> str:
    return f"{source} :: {chunk_id}"
