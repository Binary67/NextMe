import logging

from langchain_core.messages import AIMessage, ToolMessage

from graphtool.agents.knowledge.tools import (
    ChunkNeighborhoodArtifact,
    DocumentSearchArtifact,
    KnowledgeSearchArtifact,
    ToolErrorArtifact,
)
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def trailing_tool_messages(messages) -> list[ToolMessage]:
    trailing = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        trailing.append(message)
    return list(reversed(trailing))


def tool_exchange_messages(messages, tool_messages: list[ToolMessage]):
    if not tool_messages:
        return []
    first_tool_index = len(messages) - len(tool_messages)
    preceding = messages[first_tool_index - 1] if first_tool_index > 0 else None
    if isinstance(preceding, AIMessage) and preceding.tool_calls:
        return [preceding, *tool_messages]
    return tool_messages


def tool_artifact(
    message: ToolMessage,
) -> (
    DocumentSearchArtifact
    | KnowledgeSearchArtifact
    | ChunkNeighborhoodArtifact
    | ToolErrorArtifact
    | None
):
    artifact = message.artifact
    if isinstance(
        artifact,
        (
            DocumentSearchArtifact,
            KnowledgeSearchArtifact,
            ChunkNeighborhoodArtifact,
            ToolErrorArtifact,
        ),
    ):
        return artifact
    if not isinstance(artifact, dict):
        return None
    artifact_type = artifact.get("type")
    if artifact_type == "document_search":
        return DocumentSearchArtifact.model_validate(artifact)
    if artifact_type == "search":
        return KnowledgeSearchArtifact.model_validate(artifact)
    if artifact_type == "chunk_neighborhood":
        return ChunkNeighborhoodArtifact.model_validate(artifact)
    if artifact_type == "error":
        return ToolErrorArtifact.model_validate(artifact)
    return None


def log_tool_selection(tool_call: dict) -> None:
    name = str(tool_call.get("name", "unknown"))
    arguments = tool_call.get("args", {})
    RUN_LOGGER.info("Research selected: %s", name)
    if not isinstance(arguments, dict):
        return
    if name == "search_knowledge_base":
        RUN_LOGGER.info("Search query: %s", arguments.get("query", ""))
        RUN_LOGGER.info("Search sources: %s", arguments.get("sources") or "all")
    elif name == "find_documents":
        RUN_LOGGER.info("Document query: %s", arguments.get("query", ""))
    elif name == "get_chunk_neighborhood":
        RUN_LOGGER.info(
            "Chunk neighborhood: %s :: %s",
            arguments.get("source", ""),
            arguments.get("chunk_id", ""),
        )
