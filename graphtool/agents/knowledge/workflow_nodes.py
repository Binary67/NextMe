import logging

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from graphtool.agents.knowledge.state import (
    AgentResponse,
    AgentState,
    DocumentEvidenceRecord,
    EvidenceRecord,
    GraphPathEvidenceRecord,
    SubquestionOutcome,
)
from graphtool.agents.knowledge.tools import (
    ChunkNeighborhoodArtifact,
    DocumentSearchArtifact,
    KnowledgeSearchArtifact,
)
from graphtool.agents.knowledge.workflow_context import merge_references
from graphtool.agents.knowledge.workflow_evidence import (
    chunk_key,
    chunk_source_reference,
    merge_allowed_chunks,
    merge_document_evidence_record,
    merge_evidence_record,
    merge_graph_path_evidence_record,
    neighborhood_evidence_text,
)
from graphtool.agents.knowledge.workflow_tool_messages import (
    tool_artifact,
    tool_exchange_messages,
    trailing_tool_messages,
)
from graphtool.retrieval import SourceReference
from graphtool.run_logging import LOGGER_NAME
from graphtool.sequences import unique_ordered

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def record_tool_results(state: AgentState) -> dict:
    evidence = list(state["evidence"])
    document_evidence = list(state["document_evidence"])
    graph_path_evidence = list(state["graph_path_evidence"])
    references = list(state["references"])
    allowed_sources = list(state["allowed_sources"])
    allowed_chunks = list(state["allowed_chunks"])
    used_neighborhoods = list(state["used_neighborhoods"])
    search_count = state["search_count"]
    retrieval_count = state["retrieval_count"]
    retrieval_queries = list(state["retrieval_queries"])
    new_evidence_count = 0
    duplicate_evidence_count = 0
    tool_messages = trailing_tool_messages(state["messages"])

    for message in tool_messages:
        artifact = tool_artifact(message)
        retrieval_count += 1
        if isinstance(artifact, DocumentSearchArtifact):
            retrieval_queries.append(
                f"Document search: {artifact.query}"
            )
            for document in artifact.documents:
                references, reference_ids = merge_references(
                    references,
                    [SourceReference(source=document.source)],
                )
                document_evidence, is_new = merge_document_evidence_record(
                    document_evidence,
                    DocumentEvidenceRecord(
                        query=artifact.query,
                        source=document.source,
                        title=document.title,
                        headings=document.headings,
                        reference_ids=reference_ids,
                        subquestion_indexes=[state["subquestion_index"]],
                    ),
                    state["subquestion_index"],
                )
                if is_new:
                    new_evidence_count += 1
                else:
                    duplicate_evidence_count += 1
                if document.source not in allowed_sources:
                    allowed_sources.append(document.source)
        elif isinstance(artifact, KnowledgeSearchArtifact):
            retrieval_queries.append(artifact.query)
            allowed_chunks = merge_allowed_chunks(
                allowed_chunks,
                artifact.chunks,
            )
            search_count += 1
            reference_ids_by_chunk: dict[str, list[str]] = {}
            for chunk in artifact.chunks:
                references, reference_ids = merge_references(
                    references,
                    [chunk_source_reference(chunk)],
                )
                evidence, is_new = merge_evidence_record(
                    evidence,
                    EvidenceRecord(
                        query=artifact.query,
                        source=chunk.source,
                        chunk_id=chunk.chunk_id,
                        context_text=chunk.context_text,
                        reference_ids=reference_ids,
                        subquestion_indexes=[state["subquestion_index"]],
                    ),
                    state["subquestion_index"],
                )
                if is_new:
                    new_evidence_count += 1
                else:
                    duplicate_evidence_count += 1
                reference_ids_by_chunk[chunk.chunk_id] = reference_ids
            for path in artifact.graph_paths:
                path_reference_ids = unique_ordered(
                    [
                        reference_id
                        for chunk_id in path.chunk_ids
                        for reference_id in reference_ids_by_chunk[chunk_id]
                    ]
                )
                graph_path_evidence, is_new = (
                    merge_graph_path_evidence_record(
                        graph_path_evidence,
                        GraphPathEvidenceRecord(
                            query=artifact.query,
                            node_ids=path.node_ids,
                            edge_ids=path.edge_ids,
                            chunk_ids=path.chunk_ids,
                            context_text=path.context_text,
                            reference_ids=path_reference_ids,
                            subquestion_indexes=[state["subquestion_index"]],
                        ),
                        state["subquestion_index"],
                    )
                )
                if is_new:
                    new_evidence_count += 1
                else:
                    duplicate_evidence_count += 1
        elif isinstance(artifact, ChunkNeighborhoodArtifact):
            query = (
                "Chunk neighborhood: "
                f"{artifact.source} :: {artifact.chunk_id}"
            )
            retrieval_queries.append(query)
            neighborhood_chunks = [
                chunk
                for chunk in (
                    artifact.previous,
                    artifact.current,
                    artifact.next,
                )
                if chunk is not None
            ]
            for chunk in neighborhood_chunks:
                references, reference_ids = merge_references(
                    references,
                    [chunk_source_reference(chunk)],
                )
                evidence, is_new = merge_evidence_record(
                    evidence,
                    EvidenceRecord(
                        query=query,
                        source=chunk.source,
                        chunk_id=chunk.chunk_id,
                        context_text=neighborhood_evidence_text(chunk),
                        reference_ids=reference_ids,
                        subquestion_indexes=[state["subquestion_index"]],
                    ),
                    state["subquestion_index"],
                )
                if is_new:
                    new_evidence_count += 1
                else:
                    duplicate_evidence_count += 1
            key = chunk_key(artifact.source, artifact.chunk_id)
            if key not in used_neighborhoods:
                used_neighborhoods.append(key)
        else:
            retrieval_queries.append(
                f"Tool error: {message.name or 'unknown'}"
            )
            if message.name == "search_knowledge_base":
                search_count += 1

    exchange_messages = tool_exchange_messages(
        state["messages"],
        tool_messages,
    )
    RUN_LOGGER.info(
        "Retrieval progress: returned=%d, new=%d, duplicates=%d",
        new_evidence_count + duplicate_evidence_count,
        new_evidence_count,
        duplicate_evidence_count,
    )
    consecutive_empty_retrievals = (
        state["consecutive_empty_retrievals"] + 1
        if new_evidence_count == 0
        else 0
    )
    return {
        "messages": [
            RemoveMessage(id=message.id)
            for message in exchange_messages
            if message.id is not None
        ],
        "evidence": evidence,
        "document_evidence": document_evidence,
        "graph_path_evidence": graph_path_evidence,
        "references": references,
        "allowed_sources": allowed_sources,
        "allowed_chunks": allowed_chunks,
        "used_neighborhoods": used_neighborhoods,
        "search_count": search_count,
        "retrieval_count": retrieval_count,
        "retrieval_queries": retrieval_queries,
        "new_evidence_count": new_evidence_count,
        "duplicate_evidence_count": duplicate_evidence_count,
        "consecutive_empty_retrievals": consecutive_empty_retrievals,
        "research_action": None,
    }


def record_user_response(state: AgentState) -> dict:
    tool_messages = trailing_tool_messages(state["messages"])
    if len(tool_messages) != 1 or tool_messages[0].name != "ask_user":
        raise RuntimeError("Expected one ask_user tool response.")
    tool_message = tool_messages[0]
    exchange_messages = tool_exchange_messages(
        state["messages"],
        tool_messages,
    )
    if not exchange_messages or not isinstance(exchange_messages[0], AIMessage):
        raise RuntimeError("ask_user response is missing its tool call.")
    tool_call = next(
        (
            call
            for call in exchange_messages[0].tool_calls
            if call.get("id") == tool_message.tool_call_id
        ),
        None,
    )
    if tool_call is None:
        raise RuntimeError("ask_user response does not match its tool call.")
    arguments = tool_call.get("args", {})
    question = arguments.get("question") if isinstance(arguments, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise RuntimeError("ask_user tool call is missing its question.")
    if not isinstance(tool_message.content, str) or not tool_message.content.strip():
        raise RuntimeError("ask_user tool response is empty.")
    normalized_question = question.strip()
    normalized_answer = tool_message.content.strip()
    RUN_LOGGER.info("User answered clarification question")
    return {
        "messages": [
            *[
                RemoveMessage(id=message.id)
                for message in exchange_messages
                if message.id is not None
            ],
            AIMessage(content=normalized_question),
            HumanMessage(content=normalized_answer),
        ],
        "research_action": None,
        "direct_response": None,
        "evaluation": None,
    }


def complete_subquestion(state: AgentState) -> dict:
    evaluation = state["evaluation"]
    if evaluation is None or evaluation.verdict == "direct":
        raise RuntimeError("Subquestion evaluation is not complete.")
    outcome = SubquestionOutcome(
        question=state["subquestions"][state["subquestion_index"]],
        verdict=evaluation.verdict,
        missing_information=evaluation.missing_information,
    )
    return {"subquestion_outcomes": [*state["subquestion_outcomes"], outcome]}


def advance_subquestion(state: AgentState) -> dict:
    return {
        "subquestion_index": state["subquestion_index"] + 1,
        "retrieval_count": 0,
        "retrieval_queries": [],
        "new_evidence_count": 0,
        "duplicate_evidence_count": 0,
        "consecutive_empty_retrievals": 0,
        "allowed_sources": [],
        "allowed_chunks": [],
        "used_neighborhoods": [],
        "research_action": None,
        "direct_response": None,
        "evaluation": None,
    }


def finish_direct_response(state: AgentState) -> dict:
    response = AgentResponse(
        answer=state["direct_response"] or "",
        status="complete",
        references=[],
        search_count=state["search_count"],
    )
    return {
        "messages": [AIMessage(content=response.answer)],
        "response": response,
    }


def cleanup(state: AgentState) -> dict:
    return {
        "question": "",
        "knowledge_scope": None,
        "subquestions": [],
        "subquestion_index": 0,
        "subquestion_outcomes": [],
        "evidence": [],
        "document_evidence": [],
        "graph_path_evidence": [],
        "references": [],
        "search_count": 0,
        "retrieval_count": 0,
        "retrieval_queries": [],
        "new_evidence_count": 0,
        "duplicate_evidence_count": 0,
        "consecutive_empty_retrievals": 0,
        "allowed_sources": [],
        "allowed_chunks": [],
        "used_neighborhoods": [],
        "research_action": None,
        "direct_response": None,
        "evaluation": None,
    }
