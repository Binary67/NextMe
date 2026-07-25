import logging
from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from openai import APITimeoutError
from pydantic import BaseModel

from graphtool.agents.knowledge.prompts import (
    ANSWER_SYSTEM_PROMPT,
    DECOMPOSITION_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    NO_EVIDENCE_ANSWER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)
from graphtool.agents.knowledge.state import (
    AgentResponse,
    AgentChunkReference,
    AgentState,
    ConversationSummary,
    DocumentEvidenceRecord,
    EvidenceRecord,
    ExpandRecommendation,
    FinalAnswerDraft,
    GraphPathEvidenceRecord,
    QueryDecomposition,
    SearchRecommendation,
    SubquestionOutcome,
    SufficiencyDecision,
)
from graphtool.agents.knowledge.tools import (
    ChunkNeighborhoodArtifact,
    DocumentSearchArtifact,
    KnowledgeSearchArtifact,
    ToolErrorArtifact,
    create_knowledge_tools,
)
from graphtool.agents.knowledge.workflow_context import (
    answer_text,
    conversation_token_count,
    decomposition_text,
    evaluation_text,
    merge_references,
    research_context,
    summary_text,
    unique_ordered,
)
from graphtool.run_logging import LOGGER_NAME
from graphtool.retrieval import SourceReference
from graphtool.runtime import GraphToolRuntime

MAX_RETRIEVALS_PER_SUBQUESTION = 3
MAX_CONSECUTIVE_EMPTY_RETRIEVALS = 2
RUN_LOGGER = logging.getLogger(LOGGER_NAME)
RESEARCH_TOOL_CORRECTION = (
    "Your previous response did not call a retrieval tool. The available "
    "evidence is insufficient, so call exactly one retrieval tool now. Do not "
    "answer with prose."
)
SINGLE_RESEARCH_TOOL_CORRECTION = (
    "Call exactly one retrieval tool. Do not call multiple tools or answer "
    "with prose."
)
NO_EVIDENCE_DISCLOSURE = (
    "I couldn't find supporting information in the knowledge base. The following "
    "is a best-effort answer based on general knowledge and is not verified "
    "against the knowledge base."
)


def build_workflow_graph(
    answer_model: BaseChatModel,
    orchestration_model: BaseChatModel,
    runtime: GraphToolRuntime,
    checkpointer: InMemorySaver,
    *,
    compaction_trigger_tokens: int,
    retained_recent_tokens: int,
):
    tools = create_knowledge_tools(runtime)
    summary_model = orchestration_model.with_structured_output(
        ConversationSummary
    )
    decomposition_model = orchestration_model.with_structured_output(
        QueryDecomposition
    )
    research_model = orchestration_model.bind_tools(
        tools,
        parallel_tool_calls=False,
    )
    evaluator_model = answer_model.with_structured_output(SufficiencyDecision)
    answer_draft_model = answer_model.with_structured_output(FinalAnswerDraft)

    def compact(state: AgentState) -> dict:
        summary = state.get("conversation_summary", "")
        messages = state["messages"]
        if (
            conversation_token_count(summary, messages)
            < compaction_trigger_tokens
        ):
            return {}

        retained_messages = trim_messages(
            messages,
            max_tokens=retained_recent_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            allow_partial=False,
            start_on="human",
        )
        if not retained_messages:
            retained_messages = [messages[-1]]
        retained_ids = {message.id for message in retained_messages}
        messages_to_summarize = [
            message for message in messages if message.id not in retained_ids
        ]
        if not messages_to_summarize:
            return {}

        summary_messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(
                content=summary_text(summary, messages_to_summarize)
            ),
        ]
        summary_result, duration = _invoke_model(
            summary_model,
            summary_messages,
            stage="conversation summary",
        )
        updated_summary = _validated_output(
            ConversationSummary,
            summary_result,
        )
        RUN_LOGGER.info(
            "Conversation summary completed in %.2fs",
            duration,
        )
        return {
            "conversation_summary": updated_summary.summary.strip(),
            "messages": [
                RemoveMessage(id=message.id)
                for message in messages_to_summarize
                if message.id is not None
            ],
        }

    def decompose(state: AgentState) -> dict:
        knowledge_scopes = getattr(runtime, "knowledge_scopes", {})
        result, duration = _invoke_model(
            decomposition_model,
            [
                SystemMessage(content=DECOMPOSITION_SYSTEM_PROMPT),
                HumanMessage(
                    content=decomposition_text(state, knowledge_scopes)
                ),
            ],
            stage="question decomposition",
        )
        decomposition = _validated_output(QueryDecomposition, result)
        selected_scope = (
            decomposition.knowledge_scope.casefold()
            if decomposition.knowledge_scope is not None
            else None
        )
        unmatched_scope = decomposition.unmatched_scope
        if selected_scope is not None and selected_scope not in knowledge_scopes:
            unmatched_scope = decomposition.knowledge_scope or selected_scope
            selected_scope = None
        RUN_LOGGER.info(
            "Question decomposition completed in %.2fs: subquestions=%d",
            duration,
            len(decomposition.subquestions),
        )
        for index, subquestion in enumerate(decomposition.subquestions, start=1):
            RUN_LOGGER.info(
                "Decomposed subquestion %d: %s",
                index,
                subquestion,
            )
        if unmatched_scope:
            available = ", ".join(knowledge_scopes)
            direct_response = (
                "I couldn't match that folder to the knowledge-folder catalog. "
                f"Available folders are: {available}. Which folder should I search?"
                if available
                else (
                    "I couldn't match that folder because no knowledge folders "
                    "are configured."
                )
            )
            RUN_LOGGER.info(
                "Knowledge scope could not be matched: %s",
                unmatched_scope,
            )
            return {
                "subquestions": decomposition.subquestions,
                "subquestion_index": 0,
                "subquestion_outcomes": [],
                "knowledge_scope": None,
                "direct_response": direct_response,
            }
        RUN_LOGGER.info(
            "Knowledge scope selected: %s",
            selected_scope or "all",
        )
        return {
            "subquestions": decomposition.subquestions,
            "subquestion_index": 0,
            "subquestion_outcomes": [],
            "knowledge_scope": selected_scope,
            "direct_response": None,
        }

    def research(state: AgentState) -> dict:
        follow_up = (
            state.get("evaluation") is not None
            and state["evaluation"].verdict == "insufficient"
        )
        round_number = state["retrieval_count"] + 1
        research_messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=research_context(state)),
            *state["messages"],
        ]
        try:
            response, duration = _invoke_model(
                research_model,
                research_messages,
                stage=f"research round {round_number}",
            )
            research_duration = duration
            if not isinstance(response, AIMessage):
                raise TypeError(
                    "Tool-bound research model must return an AIMessage."
                )
            correction = _research_response_correction(
                state,
                response,
                follow_up=follow_up,
            )
            if correction is not None:
                RUN_LOGGER.info(
                    "Research round %d selected an invalid action; retrying "
                    "with a correction",
                    round_number,
                )
                response, correction_duration = _invoke_model(
                    research_model,
                    [
                        *research_messages,
                        response,
                        HumanMessage(content=correction),
                    ],
                    stage=f"research round {round_number} corrective retry",
                )
                research_duration += correction_duration
                if not isinstance(response, AIMessage):
                    raise TypeError(
                        "Tool-bound research model must return an AIMessage."
                    )
                correction = _research_response_correction(
                    state,
                    response,
                    follow_up=follow_up,
                )
        except APITimeoutError:
            if follow_up and state["retrieval_count"] > 0:
                RUN_LOGGER.warning(
                    "Follow-up research timed out; answering with the evidence "
                    "already retrieved"
                )
                return {"research_action": "answer", "direct_response": None}
            raise
        RUN_LOGGER.info(
            "Research round %d completed in %.2fs",
            round_number,
            research_duration,
        )
        if response.tool_calls and correction is None:
            _log_tool_selection(response.tool_calls[0])
            return {
                "messages": [response],
                "research_action": "tools",
                "direct_response": None,
                "evaluation": None,
            }
        if follow_up:
            if state["retrieval_count"] == 0:
                raise RuntimeError(
                    "Research model did not select a retrieval tool after correction."
                )
            RUN_LOGGER.warning(
                "Follow-up research did not select a valid tool after correction; "
                "answering with the evidence already retrieved"
            )
            return {"research_action": "answer", "direct_response": None}
        if correction is not None:
            raise RuntimeError(
                "Research model did not select a valid retrieval tool after "
                "correction."
            )
        return {
            "research_action": "respond",
            "direct_response": _message_text(response).strip(),
        }

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
        tool_messages = _trailing_tool_messages(state["messages"])

        for message in tool_messages:
            artifact = _tool_artifact(message)
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
                    document_evidence, is_new = _merge_document_evidence_record(
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
                allowed_chunks = _merge_allowed_chunks(
                    allowed_chunks,
                    artifact.chunks,
                )
                search_count += 1
                reference_ids_by_chunk: dict[str, list[str]] = {}
                for chunk in artifact.chunks:
                    references, reference_ids = merge_references(
                        references,
                        [_chunk_source_reference(chunk)],
                    )
                    evidence, is_new = _merge_evidence_record(
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
                        _merge_graph_path_evidence_record(
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
                        [_chunk_source_reference(chunk)],
                    )
                    evidence, is_new = _merge_evidence_record(
                        evidence,
                        EvidenceRecord(
                            query=query,
                            source=chunk.source,
                            chunk_id=chunk.chunk_id,
                            context_text=_neighborhood_evidence_text(chunk),
                            reference_ids=reference_ids,
                            subquestion_indexes=[state["subquestion_index"]],
                        ),
                        state["subquestion_index"],
                    )
                    if is_new:
                        new_evidence_count += 1
                    else:
                        duplicate_evidence_count += 1
                key = _chunk_key(artifact.source, artifact.chunk_id)
                if key not in used_neighborhoods:
                    used_neighborhoods.append(key)
            else:
                retrieval_queries.append(
                    f"Tool error: {message.name or 'unknown'}"
                )
                if message.name == "search_knowledge_base":
                    search_count += 1

        exchange_messages = _tool_exchange_messages(
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

    def evaluate(state: AgentState) -> dict:
        round_number = state["retrieval_count"]
        evaluation_messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=evaluation_text(state)),
        ]
        evaluation_result, duration = _invoke_model(
            evaluator_model,
            evaluation_messages,
            stage=f"evidence evaluation round {round_number}",
        )
        decision = _validated_output(
            SufficiencyDecision,
            evaluation_result,
        )
        if decision.verdict == "sufficient" and not _has_current_evidence(state):
            decision = SufficiencyDecision(
                verdict="insufficient",
                missing_information=[
                    "No knowledge-base evidence has been retrieved."
                ],
                recommendation=SearchRecommendation(
                    reason="The question still needs knowledge-base evidence.",
                    search_focus=state["subquestions"][
                        state["subquestion_index"]
                    ],
                ),
            )
        if decision.verdict == "conversation" and (
            state["evidence"] or not state.get("direct_response")
        ):
            decision = SufficiencyDecision(
                verdict="insufficient",
                missing_information=[
                    "The request requires a knowledge-base-grounded answer."
                ],
                recommendation=SearchRecommendation(
                    reason="The request is substantive and requires evidence.",
                    search_focus=state["subquestions"][
                        state["subquestion_index"]
                    ],
                ),
            )
        decision = _fallback_from_unavailable_expansion(state, decision)
        RUN_LOGGER.info(
            "Evidence evaluation round %d completed in %.2fs: %s",
            round_number,
            duration,
            decision.verdict,
        )
        for missing_item in decision.missing_information:
            RUN_LOGGER.info(
                "Missing information: %s",
                missing_item,
            )
        return {"evaluation": decision}

    def complete_subquestion(state: AgentState) -> dict:
        evaluation = state["evaluation"]
        if evaluation is None or evaluation.verdict == "conversation":
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

    def answer(state: AgentState) -> dict:
        answer_started_at = perf_counter()
        partial = any(
            outcome.verdict == "insufficient"
            for outcome in state["subquestion_outcomes"]
        )
        no_evidence = partial and not state["references"]
        system_prompt = (
            NO_EVIDENCE_ANSWER_SYSTEM_PROMPT
            if no_evidence
            else ANSWER_SYSTEM_PROMPT
        )
        prompt_text = answer_text(state, partial=partial)
        references_by_id = {
            item.id: item.reference for item in state["references"]
        }
        answer_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_text),
        ]
        answer_result, _ = _invoke_model(
            answer_draft_model,
            answer_messages,
            stage="answer generation",
        )
        draft = _validated_output(
            FinalAnswerDraft,
            answer_result,
        )
        cited_ids = unique_ordered(draft.cited_reference_ids)
        unknown_ids = [
            reference_id
            for reference_id in cited_ids
            if reference_id not in references_by_id
        ]
        if unknown_ids:
            valid_ids = ", ".join(references_by_id) or "[None]"
            invalid_ids = ", ".join(unknown_ids)
            retry_messages = [
                SystemMessage(
                    content=(
                        f"{system_prompt}\n\n"
                        "Your previous draft cited unknown reference IDs: "
                        f"{invalid_ids}. Regenerate the answer using only "
                        f"these available reference IDs: {valid_ids}. Remove "
                        "or qualify any claim that the available evidence "
                        "does not support."
                    )
                ),
                HumanMessage(content=prompt_text),
            ]
            retry_result, _ = _invoke_model(
                answer_draft_model,
                retry_messages,
                stage="answer citation retry",
            )
            draft = _validated_output(
                FinalAnswerDraft,
                retry_result,
            )
            cited_ids = unique_ordered(draft.cited_reference_ids)
            unknown_ids = [
                reference_id
                for reference_id in cited_ids
                if reference_id not in references_by_id
            ]
            if unknown_ids:
                joined = ", ".join(unknown_ids)
                raise RuntimeError(
                    "Knowledge agent answer cited unknown references after retry: "
                    f"{joined}."
                )
        cited_references = [
            references_by_id[reference_id] for reference_id in cited_ids
        ]
        if state["references"] and not cited_references:
            raise RuntimeError(
                "Knowledge agent answer did not cite retrieved evidence."
            )
        response_text = draft.answer.strip()
        if no_evidence:
            response_text = f"{NO_EVIDENCE_DISCLOSURE}\n\n{response_text}"
        response = AgentResponse(
            answer=response_text,
            status="partial" if partial else "complete",
            references=cited_references,
            search_count=state["search_count"],
        )
        RUN_LOGGER.info(
            "Answer completed in %.2fs: status=%s, references=%d",
            perf_counter() - answer_started_at,
            response.status,
            len(response.references),
        )
        return {
            "messages": [AIMessage(content=response.answer)],
            "response": response,
        }

    def finish_conversation(state: AgentState) -> dict:
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

    builder = StateGraph(AgentState)
    builder.add_node("compact", compact)
    builder.add_node("decompose", decompose)
    builder.add_node("research", research)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("record_tool_results", record_tool_results)
    builder.add_node("evaluate", evaluate)
    builder.add_node("complete_subquestion", complete_subquestion)
    builder.add_node("advance_subquestion", advance_subquestion)
    builder.add_node("answer", answer)
    builder.add_node("finish_conversation", finish_conversation)
    builder.add_node("cleanup", cleanup)
    builder.add_edge(START, "compact")
    builder.add_edge("compact", "decompose")
    builder.add_conditional_edges(
        "decompose",
        _route_decomposition,
        {
            "research": "research",
            "finish_conversation": "finish_conversation",
        },
    )
    builder.add_conditional_edges(
        "research",
        _route_research,
        {
            "tools": "tools",
            "evaluate": "evaluate",
            "complete_subquestion": "complete_subquestion",
        },
    )
    builder.add_edge("tools", "record_tool_results")
    builder.add_edge("record_tool_results", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        _route_evaluation,
        {
            "finish_conversation": "finish_conversation",
            "research": "research",
            "complete_subquestion": "complete_subquestion",
        },
    )
    builder.add_conditional_edges(
        "complete_subquestion",
        _route_completed_subquestion,
        {"advance_subquestion": "advance_subquestion", "answer": "answer"},
    )
    builder.add_edge("advance_subquestion", "research")
    builder.add_edge("answer", "cleanup")
    builder.add_edge("finish_conversation", "cleanup")
    builder.add_edge("cleanup", END)
    return builder.compile(checkpointer=checkpointer)


def _route_research(state: AgentState) -> str:
    if state.get("research_action") == "tools":
        return "tools"
    if state.get("research_action") == "respond":
        return "evaluate"
    if state.get("research_action") == "answer":
        return "complete_subquestion"
    raise RuntimeError("Research action is missing.")


def _route_decomposition(state: AgentState) -> str:
    return (
        "finish_conversation"
        if state.get("direct_response")
        else "research"
    )


def _route_evaluation(state: AgentState) -> str:
    evaluation = state["evaluation"]
    if evaluation is None:
        raise RuntimeError("Evidence evaluation is missing.")
    if evaluation.verdict == "conversation":
        return "finish_conversation"
    if evaluation.verdict == "sufficient":
        return "complete_subquestion"
    if (
        state["consecutive_empty_retrievals"]
        >= MAX_CONSECUTIVE_EMPTY_RETRIEVALS
    ):
        RUN_LOGGER.info(
            "Early stopping: %d consecutive retrievals returned no new evidence",
            state["consecutive_empty_retrievals"],
        )
        return "complete_subquestion"
    if state["retrieval_count"] >= MAX_RETRIEVALS_PER_SUBQUESTION:
        RUN_LOGGER.info(
            "Retrieval limit reached after %d retrievals",
            state["retrieval_count"],
        )
        return "complete_subquestion"
    return "research"


def _route_completed_subquestion(state: AgentState) -> str:
    if state["subquestion_index"] + 1 < len(state["subquestions"]):
        return "advance_subquestion"
    return "answer"


def _research_response_correction(
    state: AgentState,
    response: AIMessage,
    *,
    follow_up: bool,
) -> str | None:
    if not response.tool_calls:
        return RESEARCH_TOOL_CORRECTION if follow_up else None
    if len(response.tool_calls) != 1:
        return SINGLE_RESEARCH_TOOL_CORRECTION

    tool_call = response.tool_calls[0]
    name = str(tool_call.get("name", ""))
    arguments = tool_call.get("args", {})
    if not isinstance(arguments, dict):
        return SINGLE_RESEARCH_TOOL_CORRECTION

    evaluation = state.get("evaluation")
    recommendation = (
        evaluation.recommendation if evaluation is not None else None
    )
    if isinstance(recommendation, ExpandRecommendation):
        if (
            name == "get_chunk_neighborhood"
            and arguments.get("source") == recommendation.source
            and arguments.get("chunk_id") == recommendation.chunk_id
        ):
            return None
        return (
            "Call get_chunk_neighborhood exactly once with source "
            f"{recommendation.source!r} and chunk_id "
            f"{recommendation.chunk_id!r}. Do not call another tool."
        )

    if name == "get_chunk_neighborhood":
        return (
            "The recommended action is search. Call exactly one of "
            "find_documents or search_knowledge_base, not "
            "get_chunk_neighborhood."
        )
    if name not in {"find_documents", "search_knowledge_base"}:
        return SINGLE_RESEARCH_TOOL_CORRECTION
    if _is_duplicate_retrieval(state, name, arguments):
        return (
            "That retrieval was already attempted. Call exactly one search "
            "tool with a different focused query."
        )
    return None


def _is_duplicate_retrieval(
    state: AgentState,
    tool_name: str,
    arguments: dict,
) -> bool:
    query = str(arguments.get("query", "")).strip()
    if tool_name == "find_documents":
        query = f"Document search: {query}"
    normalized_queries = {
        item.casefold() for item in state["retrieval_queries"]
    }
    return query.casefold() in normalized_queries


def _fallback_from_unavailable_expansion(
    state: AgentState,
    decision: SufficiencyDecision,
) -> SufficiencyDecision:
    recommendation = decision.recommendation
    if not isinstance(recommendation, ExpandRecommendation):
        return decision
    if _expansion_is_available(state, recommendation):
        return decision

    RUN_LOGGER.info(
        "Recommended expansion is unavailable; falling back to search: %s :: %s",
        recommendation.source,
        recommendation.chunk_id,
    )
    return decision.model_copy(
        update={
            "recommendation": SearchRecommendation(
                reason=(
                    "The recommended chunk is unavailable or was already "
                    "expanded, so a different passage is needed."
                ),
                search_focus="; ".join(decision.missing_information),
            )
        }
    )


def _expansion_is_available(
    state: AgentState,
    recommendation: ExpandRecommendation,
) -> bool:
    key = _chunk_key(recommendation.source, recommendation.chunk_id)
    return (
        key not in state["used_neighborhoods"]
        and any(
            _chunk_key(chunk.source, chunk.chunk_id) == key
            for chunk in state["allowed_chunks"]
        )
    )


def _log_tool_selection(tool_call: dict) -> None:
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


def _invoke_model(model, messages: list, *, stage: str):
    RUN_LOGGER.info(
        "Starting %s: prompt approximately %d tokens",
        stage,
        count_tokens_approximately(messages),
    )
    started_at = perf_counter()
    try:
        return model.invoke(messages), perf_counter() - started_at
    except Exception as exc:
        duration = perf_counter() - started_at
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            RUN_LOGGER.error(
                "%s failed after %.2fs: %s",
                stage.capitalize(),
                duration,
                type(exc).__name__,
            )
        else:
            RUN_LOGGER.error(
                "%s failed after %.2fs: %s (status=%s)",
                stage.capitalize(),
                duration,
                type(exc).__name__,
                status_code,
            )
        raise


def _validated_output(model_type: type[BaseModel], value):
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


def _trailing_tool_messages(messages) -> list[ToolMessage]:
    trailing = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        trailing.append(message)
    return list(reversed(trailing))


def _tool_exchange_messages(messages, tool_messages: list[ToolMessage]):
    if not tool_messages:
        return []
    first_tool_index = len(messages) - len(tool_messages)
    preceding = messages[first_tool_index - 1] if first_tool_index > 0 else None
    if isinstance(preceding, AIMessage) and preceding.tool_calls:
        return [preceding, *tool_messages]
    return tool_messages


def _tool_artifact(
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


def _merge_allowed_chunks(
    existing: list[AgentChunkReference],
    incoming: list[AgentChunkReference],
) -> list[AgentChunkReference]:
    merged = list(existing)
    keys = {_chunk_key(item.source, item.chunk_id) for item in existing}
    for item in incoming:
        key = _chunk_key(item.source, item.chunk_id)
        if key not in keys:
            merged.append(item)
            keys.add(key)
    return merged


def _merge_evidence_record(
    existing: list[EvidenceRecord],
    incoming: EvidenceRecord,
    subquestion_index: int,
) -> tuple[list[EvidenceRecord], bool]:
    merged = list(existing)
    key = _chunk_key(incoming.source, incoming.chunk_id)
    for index, record in enumerate(merged):
        if _chunk_key(record.source, record.chunk_id) != key:
            continue
        if subquestion_index not in record.subquestion_indexes:
            merged[index] = record.model_copy(
                update={
                    "subquestion_indexes": [
                        *record.subquestion_indexes,
                        subquestion_index,
                    ]
                }
            )
        return merged, False
    merged.append(incoming)
    return merged, True


def _merge_document_evidence_record(
    existing: list[DocumentEvidenceRecord],
    incoming: DocumentEvidenceRecord,
    subquestion_index: int,
) -> tuple[list[DocumentEvidenceRecord], bool]:
    merged = list(existing)
    for index, record in enumerate(merged):
        if (
            record.source != incoming.source
            or record.query.casefold() != incoming.query.casefold()
        ):
            continue
        if subquestion_index not in record.subquestion_indexes:
            merged[index] = record.model_copy(
                update={
                    "subquestion_indexes": [
                        *record.subquestion_indexes,
                        subquestion_index,
                    ]
                }
            )
        return merged, False
    merged.append(incoming)
    return merged, True


def _merge_graph_path_evidence_record(
    existing: list[GraphPathEvidenceRecord],
    incoming: GraphPathEvidenceRecord,
    subquestion_index: int,
) -> tuple[list[GraphPathEvidenceRecord], bool]:
    merged = list(existing)
    key = (tuple(incoming.node_ids), tuple(incoming.edge_ids))
    for index, record in enumerate(merged):
        if (tuple(record.node_ids), tuple(record.edge_ids)) != key:
            continue
        if subquestion_index not in record.subquestion_indexes:
            merged[index] = record.model_copy(
                update={
                    "subquestion_indexes": [
                        *record.subquestion_indexes,
                        subquestion_index,
                    ]
                }
            )
        return merged, False
    merged.append(incoming)
    return merged, True


def _chunk_source_reference(chunk: AgentChunkReference) -> SourceReference:
    return SourceReference(
        source=chunk.source,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
    )


def _neighborhood_evidence_text(chunk) -> str:
    heading = " > ".join(chunk.heading_path)
    metadata = f"{chunk.chunk_id} | {chunk.source}"
    if chunk.page_start is not None:
        metadata = f"{metadata} | pages {chunk.page_start}-{chunk.page_end}"
    if heading:
        metadata = f"{metadata} | {heading}"
    return f"[{metadata}]\n{chunk.text}"


def _has_current_evidence(state: AgentState) -> bool:
    subquestion_index = state["subquestion_index"]
    return any(
        subquestion_index in record.subquestion_indexes
        for record in state["evidence"]
    ) or any(
        subquestion_index in record.subquestion_indexes
        for record in state["document_evidence"]
    )


def _chunk_key(source: str, chunk_id: str) -> str:
    return f"{source} :: {chunk_id}"
