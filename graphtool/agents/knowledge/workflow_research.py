import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APITimeoutError

from graphtool.agents.knowledge.prompts import RESEARCH_SYSTEM_PROMPT
from graphtool.agents.knowledge.state import (
    AgentState,
    AskUserRecommendation,
    ExpandRecommendation,
    SearchRecommendation,
    SufficiencyDecision,
)
from graphtool.agents.knowledge.workflow_context import research_context
from graphtool.agents.knowledge.workflow_evidence import chunk_key
from graphtool.agents.knowledge.workflow_model_io import (
    invoke_model,
    message_text,
)
from graphtool.agents.knowledge.workflow_tool_messages import log_tool_selection
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)
RESEARCH_TOOL_CORRECTION = (
    "Your previous response did not call a tool. The available evidence is "
    "insufficient, so call exactly one recommended tool now. Do not answer "
    "with prose."
)
SINGLE_RESEARCH_TOOL_CORRECTION = (
    "Call exactly one tool. Do not call multiple tools or answer with prose."
)


def make_research_node(research_model):
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
            response, duration = invoke_model(
                research_model,
                research_messages,
                stage=f"research round {round_number}",
            )
            research_duration = duration
            if not isinstance(response, AIMessage):
                raise TypeError(
                    "Tool-bound research model must return an AIMessage."
                )
            correction = research_response_correction(
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
                response, correction_duration = invoke_model(
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
                correction = research_response_correction(
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
            log_tool_selection(response.tool_calls[0])
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
            "direct_response": message_text(response).strip(),
        }

    return research


def research_response_correction(
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
    if isinstance(recommendation, AskUserRecommendation):
        if (
            name == "ask_user"
            and arguments.get("question") == recommendation.question
        ):
            return None
        return (
            "Call ask_user exactly once with question "
            f"{recommendation.question!r}. Do not call another tool."
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

    if name == "ask_user":
        if isinstance(recommendation, SearchRecommendation):
            return (
                "The recommended action is search. Call exactly one of "
                "find_documents or search_knowledge_base, not ask_user."
            )
        question = arguments.get("question")
        if isinstance(question, str) and question.strip():
            return None
        return "Call ask_user with one non-empty focused question."
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


def fallback_from_unavailable_expansion(
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


def _expansion_is_available(
    state: AgentState,
    recommendation: ExpandRecommendation,
) -> bool:
    key = chunk_key(recommendation.source, recommendation.chunk_id)
    return (
        key not in state["used_neighborhoods"]
        and any(
            chunk_key(chunk.source, chunk.chunk_id) == key
            for chunk in state["allowed_chunks"]
        )
    )
