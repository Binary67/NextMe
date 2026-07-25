import logging

from graphtool.agents.knowledge.state import AgentState
from graphtool.run_logging import LOGGER_NAME

MAX_RETRIEVALS_PER_SUBQUESTION = 3
MAX_CONSECUTIVE_EMPTY_RETRIEVALS = 2
RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def route_decomposition(state: AgentState) -> str:
    return (
        "finish_direct_response"
        if state.get("direct_response")
        else "research"
    )


def route_tool_result(state: AgentState) -> str:
    messages = state["messages"]
    if messages and getattr(messages[-1], "name", None) == "ask_user":
        return "record_user_response"
    return "record_tool_results"


def route_research(state: AgentState) -> str:
    if state.get("research_action") == "tools":
        return "tools"
    if state.get("research_action") == "respond":
        return "evaluate"
    if state.get("research_action") == "answer":
        return "complete_subquestion"
    raise RuntimeError("Research action is missing.")


def route_evaluation(state: AgentState) -> str:
    evaluation = state["evaluation"]
    if evaluation is None:
        raise RuntimeError("Evidence evaluation is missing.")
    if evaluation.verdict == "direct":
        return "finish_direct_response"
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


def route_completed_subquestion(state: AgentState) -> str:
    if state["subquestion_index"] + 1 < len(state["subquestions"]):
        return "advance_subquestion"
    return "answer"
