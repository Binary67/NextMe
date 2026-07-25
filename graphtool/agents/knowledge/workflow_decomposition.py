import logging

from langchain_core.messages import HumanMessage, SystemMessage

from graphtool.agents.knowledge.prompts import DECOMPOSITION_SYSTEM_PROMPT
from graphtool.agents.knowledge.state import AgentState, QueryDecomposition
from graphtool.agents.knowledge.workflow_context import decomposition_text
from graphtool.agents.knowledge.workflow_model_io import (
    invoke_model,
    validated_output,
)
from graphtool.run_logging import LOGGER_NAME
from graphtool.runtime import GraphToolRuntime

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def make_decompose_node(decomposition_model, runtime: GraphToolRuntime):
    def decompose(state: AgentState) -> dict:
        knowledge_scopes = getattr(runtime, "knowledge_scopes", {})
        result, duration = invoke_model(
            decomposition_model,
            [
                SystemMessage(content=DECOMPOSITION_SYSTEM_PROMPT),
                HumanMessage(
                    content=decomposition_text(state, knowledge_scopes)
                ),
            ],
            stage="question decomposition",
        )
        decomposition = validated_output(QueryDecomposition, result)
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

    return decompose
