import logging

from langchain_core.messages import HumanMessage, SystemMessage

from graphtool.agents.knowledge.prompts import EVALUATOR_SYSTEM_PROMPT
from graphtool.agents.knowledge.state import (
    AgentState,
    SearchRecommendation,
    SufficiencyDecision,
)
from graphtool.agents.knowledge.workflow_context import evaluation_text
from graphtool.agents.knowledge.workflow_evidence import has_current_evidence
from graphtool.agents.knowledge.workflow_model_io import (
    invoke_model,
    validated_output,
)
from graphtool.agents.knowledge.workflow_research import (
    fallback_from_unavailable_expansion,
)
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def make_evaluate_node(evaluator_model):
    def evaluate(state: AgentState) -> dict:
        round_number = state["retrieval_count"]
        evaluation_messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=evaluation_text(state)),
        ]
        evaluation_result, duration = invoke_model(
            evaluator_model,
            evaluation_messages,
            stage=f"evidence evaluation round {round_number}",
        )
        decision = validated_output(
            SufficiencyDecision,
            evaluation_result,
        )
        if decision.verdict == "sufficient" and not has_current_evidence(state):
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
        if decision.verdict == "direct" and (
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
        decision = fallback_from_unavailable_expansion(state, decision)
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

    return evaluate
