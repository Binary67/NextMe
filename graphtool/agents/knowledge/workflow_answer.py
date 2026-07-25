import logging
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graphtool.agents.knowledge.prompts import (
    ANSWER_SYSTEM_PROMPT,
    NO_EVIDENCE_ANSWER_SYSTEM_PROMPT,
)
from graphtool.agents.knowledge.state import (
    AgentResponse,
    AgentState,
    FinalAnswerDraft,
)
from graphtool.agents.knowledge.workflow_context import answer_text
from graphtool.agents.knowledge.workflow_model_io import (
    invoke_model,
    validated_output,
)
from graphtool.run_logging import LOGGER_NAME
from graphtool.sequences import unique_ordered

RUN_LOGGER = logging.getLogger(LOGGER_NAME)
NO_EVIDENCE_DISCLOSURE = (
    "I couldn't find supporting information in the knowledge base. The following "
    "is a best-effort answer based on general knowledge and is not verified "
    "against the knowledge base."
)


def make_answer_node(answer_draft_model):
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
        answer_result, _ = invoke_model(
            answer_draft_model,
            answer_messages,
            stage="answer generation",
        )
        draft = validated_output(
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
            retry_result, _ = invoke_model(
                answer_draft_model,
                retry_messages,
                stage="answer citation retry",
            )
            draft = validated_output(
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

    return answer
