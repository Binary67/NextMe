import logging

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)

from graphtool.agents.knowledge.prompts import SUMMARY_SYSTEM_PROMPT
from graphtool.agents.knowledge.state import AgentState, ConversationSummary
from graphtool.agents.knowledge.workflow_context import (
    conversation_token_count,
    summary_text,
)
from graphtool.agents.knowledge.workflow_model_io import (
    invoke_model,
    validated_output,
)
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def make_compact_node(
    summary_model,
    *,
    compaction_trigger_tokens: int,
    retained_recent_tokens: int,
):
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
        summary_result, duration = invoke_model(
            summary_model,
            summary_messages,
            stage="conversation summary",
        )
        updated_summary = validated_output(
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

    return compact
