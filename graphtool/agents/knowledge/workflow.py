import logging
from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from graphtool.agents.knowledge.state import AgentResponse
from graphtool.agents.knowledge.workflow_graph import build_workflow_graph
from graphtool.run_logging import LOGGER_NAME
from graphtool.runtime import GraphToolRuntime

DEFAULT_COMPACTION_TRIGGER_TOKENS = 256_000
DEFAULT_RETAINED_RECENT_TOKENS = 64_000
RUN_LOGGER = logging.getLogger(LOGGER_NAME)


class KnowledgeAgent:
    def __init__(
        self,
        answer_model: BaseChatModel,
        orchestration_model: BaseChatModel,
        runtime: GraphToolRuntime,
        *,
        compaction_trigger_tokens: int = DEFAULT_COMPACTION_TRIGGER_TOKENS,
        retained_recent_tokens: int = DEFAULT_RETAINED_RECENT_TOKENS,
    ) -> None:
        if compaction_trigger_tokens < 1:
            raise ValueError("Compaction trigger token count must be positive.")
        if retained_recent_tokens < 1:
            raise ValueError("Retained recent token count must be positive.")
        if retained_recent_tokens >= compaction_trigger_tokens:
            raise ValueError(
                "Retained recent token count must be less than the "
                "compaction trigger."
            )
        self._checkpointer = InMemorySaver()
        self._graph = build_workflow_graph(
            answer_model,
            orchestration_model,
            runtime,
            self._checkpointer,
            compaction_trigger_tokens=compaction_trigger_tokens,
            retained_recent_tokens=retained_recent_tokens,
        )

    def ask(self, question: str, *, thread_id: str) -> AgentResponse:
        started_at = perf_counter()
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("Thread ID must not be empty.")
        RUN_LOGGER.info("Agent processing started")

        config = {
            "configurable": {"thread_id": normalized_thread_id},
            "recursion_limit": 150,
        }
        if self._pending_interrupt(config) is not None:
            result = self._graph.invoke(
                Command(resume=normalized_question),
                config=config,
            )
        else:
            result = self._graph.invoke(
                {
                    "messages": [HumanMessage(content=normalized_question)],
                    "question": normalized_question,
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
                    "response": None,
                },
                config=config,
            )
        pending_interrupt = self._pending_interrupt(config)
        if pending_interrupt is not None:
            payload = pending_interrupt.value
            if (
                not isinstance(payload, dict)
                or payload.get("kind") != "ask_user"
                or not isinstance(payload.get("question"), str)
                or not payload["question"].strip()
            ):
                raise RuntimeError("Knowledge agent returned an invalid interrupt.")
            response = AgentResponse(
                answer=payload["question"].strip(),
                status="needs_input",
                references=[],
                search_count=result.get("search_count", 0),
            )
            RUN_LOGGER.info(
                "Agent processing paused in %.2fs for user input",
                perf_counter() - started_at,
            )
            return response
        response = result.get("response")
        if not isinstance(response, AgentResponse):
            raise RuntimeError("Knowledge agent completed without a response.")
        RUN_LOGGER.info(
            "Agent processing completed in %.2fs: status=%s, searches=%d, "
            "references=%d",
            perf_counter() - started_at,
            response.status,
            response.search_count,
            len(response.references),
        )
        self._checkpointer.delete_thread(normalized_thread_id)
        checkpoint_state = {**result, "response": None}
        self._graph.update_state(config, checkpoint_state, as_node="cleanup")
        return response

    def _pending_interrupt(self, config: dict):
        state = self._graph.get_state(config)
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0]
        return None

    def reset(self, thread_id: str) -> None:
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("Thread ID must not be empty.")
        self._checkpointer.delete_thread(normalized_thread_id)


def create_knowledge_agent(
    answer_model: BaseChatModel,
    orchestration_model: BaseChatModel,
    runtime: GraphToolRuntime,
    *,
    compaction_trigger_tokens: int = DEFAULT_COMPACTION_TRIGGER_TOKENS,
    retained_recent_tokens: int = DEFAULT_RETAINED_RECENT_TOKENS,
) -> KnowledgeAgent:
    return KnowledgeAgent(
        answer_model,
        orchestration_model,
        runtime,
        compaction_trigger_tokens=compaction_trigger_tokens,
        retained_recent_tokens=retained_recent_tokens,
    )
