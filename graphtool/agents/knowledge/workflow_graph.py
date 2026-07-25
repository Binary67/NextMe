from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graphtool.agents.knowledge.state import (
    AgentState,
    ConversationSummary,
    FinalAnswerDraft,
    QueryDecomposition,
    SufficiencyDecision,
)
from graphtool.agents.knowledge.tools import create_knowledge_tools
from graphtool.agents.knowledge.workflow_answer import make_answer_node
from graphtool.agents.knowledge.workflow_compaction import make_compact_node
from graphtool.agents.knowledge.workflow_decomposition import (
    make_decompose_node,
)
from graphtool.agents.knowledge.workflow_evaluation import make_evaluate_node
from graphtool.agents.knowledge.workflow_nodes import (
    advance_subquestion,
    cleanup,
    complete_subquestion,
    finish_conversation,
    record_tool_results,
)
from graphtool.agents.knowledge.workflow_research import make_research_node
from graphtool.agents.knowledge.workflow_routing import (
    route_completed_subquestion,
    route_decomposition,
    route_evaluation,
    route_research,
)
from graphtool.runtime import GraphToolRuntime


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
    compact = make_compact_node(
        orchestration_model.with_structured_output(ConversationSummary),
        compaction_trigger_tokens=compaction_trigger_tokens,
        retained_recent_tokens=retained_recent_tokens,
    )
    decompose = make_decompose_node(
        orchestration_model.with_structured_output(QueryDecomposition),
        runtime,
    )
    research = make_research_node(
        orchestration_model.bind_tools(tools, parallel_tool_calls=False)
    )
    evaluate = make_evaluate_node(
        answer_model.with_structured_output(SufficiencyDecision)
    )
    answer = make_answer_node(
        answer_model.with_structured_output(FinalAnswerDraft)
    )

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
        route_decomposition,
        {
            "research": "research",
            "finish_conversation": "finish_conversation",
        },
    )
    builder.add_conditional_edges(
        "research",
        route_research,
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
        route_evaluation,
        {
            "finish_conversation": "finish_conversation",
            "research": "research",
            "complete_subquestion": "complete_subquestion",
        },
    )
    builder.add_conditional_edges(
        "complete_subquestion",
        route_completed_subquestion,
        {"advance_subquestion": "advance_subquestion", "answer": "answer"},
    )
    builder.add_edge("advance_subquestion", "research")
    builder.add_edge("answer", "cleanup")
    builder.add_edge("finish_conversation", "cleanup")
    builder.add_edge("cleanup", END)
    return builder.compile(checkpointer=checkpointer)
