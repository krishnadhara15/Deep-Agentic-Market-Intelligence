"""LangGraph workflow for Deep Agentic Market Intelligence.

Branch-aware orchestration:
  plan -> (parallel) research -> knowledge_graph -> verifier
                                                      |-> (gaps) back to research
                                                      |-> (sufficient) synthesize -> report
"""

from typing import List, Literal, Union

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import Config
from src.nodes import (
    knowledge_graph_node,
    plan_node,
    research_node,
    synthesize_node,
    verifier_node,
    write_report_node,
)
from src.nodes_tavily import (
    knowledge_graph_node_tavily,
    plan_node_tavily,
    research_node_tavily,
    synthesize_node_tavily,
    verifier_node_tavily,
    write_report_node_tavily,
)
from src.state import ResearchState, SubQuestion


def _branch(sq: SubQuestion, state: ResearchState, config: Config) -> Send:
    return Send(
        "research",
        {
            "sub_question": sq,
            "research_question": state["research_question"],
            "target": state.get("target", config.target),
            "loop": state.get("loop_count", 0),
        },
    )


def build_graph(config: Config):
    """Build and compile the deep-agent market-intelligence workflow."""
    use_tavily = config.llm_provider == "tavily"

    nodes = {
        "plan": plan_node_tavily if use_tavily else plan_node,
        "research": research_node_tavily if use_tavily else research_node,
        "knowledge_graph": knowledge_graph_node_tavily if use_tavily else knowledge_graph_node,
        "verifier": verifier_node_tavily if use_tavily else verifier_node,
        "synthesize": synthesize_node_tavily if use_tavily else synthesize_node,
        "write_report": write_report_node_tavily if use_tavily else write_report_node,
    }

    def dispatch_research(state: ResearchState) -> List[Send]:
        return [_branch(sq, state, config) for sq in state["sub_questions"]]

    def route_after_verify(
        state: ResearchState,
    ) -> Union[List[Send], Literal["synthesize"]]:
        if state.get("research_sufficient", True):
            return "synthesize"
        pending = state.get("pending_sub_questions", [])
        if not pending:
            return "synthesize"
        return [_branch(sq, state, config) for sq in pending]

    workflow = StateGraph(ResearchState)
    workflow.add_node("plan", lambda s: nodes["plan"](s, config))
    workflow.add_node("research", lambda s: nodes["research"](s, config))
    workflow.add_node("knowledge_graph", lambda s: nodes["knowledge_graph"](s, config))
    workflow.add_node("verifier", lambda s: nodes["verifier"](s, config))
    workflow.add_node("synthesize", lambda s: nodes["synthesize"](s, config))
    workflow.add_node("write_report", lambda s: nodes["write_report"](s, config))

    workflow.add_edge(START, "plan")
    workflow.add_conditional_edges("plan", dispatch_research, ["research"])
    workflow.add_edge("research", "knowledge_graph")
    workflow.add_edge("knowledge_graph", "verifier")
    workflow.add_conditional_edges(
        "verifier", route_after_verify, ["research", "synthesize"]
    )
    workflow.add_edge("synthesize", "write_report")
    workflow.add_edge("write_report", END)

    return workflow.compile()
