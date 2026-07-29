"""Builds the LangGraph StateGraph wiring Researcher -> Writer -> Critic,
with a conditional revision loop back to the Writer."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents import make_critic, make_researcher, make_writer, route_after_critic
from .llm import LLM, get_llm
from .state import CrewState


def build_graph(llm: LLM | None = None):
    llm = llm or get_llm()

    graph = StateGraph(CrewState)
    graph.add_node("researcher", make_researcher(llm))
    graph.add_node("writer", make_writer(llm))
    graph.add_node("critic", make_critic(llm))

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"revise": "writer", "done": END})

    return graph.compile()


def run_crew(topic: str, max_revisions: int = 2, llm: LLM | None = None) -> CrewState:
    app = build_graph(llm)
    initial_state: CrewState = {
        "topic": topic,
        "revision_count": 0,
        "max_revisions": max_revisions,
        "trace": [],
    }
    return app.invoke(initial_state)
