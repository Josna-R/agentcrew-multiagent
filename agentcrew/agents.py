"""Individual agent node functions. Each takes/returns a CrewState dict, the
shape LangGraph expects for a node in a StateGraph."""
from __future__ import annotations

from .llm import LLM
from .state import CrewState


def make_researcher(llm: LLM):
    def researcher_node(state: CrewState) -> CrewState:
        prompt = f"You are a research analyst. Topic: {state['topic']}\nProvide 3-5 key findings."
        notes = llm.complete("researcher", prompt)
        trace = state.get("trace", []) + [f"[researcher] gathered notes on '{state['topic']}'"]
        return {**state, "research_notes": notes, "trace": trace}

    return researcher_node


def make_writer(llm: LLM):
    def writer_node(state: CrewState) -> CrewState:
        feedback = f"\n\nPrevious critique to address: {state['critique']}" if state.get("critique") else ""
        prompt = (
            f"Topic: {state['topic']}\nResearch notes: {state['research_notes']}{feedback}\n"
            "Write a structured report with an Overview, Key Findings, and a Recommendation section."
        )
        draft = llm.complete("writer", prompt)
        revision = state.get("revision_count", 0) + (1 if state.get("critique") else 0)
        trace = state.get("trace", []) + [f"[writer] drafted report (revision {revision})"]
        return {**state, "draft": draft, "revision_count": revision, "trace": trace}

    return writer_node


def make_critic(llm: LLM):
    def critic_node(state: CrewState) -> CrewState:
        prompt = f"Review this report and either APPROVE it or explain what to REVISE:\n\n{state['draft']}"
        verdict = llm.complete("critic", prompt)
        approved = verdict.strip().upper().startswith("APPROVE")
        trace = state.get("trace", []) + [f"[critic] verdict: {'APPROVE' if approved else 'REVISE'}"]
        return {**state, "critique": verdict, "approved": approved, "trace": trace}

    return critic_node


def route_after_critic(state: CrewState) -> str:
    """Conditional edge: loop back to the writer for revisions, up to max_revisions, else finish."""
    if state.get("approved"):
        return "done"
    if state.get("revision_count", 0) >= state.get("max_revisions", 2):
        return "done"  # give up gracefully after the revision cap instead of looping forever
    return "revise"
