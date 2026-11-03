"""The AgentCrew LangGraph wiring.

::

    planner --(Send fan-out, one per subtopic)--> researcher (parallel)
                                                        |
                                                        v
                                              aggregate_research (fan-in)
                                                        |
                                                        v
                                                     writer <---------+
                                                        |             |
                                                        v             | "revise"
                                                     critic -----------+
                                                        |
                                                        | "fact_check"
                                                        v
                                                  fact_checker --> END

``_route_to_researchers`` is a conditional-edge path function that returns a
list of :class:`langgraph.types.Send` objects -- one per entry in
``state["subtopics"]`` -- rather than a hardcoded set of parallel branches.
This means the degree of parallelism is *data-driven*: if the planner
decomposes a topic into 3 subtopics, 3 researcher branches run concurrently;
if a future planner decomposed it into 5, 5 branches would run, with no
graph-definition changes required.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agentcrew.agents import (
    aggregate_research_node,
    make_critic,
    make_fact_checker,
    make_planner,
    make_researcher,
    make_writer,
    route_after_critic,
)
from agentcrew.llm import LLM, FakeLLM
from agentcrew.state import CrewState


# CrewState stores plain dataclasses (SubtopicFinding, FactCheckResult) that
# are not LangChain/langgraph built-ins, so checkpoint (de)serialization needs
# to be told these two modules/classes are safe to reconstruct across process
# boundaries (e.g. a CLI --resume in a fresh process reading a sqlite
# checkpoint). Without this, newer langgraph releases log a deprecation
# warning and, eventually, refuse to deserialize them at all.
def default_serde() -> JsonPlusSerializer:
    """A JsonPlusSerializer that explicitly allows AgentCrew's state dataclasses."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("agentcrew.state", "SubtopicFinding"),
            ("agentcrew.state", "FactCheckResult"),
        ]
    )


def _route_to_researchers(state: CrewState) -> List[Send]:
    """Dynamic parallel fan-out: one ``Send`` to "researcher" per subtopic."""
    return [
        Send("researcher", {"subtopic": subtopic, "topic": state["topic"]})
        for subtopic in state["subtopics"]
    ]


def build_graph(llm: LLM, checkpointer: Optional[BaseCheckpointSaver] = None):
    """Wire up and compile the AgentCrew graph for the given ``llm`` backend."""
    graph = StateGraph(CrewState)

    graph.add_node("planner", make_planner(llm))
    graph.add_node("researcher", make_researcher(llm))
    graph.add_node("aggregate_research", aggregate_research_node)
    graph.add_node("writer", make_writer(llm))
    graph.add_node("critic", make_critic(llm))
    graph.add_node("fact_checker", make_fact_checker(llm))

    graph.set_entry_point("planner")

    # Dynamic parallel fan-out: branch count comes from len(state["subtopics"]),
    # never hardcoded.
    graph.add_conditional_edges("planner", _route_to_researchers, ["researcher"])

    # Fan-in: every researcher branch converges here before the writer runs.
    graph.add_edge("researcher", "aggregate_research")
    graph.add_edge("aggregate_research", "writer")
    graph.add_edge("writer", "critic")

    # Bounded writer/critic revision loop, falling through to fact_checker
    # once approved or once max_revisions is exhausted.
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"revise": "writer", "fact_check": "fact_checker"},
    )
    graph.add_edge("fact_checker", END)

    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver(serde=default_serde())
    return graph.compile(checkpointer=resolved_checkpointer)


def _initial_state(topic: str, max_revisions: int) -> CrewState:
    return {
        "topic": topic,
        "subtopics": [],
        "subtopic_findings": [],
        "research_notes": "",
        "draft": "",
        "critique": "",
        "approved": False,
        "revision_count": 0,
        "max_revisions": max_revisions,
        "fact_check": None,
        "trace": [],
        "usage_log": [],
    }


def run_crew(
    topic: str,
    max_revisions: int = 2,
    llm: Optional[LLM] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    thread_id: Optional[str] = None,
) -> CrewState:
    """Compile and run the AgentCrew graph for ``topic``.

    When ``thread_id`` refers to a thread that already has a saved
    checkpoint on ``checkpointer`` (e.g. a prior run that raised partway
    through, or simply a previous completed run), this resumes execution
    from that checkpoint instead of starting over from the planner -- this
    is how AgentCrew supports crash recovery / resumable runs. A brand-new
    ``thread_id`` (the default: a fresh UUID) always starts a clean run.
    """
    resolved_llm = llm if llm is not None else FakeLLM()
    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    app = build_graph(resolved_llm, checkpointer=resolved_checkpointer)

    resolved_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": resolved_thread_id}}

    existing_checkpoint = resolved_checkpointer.get_tuple(config)
    if existing_checkpoint is not None:
        # Resume: LangGraph continues from the last saved checkpoint for this
        # thread when invoked with no new input.
        result = app.invoke(None, config=config)
    else:
        result = app.invoke(_initial_state(topic, max_revisions), config=config)

    return result
