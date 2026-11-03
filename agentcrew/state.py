"""Shared state schema for the AgentCrew LangGraph graph.

The ``Annotated[..., operator.add]`` reducers on ``subtopic_findings``,
``trace``, and ``usage_log`` are essential to this design: when the
researcher node fans out into several concurrent branches via LangGraph's
``Send`` API, each branch returns a *partial* state update. Without a
reducer, LangGraph would see multiple branches writing to the same state key
in the same "superstep" and raise an ``InvalidUpdateError`` for a
conflicting write. With ``operator.add`` as the reducer, LangGraph instead
concatenates every branch's single-element list onto the running list, which
is exactly how results from real, concurrent parallel work are supposed to
merge back into shared state.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, List, Optional, TypedDict


@dataclass
class SubtopicFinding:
    """Findings produced by a single researcher branch for one subtopic."""

    subtopic: str
    findings: str
    sources: List[str] = field(default_factory=list)


@dataclass
class FactCheckResult:
    """Output of the fact-checker node."""

    groundedness_score: float
    flagged_sentences: List[str] = field(default_factory=list)
    summary: str = ""


class CrewState(TypedDict, total=False):
    """The full state threaded through the AgentCrew graph."""

    # Top-level input
    topic: str

    # Set by the planner; fanned out to parallel researcher branches
    subtopics: List[str]

    # Per-branch field: only meaningful inside a single researcher node
    # invocation (each Send carries its own copy, it is never read back from
    # merged state).
    subtopic: str

    # Accumulated across all parallel researcher branches via operator.add
    subtopic_findings: Annotated[List[SubtopicFinding], operator.add]

    # Produced by the aggregate_research fan-in node
    research_notes: str

    # Writer/critic revision loop
    draft: str
    critique: str
    approved: bool
    revision_count: int
    max_revisions: int

    # Fact-checker output
    fact_check: Optional[FactCheckResult]

    # Observability: accumulated log lines and per-call usage records
    trace: Annotated[List[str], operator.add]
    usage_log: Annotated[List[dict], operator.add]
