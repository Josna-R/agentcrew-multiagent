"""Shared state that flows through the LangGraph multi-agent graph."""
from __future__ import annotations

from typing import TypedDict


class CrewState(TypedDict, total=False):
    topic: str
    research_notes: str
    draft: str
    critique: str
    approved: bool
    revision_count: int
    max_revisions: int
    trace: list[str]  # human-readable log of which agent ran and what it produced
