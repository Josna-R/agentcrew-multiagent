"""Unit tests for individual AgentCrew node functions in agentcrew.agents."""

from __future__ import annotations

from agentcrew.agents import (
    aggregate_research_node,
    make_fact_checker,
    make_planner,
    make_researcher,
    route_after_critic,
)
from agentcrew.llm import FakeLLM
from agentcrew.state import SubtopicFinding


def test_make_planner_produces_exactly_three_subtopics():
    planner = make_planner(FakeLLM())
    update = planner({"topic": "Autonomous vehicles"})
    assert len(update["subtopics"]) == 3
    assert update["trace"][0].startswith("[planner]")
    assert update["usage_log"][0]["role"] == "planner"


def test_make_researcher_returns_one_element_finding_list():
    researcher = make_researcher(FakeLLM())
    update = researcher({"subtopic": "Battery technology", "topic": "Autonomous vehicles"})
    assert len(update["subtopic_findings"]) == 1
    finding = update["subtopic_findings"][0]
    assert isinstance(finding, SubtopicFinding)
    assert finding.subtopic == "Battery technology"
    assert finding.sources  # offline KB should return at least one source


def test_aggregate_research_node_combines_all_findings():
    state = {
        "subtopic_findings": [
            SubtopicFinding(subtopic="A", findings="findings about A", sources=["kb-001"]),
            SubtopicFinding(subtopic="B", findings="findings about B", sources=["kb-002"]),
        ]
    }
    update = aggregate_research_node(state)
    assert "A" in update["research_notes"]
    assert "B" in update["research_notes"]
    assert "findings about A" in update["research_notes"]


def test_route_after_critic_revises_when_not_approved_and_under_budget():
    state = {"approved": False, "revision_count": 0, "max_revisions": 2}
    assert route_after_critic(state) == "revise"


def test_route_after_critic_moves_on_when_approved():
    state = {"approved": True, "revision_count": 0, "max_revisions": 2}
    assert route_after_critic(state) == "fact_check"


def test_route_after_critic_moves_on_when_budget_exhausted():
    state = {"approved": False, "revision_count": 2, "max_revisions": 2}
    assert route_after_critic(state) == "fact_check"


def test_make_fact_checker_flags_unsupported_sentences():
    fact_checker = make_fact_checker(FakeLLM())
    state = {
        "draft": "# Report\nSolar power is driven by falling manufacturing costs.\nDragons can breathe fire.",
        "research_notes": "Solar photovoltaic and wind power have become the fastest growing sources of "
        "electricity generation worldwide, driven by falling manufacturing costs and supportive policy.",
    }
    update = fact_checker(state)
    fact_check = update["fact_check"]
    assert 0.0 <= fact_check.groundedness_score <= 1.0
    assert any("Dragons" in s for s in fact_check.flagged_sentences)
    assert not any("Solar power" in s for s in fact_check.flagged_sentences)


def test_make_fact_checker_perfect_score_when_research_notes_empty_and_draft_empty():
    fact_checker = make_fact_checker(FakeLLM())
    state = {"draft": "", "research_notes": ""}
    update = fact_checker(state)
    assert update["fact_check"].groundedness_score == 1.0
