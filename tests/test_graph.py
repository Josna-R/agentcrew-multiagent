"""End-to-end tests for the real AgentCrew LangGraph graph (no mocking of
LangGraph internals): these exercise the actual compiled graph, the dynamic
Send-based parallel fan-out, the bounded writer/critic revision loop, and
checkpointer save/resume.
"""

from __future__ import annotations

import re

from langgraph.checkpoint.memory import MemorySaver

from agentcrew.graph import build_graph, default_serde, run_crew
from agentcrew.llm import LLM
from agentcrew.state import SubtopicFinding


class ScriptedLLM(LLM):
    """A small deterministic LLM used to drive specific graph scenarios
    precisely (e.g. forcing N revisions before approval), independent of
    FakeLLM's own default behavior."""

    def __init__(self, approve_after_revision: int = 1, researcher_fails_times: int = 0):
        self.approve_after_revision = approve_after_revision
        self.researcher_fails_times = researcher_fails_times
        self._researcher_call_count = 0

    @property
    def model_name(self) -> str:
        return "scripted-llm"

    def generate(self, prompt: str, role=None) -> str:
        if role == "planner":
            return "Alpha\nBeta\nGamma"
        if role == "researcher":
            self._researcher_call_count += 1
            if self._researcher_call_count <= self.researcher_fails_times:
                raise RuntimeError("simulated transient researcher failure")
            return f"Findings about {prompt.strip()}."
        if role == "writer":
            return "# Report\n## Summary\nContent.\n## Conclusion\nDone."
        if role == "critic":
            match = re.search(r"REVISION_COUNT:\s*(\d+)", prompt)
            revision_count = int(match.group(1)) if match else 0
            if revision_count >= self.approve_after_revision:
                return "APPROVE\nLooks good."
            return "REVISE\nNeeds more detail."
        if role == "fact_checker":
            return "checked"
        return "ok"


class TestEndToEndRun:
    def test_full_run_produces_complete_state(self):
        result = run_crew(topic="Climate change mitigation", max_revisions=2, llm=ScriptedLLM())
        assert result["topic"] == "Climate change mitigation"
        assert len(result["subtopics"]) == 3
        assert result["research_notes"]
        assert result["draft"]
        assert result["fact_check"] is not None
        assert result["approved"] is True

    def test_parallel_fan_out_produces_one_finding_per_subtopic(self):
        result = run_crew(topic="Space exploration", max_revisions=2, llm=ScriptedLLM())
        subtopics = result["subtopics"]
        findings = result["subtopic_findings"]

        assert len(findings) == len(subtopics)
        assert all(isinstance(f, SubtopicFinding) for f in findings)

        # Every subtopic the planner chose has exactly one corresponding
        # finding -- proving the Send-based fan-out ran one researcher
        # branch per subtopic and operator.add correctly accumulated every
        # branch's single-element contribution without dropping or
        # duplicating any.
        found_subtopics = sorted(f.subtopic for f in findings)
        assert found_subtopics == sorted(subtopics)

    def test_fan_out_branch_count_matches_subtopic_count_not_hardcoded(self):
        # Prove the branch count is driven by len(subtopics), not a fixed
        # constant, by using a planner that returns a different count.
        class TwoSubtopicLLM(ScriptedLLM):
            def generate(self, prompt, role=None):
                if role == "planner":
                    return "Only One\nOnly Two"
                return super().generate(prompt, role=role)

        llm = TwoSubtopicLLM()
        app = build_graph(llm, checkpointer=MemorySaver(serde=default_serde()))
        result = app.invoke(
            {
                "topic": "Topic",
                "subtopics": [],
                "subtopic_findings": [],
                "research_notes": "",
                "draft": "",
                "critique": "",
                "approved": False,
                "revision_count": 0,
                "max_revisions": 2,
                "fact_check": None,
                "trace": [],
                "usage_log": [],
            },
            config={"configurable": {"thread_id": "branch-count-test"}},
        )
        assert len(result["subtopics"]) == 2
        assert len(result["subtopic_findings"]) == 2


class TestRevisionLoop:
    def test_revision_loop_fires_then_approves(self):
        llm = ScriptedLLM(approve_after_revision=1)
        result = run_crew(topic="Robotics", max_revisions=3, llm=llm)

        assert result["approved"] is True
        # revision_count increments on every critic call (REVISE then
        # APPROVE), so two critic calls -> revision_count == 2.
        assert result["revision_count"] == 2
        revise_entries = [t for t in result["trace"] if "verdict=REVISE" in t]
        approve_entries = [t for t in result["trace"] if "verdict=APPROVE" in t]
        assert len(revise_entries) == 1
        assert len(approve_entries) == 1
        # The writer should have run twice: once to draft, once to revise.
        write_entries = [t for t in result["trace"] if t.startswith("[writer]")]
        assert len(write_entries) == 2
        assert "drafted" in write_entries[0]
        assert "revised" in write_entries[1]

    def test_max_revisions_cap_terminates_into_fact_check(self):
        # This critic never approves; without the max_revisions cap this
        # would loop forever. approve_after_revision=99 guarantees REVISE
        # every single time within any reasonable max_revisions budget.
        llm = ScriptedLLM(approve_after_revision=99)
        result = run_crew(topic="Perpetual motion", max_revisions=2, llm=llm)

        # The loop must have stopped at exactly max_revisions, never approved,
        # and still produced a fact_check result (proving the graph reached
        # fact_checker -> END instead of hanging).
        assert result["revision_count"] == 2
        assert result["approved"] is False
        assert result["fact_check"] is not None
        revise_entries = [t for t in result["trace"] if "verdict=REVISE" in t]
        assert len(revise_entries) == 2
        approve_entries = [t for t in result["trace"] if "verdict=APPROVE" in t]
        assert len(approve_entries) == 0
        fact_check_entries = [t for t in result["trace"] if t.startswith("[fact_checker]")]
        assert len(fact_check_entries) == 1

    def test_zero_max_revisions_skips_loop_entirely(self):
        llm = ScriptedLLM(approve_after_revision=99)
        result = run_crew(topic="Instant topic", max_revisions=0, llm=llm)
        assert result["revision_count"] == 1
        assert result["approved"] is False
        assert result["fact_check"] is not None


class TestCheckpointerResume:
    def test_checkpointer_persists_state_for_a_thread(self):
        checkpointer = MemorySaver(serde=default_serde())
        thread_id = "persist-thread"
        run_crew(topic="Oceanography", max_revisions=1, llm=ScriptedLLM(), checkpointer=checkpointer, thread_id=thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = checkpointer.get_tuple(config)
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.checkpoint["channel_values"]["topic"] == "Oceanography"

    def test_resume_after_mid_run_failure_continues_without_rerunning_planner(self):
        checkpointer = MemorySaver(serde=default_serde())
        thread_id = "resume-thread"
        # Only the very first researcher call (out of the 3 concurrent
        # branches) fails -- enough to fail the whole run once, without
        # relying on exactly which of the 3 racing branches gets there first.
        llm = ScriptedLLM(approve_after_revision=1, researcher_fails_times=1)

        try:
            run_crew(topic="Marine biology", max_revisions=2, llm=llm, checkpointer=checkpointer, thread_id=thread_id)
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "expected the simulated researcher failure to propagate"

        # The planner node must have completed and been checkpointed even
        # though the run as a whole failed.
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = checkpointer.get_tuple(config)
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.checkpoint["channel_values"].get("subtopics")

        # Resume: same thread_id, same checkpointer, no more simulated
        # failures left (researcher_fails_times already exhausted by the
        # calls above, so the next attempts succeed).
        result = run_crew(topic="Marine biology", max_revisions=2, llm=llm, checkpointer=checkpointer, thread_id=thread_id)
        assert result["approved"] is True
        planner_entries = [t for t in result["trace"] if t.startswith("[planner]")]
        # The planner should only have run once across both calls, proving
        # the second run resumed from the checkpoint instead of starting over.
        assert len(planner_entries) == 1

    def test_different_thread_ids_are_independent(self):
        checkpointer = MemorySaver(serde=default_serde())
        result_a = run_crew(topic="Topic A", max_revisions=1, llm=ScriptedLLM(), checkpointer=checkpointer, thread_id="thread-a")
        result_b = run_crew(topic="Topic B", max_revisions=1, llm=ScriptedLLM(), checkpointer=checkpointer, thread_id="thread-b")
        assert result_a["topic"] == "Topic A"
        assert result_b["topic"] == "Topic B"


class TestAggregateResearch:
    def test_research_notes_mentions_every_subtopic(self):
        result = run_crew(topic="Deep sea exploration", max_revisions=1, llm=ScriptedLLM())
        for subtopic in result["subtopics"]:
            assert subtopic in result["research_notes"]
