import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentcrew.graph import run_crew
from agentcrew.llm import FakeLLM, LLM


def test_fake_llm_smoke_run_produces_report():
    result = run_crew("Vector databases for RAG", llm=FakeLLM())
    assert "Report:" in result["draft"]
    assert "Recommendation" in result["draft"]
    assert result["trace"][0].startswith("[researcher]")
    assert any(t.startswith("[writer]") for t in result["trace"])
    assert any(t.startswith("[critic]") for t in result["trace"])


class RevisesThenApprovesLLM(LLM):
    """Stub that forces exactly one revision loop, to prove the
    critic -> writer conditional edge actually re-enters the writer node
    and that the revision counter and cap work as designed."""

    def __init__(self):
        self.critic_calls = 0

    def complete(self, role: str, prompt: str) -> str:
        if role == "researcher":
            return "Finding 1. Finding 2."
        if role == "writer":
            return f"Draft v{self.critic_calls}"
        if role == "critic":
            self.critic_calls += 1
            return "REVISE: needs more detail" if self.critic_calls == 1 else "APPROVE: looks good"
        return "n/a"


def test_revision_loop_runs_exactly_once_then_approves():
    llm = RevisesThenApprovesLLM()
    result = run_crew("test topic", max_revisions=2, llm=llm)
    assert result["approved"] is True
    assert result["revision_count"] == 1
    assert llm.critic_calls == 2
    assert result["draft"] == "Draft v1"


class AlwaysRevisesLLM(LLM):
    """Stub that never approves, to prove the max_revisions cap terminates the graph
    instead of looping forever."""

    def complete(self, role: str, prompt: str) -> str:
        if role == "researcher":
            return "Finding."
        if role == "writer":
            return "Draft"
        if role == "critic":
            return "REVISE: never good enough"
        return "n/a"


def test_max_revisions_cap_terminates_graph():
    result = run_crew("test topic", max_revisions=2, llm=AlwaysRevisesLLM())
    assert result["approved"] is False
    assert result["revision_count"] == 2  # stopped at the cap, did not loop forever
