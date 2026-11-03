"""Node functions (agents) for the AgentCrew graph.

Each ``make_*`` factory takes an :class:`~agentcrew.llm.LLM` and returns a
plain function ``state -> partial_state_update`` suitable for use as a
LangGraph node. Keeping the LLM as a closure argument (rather than a global)
is what lets ``build_graph`` wire the exact same node functions to a
``FakeLLM`` in tests/demo or a real ``OpenAILLM``/``AnthropicLLM`` in
production, with no other code changes.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal

from agentcrew.llm import LLM
from agentcrew.state import CrewState, FactCheckResult, SubtopicFinding
from agentcrew.tools import WebSearchTool
from agentcrew.usage import estimate_tokens, record_usage

# Minimal English stopwords filtered out before computing word overlap, so
# groundedness scoring in make_fact_checker isn't dominated by filler words.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "was",
    "were", "for", "with", "as", "by", "this", "that", "it", "its", "at",
    "be", "from", "into", "report", "summary", "conclusion",
}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def make_planner(llm: LLM):
    """Build the planner node: decomposes ``state['topic']`` into 3 subtopics."""

    def planner(state: CrewState) -> Dict:
        topic = state["topic"]
        raw = llm.generate(topic, role="planner")
        # Parsing is intentionally dumb: one subtopic per non-empty line. The
        # number of subtopics returned here -- not a constant baked into this
        # function or into the graph -- is what determines how many parallel
        # researcher branches _route_to_researchers fans out to. FakeLLM's
        # planner always returns exactly 3, but nothing here enforces that.
        subtopics = [line.strip("-* \t") for line in raw.splitlines() if line.strip()]
        if not subtopics:
            subtopics = [topic]

        usage = record_usage(
            role="planner",
            input_tokens=estimate_tokens(topic),
            output_tokens=estimate_tokens(raw),
            model=llm.model_name,
        )
        return {
            "subtopics": subtopics,
            "trace": [f"[planner] decomposed '{topic}' into {len(subtopics)} subtopics"],
            "usage_log": [usage],
        }

    return planner


def make_researcher(llm: LLM, search_tool: WebSearchTool | None = None):
    """Build the researcher node, run once per parallel branch via ``Send``.

    Reads the per-branch ``state['subtopic']``, searches the (offline) web
    search tool, asks the LLM (role="researcher") to write findings, and
    returns ``subtopic_findings`` as a *one-element list* -- the
    ``operator.add`` reducer on that state key is what accumulates every
    branch's single finding into one combined list, regardless of how many
    branches ran concurrently.
    """

    tool = search_tool or WebSearchTool()

    def researcher(state: CrewState) -> Dict:
        subtopic = state["subtopic"]
        hits = tool.run(subtopic, top_k=2)
        findings_text = llm.generate(subtopic, role="researcher")

        usage = record_usage(
            role="researcher",
            input_tokens=estimate_tokens(subtopic),
            output_tokens=estimate_tokens(findings_text),
            model=llm.model_name,
        )
        finding = SubtopicFinding(
            subtopic=subtopic,
            findings=findings_text,
            sources=[hit.source_id for hit in hits],
        )
        return {
            "subtopic_findings": [finding],
            "trace": [f"[researcher] gathered {len(hits)} source(s) for '{subtopic}'"],
            "usage_log": [usage],
        }

    return researcher


def aggregate_research_node(state: CrewState) -> Dict:
    """Fan-in step: combine every branch's findings into one research_notes string."""
    findings: List[SubtopicFinding] = state.get("subtopic_findings", [])
    sections = []
    for item in findings:
        sources = ", ".join(item.sources) if item.sources else "no sources"
        sections.append(f"### {item.subtopic}\n{item.findings}\n(Sources: {sources})")
    research_notes = "\n\n".join(sections)
    return {
        "research_notes": research_notes,
        "trace": [f"[aggregate_research] combined {len(findings)} subtopic finding(s)"],
    }


def make_writer(llm: LLM):
    """Build the writer node: drafts (or revises) a report from research_notes."""

    def writer(state: CrewState) -> Dict:
        research_notes = state.get("research_notes", "")
        revision_count = state.get("revision_count", 0)
        critique = state.get("critique", "")

        if revision_count > 0 and critique:
            prompt = (
                f"Revise the report using this critique:\n{critique}\n\n"
                f"Research notes:\n{research_notes}"
            )
        else:
            prompt = f"Research notes:\n{research_notes}"

        draft = llm.generate(prompt, role="writer")

        usage = record_usage(
            role="writer",
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(draft),
            model=llm.model_name,
        )
        action = "revised" if revision_count > 0 else "drafted"
        return {
            "draft": draft,
            "trace": [f"[writer] {action} report (revision {revision_count})"],
            "usage_log": [usage],
        }

    return writer


def make_critic(llm: LLM):
    """Build the critic node: reviews the draft's structure and sets approved/critique.

    The prompt embeds the current ``revision_count`` so that ``FakeLLM``'s
    deterministic critic rule (always REVISE on the very first pass, then
    APPROVE) is driven by real graph state rather than hidden logic, giving
    a realistic "revise once, then approve" loop in the offline demo while
    remaining perfectly deterministic for tests.
    """

    def critic(state: CrewState) -> Dict:
        draft = state.get("draft", "")
        revision_count = state.get("revision_count", 0)
        prompt = f"REVISION_COUNT: {revision_count}\n\nDRAFT:\n{draft}"

        response = llm.generate(prompt, role="critic")
        approved = response.strip().upper().startswith("APPROVE")

        usage = record_usage(
            role="critic",
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(response),
            model=llm.model_name,
        )
        verdict = "APPROVE" if approved else "REVISE"
        return {
            "approved": approved,
            "critique": response,
            "revision_count": revision_count + 1,
            "trace": [f"[critic] verdict={verdict} after revision {revision_count}"],
            "usage_log": [usage],
        }

    return critic


def make_fact_checker(llm: LLM):
    """Build the fact-checker node: scores the draft's groundedness in research_notes.

    This deliberately duplicates none of the critic's job: the critic judges
    structure/completeness of the draft, while the fact-checker verifies
    that claims in the draft are actually supported by the research notes --
    a different failure mode (a well-structured report can still invent
    facts not present in any source).
    """

    def fact_checker(state: CrewState) -> Dict:
        draft = state.get("draft", "")
        research_notes = state.get("research_notes", "")

        # Invoke the LLM so usage is tracked and real backends get a chance
        # to produce a richer qualitative summary; FakeLLM just echoes input,
        # so the actual scoring below is computed deterministically here.
        llm_response = llm.generate(f"DRAFT:\n{draft}\n\nRESEARCH NOTES:\n{research_notes}", role="fact_checker")

        notes_words = _words(research_notes)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", draft) if s.strip()]
        # Ignore markdown heading lines ("# Report", "## Summary", ...) which
        # carry no factual claims to ground.
        content_sentences = [s for s in sentences if not s.lstrip().startswith("#")]

        flagged: List[str] = []
        supported = 0
        for sentence in content_sentences:
            sentence_words = _words(sentence)
            if not sentence_words:
                supported += 1  # nothing to ground, don't penalize
                continue
            overlap = sentence_words & notes_words
            if notes_words and len(overlap) / len(sentence_words) >= 0.2:
                supported += 1
            else:
                flagged.append(sentence)

        groundedness = supported / len(content_sentences) if content_sentences else 1.0

        usage = record_usage(
            role="fact_checker",
            input_tokens=estimate_tokens(draft) + estimate_tokens(research_notes),
            output_tokens=estimate_tokens(llm_response),
            model=llm.model_name,
        )
        fact_check = FactCheckResult(
            groundedness_score=round(groundedness, 3),
            flagged_sentences=flagged,
            summary=f"{supported}/{len(content_sentences)} sentences grounded in research notes.",
        )
        return {
            "fact_check": fact_check,
            "trace": [f"[fact_checker] groundedness={fact_check.groundedness_score}"],
            "usage_log": [usage],
        }

    return fact_checker


def route_after_critic(state: CrewState) -> Literal["revise", "fact_check"]:
    """Conditional edge after the critic: loop back to the writer, or move on.

    Returns "revise" only while the draft is not yet approved AND the
    bounded revision budget has not been exhausted, guaranteeing the
    writer/critic loop always terminates even if the critic never approves.
    """
    approved = state.get("approved", False)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)
    if not approved and revision_count < max_revisions:
        return "revise"
    return "fact_check"
