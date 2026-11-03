#!/usr/bin/env python3
"""AgentCrew demo: runs the full graph offline and prints a readable trace.

    python demo.py
    python demo.py "The future of quantum computing"

This script is meant to be read top-to-bottom as a tour of the pipeline: the
planner's subtopics, each parallel researcher's findings, the writer/critic
revision loop, the fact-checker's groundedness verdict, and the aggregated
token/cost usage across every LLM call in the run.
"""

from __future__ import annotations

import sys

from agentcrew.graph import run_crew
from agentcrew.llm import FakeLLM
from agentcrew.usage import aggregate_usage

DEFAULT_TOPIC = "The impact of artificial intelligence on scientific research"


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC

    _print_header(f"AgentCrew demo run: '{topic}'")
    print("Running fully offline with FakeLLM + the offline WebSearchTool knowledge base...")

    result = run_crew(topic=topic, max_revisions=2, llm=FakeLLM())

    _print_header("1. Planner")
    print(f"[planner] decomposed '{topic}' into {len(result['subtopics'])} subtopics")
    for i, subtopic in enumerate(result["subtopics"], start=1):
        print(f"  {i}. {subtopic}")

    _print_header("2. Parallel Researchers (fan-out via Send)")
    for finding in result["subtopic_findings"]:
        print(f"\n[researcher] subtopic: {finding.subtopic}")
        print(f"  sources: {', '.join(finding.sources) if finding.sources else 'none'}")
        for line in finding.findings.splitlines():
            print(f"  {line}")

    _print_header("3. Aggregated Research Notes (fan-in)")
    print(result["research_notes"][:800])
    if len(result["research_notes"]) > 800:
        print("... (truncated)")

    _print_header("4. Writer / Critic Revision Loop")
    print(f"Revisions used: {result['revision_count']} / {result['max_revisions']}")
    print(f"Approved: {result['approved']}")
    print(f"Final critique: {result['critique']}")

    _print_header("5. Final Draft")
    print(result["draft"])

    _print_header("6. Fact-Check")
    fact_check = result["fact_check"]
    if fact_check is not None:
        print(f"Groundedness score: {fact_check.groundedness_score}")
        print(f"Summary: {fact_check.summary}")
        if fact_check.flagged_sentences:
            print("Flagged (potentially unsupported) sentences:")
            for sentence in fact_check.flagged_sentences:
                print(f"  - {sentence}")
        else:
            print("No unsupported sentences flagged.")

    _print_header("7. Full Trace")
    for i, entry in enumerate(result["trace"], start=1):
        print(f"{i:2d}. {entry}")

    _print_header("8. Aggregated Usage")
    usage_summary = aggregate_usage(result["usage_log"])
    print(f"Total LLM calls: {usage_summary['total_calls']}")
    print(f"Total tokens: {usage_summary['total_tokens']}")
    print(f"Estimated cost (USD): {usage_summary['total_estimated_cost_usd']}")
    print("By role:")
    for role, stats in usage_summary["by_role"].items():
        print(f"  {role:14s} calls={stats['calls']:2d} tokens={stats['input_tokens'] + stats['output_tokens']:4d}")

    print()
    print("Demo complete.")


if __name__ == "__main__":
    main()
