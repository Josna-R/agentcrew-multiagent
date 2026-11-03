"""Command-line interface for AgentCrew.

    python -m agentcrew.cli "The future of renewable energy"
    python -m agentcrew.cli "The future of renewable energy" --max-revisions 3 --output json
    python -m agentcrew.cli "The future of renewable energy" --thread-id my-run-1
    python -m agentcrew.cli "The future of renewable energy" --thread-id my-run-1 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver

from agentcrew.graph import default_serde, run_crew
from agentcrew.llm import FakeLLM
from agentcrew.state import CrewState
from agentcrew.usage import aggregate_usage

# A small on-disk SQLite checkpoint store, so `--resume --thread-id <id>`
# can pick a run back up in a *separate* CLI invocation (a fresh process),
# not just within one Python session.
DEFAULT_CHECKPOINT_DB = os.environ.get("AGENTCREW_CHECKPOINT_DB", ".agentcrew_checkpoints.sqlite")


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/lists/dicts into plain JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _render_markdown(result: CrewState) -> str:
    """Render a CrewState result as a readable Markdown report."""
    lines: List[str] = []
    lines.append(f"# AgentCrew Report: {result.get('topic', '')}")
    lines.append("")
    subtopics = result.get("subtopics", [])
    if subtopics:
        lines.append("## Subtopics Investigated")
        for subtopic in subtopics:
            lines.append(f"- {subtopic}")
        lines.append("")

    lines.append("## Draft Report")
    lines.append(result.get("draft", "") or "_(no draft produced)_")
    lines.append("")

    fact_check = result.get("fact_check")
    if fact_check is not None:
        lines.append("## Fact-Check")
        lines.append(f"- Groundedness score: {fact_check.groundedness_score}")
        lines.append(f"- Summary: {fact_check.summary}")
        if fact_check.flagged_sentences:
            lines.append("- Flagged (unsupported) sentences:")
            for sentence in fact_check.flagged_sentences:
                lines.append(f"  - {sentence}")
        lines.append("")

    lines.append("## Process")
    lines.append(f"- Approved: {result.get('approved')}")
    lines.append(f"- Revisions used: {result.get('revision_count')} / {result.get('max_revisions')}")
    lines.append("")

    usage_summary = aggregate_usage(result.get("usage_log", []))
    lines.append("## Usage")
    lines.append(f"- Total LLM calls: {usage_summary['total_calls']}")
    lines.append(f"- Total tokens: {usage_summary['total_tokens']}")
    lines.append(f"- Estimated cost (USD): {usage_summary['total_estimated_cost_usd']}")
    lines.append("")

    lines.append("## Trace")
    for entry in result.get("trace", []):
        lines.append(f"1. {entry}")

    return "\n".join(lines)


def _render_json(result: CrewState) -> str:
    payload = _jsonable(dict(result))
    payload["usage_summary"] = aggregate_usage(result.get("usage_log", []))
    return json.dumps(payload, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentcrew",
        description="Run the AgentCrew multi-agent research & writing crew on a topic.",
    )
    parser.add_argument("topic", help="The research topic to investigate and write about.")
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Maximum number of writer/critic revision cycles (default: 2).",
    )
    parser.add_argument(
        "--output",
        choices=["md", "json"],
        default="md",
        help="Output format: Markdown report (default) or raw JSON state.",
    )
    parser.add_argument(
        "--thread-id",
        dest="thread_id",
        default=None,
        help="Checkpoint thread id. Required (together with --resume) to resume a run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously checkpointed run identified by --thread-id.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.resume and not args.thread_id:
        parser.error("--resume requires --thread-id")

    thread_id = args.thread_id or (str(uuid.uuid4()) if not args.resume else None)

    conn = sqlite3.connect(DEFAULT_CHECKPOINT_DB, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn, serde=default_serde())
        result = run_crew(
            topic=args.topic,
            max_revisions=args.max_revisions,
            llm=FakeLLM(),
            checkpointer=checkpointer,
            thread_id=thread_id,
        )
    finally:
        conn.close()

    if args.output == "json":
        print(_render_json(result))
    else:
        print(_render_markdown(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
