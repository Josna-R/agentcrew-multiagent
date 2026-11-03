"""Tests for the agentcrew.cli command-line interface."""

from __future__ import annotations

import json
import os

import pytest

from agentcrew import cli
from agentcrew.state import FactCheckResult


@pytest.fixture
def isolated_checkpoint_db(tmp_path, monkeypatch):
    """Point the CLI's sqlite checkpoint store at a throwaway file per test."""
    db_path = str(tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(cli, "DEFAULT_CHECKPOINT_DB", db_path)
    yield db_path


def test_main_runs_markdown_report(capsys, isolated_checkpoint_db):
    exit_code = cli.main(["Ocean currents", "--max-revisions", "1"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "# AgentCrew Report: Ocean currents" in output
    assert "## Subtopics Investigated" in output
    assert "## Fact-Check" in output
    assert "## Usage" in output


def test_main_runs_json_report(capsys, isolated_checkpoint_db):
    exit_code = cli.main(["Blockchain basics", "--max-revisions", "1", "--output", "json"])
    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["topic"] == "Blockchain basics"
    assert isinstance(payload["subtopics"], list)
    assert len(payload["subtopics"]) == 3
    assert "usage_summary" in payload


def test_resume_without_thread_id_is_an_error(isolated_checkpoint_db):
    with pytest.raises(SystemExit):
        cli.main(["Some topic", "--resume"])


def test_resume_continues_a_checkpointed_run(capsys, isolated_checkpoint_db):
    exit_code = cli.main(["CRISPR gene editing", "--thread-id", "cli-thread-1", "--max-revisions", "1"])
    assert exit_code == 0
    capsys.readouterr()  # discard first run's output

    exit_code = cli.main(
        ["CRISPR gene editing", "--resume", "--thread-id", "cli-thread-1", "--output", "json"]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["topic"] == "CRISPR gene editing"


def test_render_markdown_includes_flagged_sentences():
    result = {
        "topic": "Test topic",
        "subtopics": ["A", "B", "C"],
        "draft": "# Report\nSome content.",
        "fact_check": FactCheckResult(groundedness_score=0.5, flagged_sentences=["A dubious claim."], summary="1/2 grounded"),
        "approved": True,
        "revision_count": 1,
        "max_revisions": 2,
        "trace": ["[planner] decomposed 'Test topic' into 3 subtopics"],
        "usage_log": [],
    }
    markdown = cli._render_markdown(result)
    assert "A dubious claim." in markdown
    assert "Groundedness score: 0.5" in markdown


def test_arg_parser_defaults():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["My topic"])
    assert args.topic == "My topic"
    assert args.max_revisions == 2
    assert args.output == "md"
    assert args.resume is False
