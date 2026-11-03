"""Tests for agentcrew.usage: token estimation and usage aggregation."""

from __future__ import annotations

from agentcrew.usage import aggregate_usage, estimate_tokens, record_usage


class TestEstimateTokens:
    def test_empty_string_is_zero(self):
        assert estimate_tokens("") == 0

    def test_roughly_chars_over_four(self):
        text = "a" * 40
        assert estimate_tokens(text) == 10

    def test_short_nonempty_text_is_at_least_one(self):
        assert estimate_tokens("hi") == 1


class TestRecordUsage:
    def test_record_shape_and_fields(self):
        record = record_usage(role="writer", input_tokens=100, output_tokens=50, model="fake-llm")
        assert record["role"] == "writer"
        assert record["model"] == "fake-llm"
        assert record["input_tokens"] == 100
        assert record["output_tokens"] == 50
        assert record["total_tokens"] == 150
        assert "estimated_cost_usd" in record

    def test_fake_llm_is_free(self):
        record = record_usage(role="planner", input_tokens=1000, output_tokens=1000, model="fake-llm")
        assert record["estimated_cost_usd"] == 0.0

    def test_known_model_has_positive_cost(self):
        record = record_usage(role="writer", input_tokens=1000, output_tokens=1000, model="gpt-4o")
        assert record["estimated_cost_usd"] > 0

    def test_unknown_model_falls_back_to_default_pricing(self):
        record = record_usage(role="writer", input_tokens=1000, output_tokens=1000, model="some-unknown-model")
        assert record["estimated_cost_usd"] > 0


class TestAggregateUsage:
    def test_empty_log(self):
        summary = aggregate_usage([])
        assert summary["total_calls"] == 0
        assert summary["total_tokens"] == 0
        assert summary["total_estimated_cost_usd"] == 0.0
        assert summary["by_role"] == {}

    def test_sums_calls_and_tokens_across_records(self):
        log = [
            record_usage(role="planner", input_tokens=10, output_tokens=20, model="fake-llm"),
            record_usage(role="researcher", input_tokens=30, output_tokens=40, model="fake-llm"),
            record_usage(role="researcher", input_tokens=5, output_tokens=5, model="fake-llm"),
        ]
        summary = aggregate_usage(log)
        assert summary["total_calls"] == 3
        assert summary["total_input_tokens"] == 45
        assert summary["total_output_tokens"] == 65
        assert summary["total_tokens"] == 110
        assert summary["by_role"]["researcher"]["calls"] == 2
        assert summary["by_role"]["planner"]["calls"] == 1

    def test_cost_aggregation_across_models(self):
        log = [
            record_usage(role="writer", input_tokens=1000, output_tokens=1000, model="gpt-4o-mini"),
            record_usage(role="writer", input_tokens=1000, output_tokens=1000, model="claude-3-5-sonnet"),
        ]
        summary = aggregate_usage(log)
        expected = round(
            (1000 / 1000.0) * 0.00015 + (1000 / 1000.0) * 0.0006
            + (1000 / 1000.0) * 0.003 + (1000 / 1000.0) * 0.015,
            6,
        )
        assert summary["total_estimated_cost_usd"] == expected
