"""Token/cost usage estimation and aggregation for AgentCrew.

Every LLM call in AgentCrew (fake or real) is expected to append a usage
record produced by :func:`record_usage` to ``CrewState["usage_log"]``, so
that the full run's token and cost footprint can be summarized at the end
with :func:`aggregate_usage`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

# Rough estimated $ price per 1,000 tokens, split by input/output, for a
# handful of representative models. These are illustrative "estimated cost"
# figures for demo/portfolio purposes, not a live pricing feed.
_PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    "fake-llm": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
}

_DEFAULT_PRICING = {"input": 0.001, "output": 0.002}


def estimate_tokens(text: str) -> int:
    """Rough token count heuristic: ~4 characters per token.

    This is intentionally simple (no real tokenizer dependency) and errs on
    the side of a stable, deterministic estimate suitable for offline tests
    and demos.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _pricing_for(model: str) -> Dict[str, float]:
    return _PRICING_PER_1K.get(model, _DEFAULT_PRICING)


def record_usage(role: str, input_tokens: int, output_tokens: int, model: str) -> dict:
    """Build one usage-log record, including an estimated dollar cost.

    Returns a plain dict (not a dataclass) because it is appended directly
    into ``CrewState["usage_log"]``, which LangGraph merges via
    ``operator.add`` and which the CLI/demo serialize straight to JSON.
    """
    pricing = _pricing_for(model)
    cost = (input_tokens / 1000.0) * pricing["input"] + (output_tokens / 1000.0) * pricing["output"]
    return {
        "role": role,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost, 6),
    }


def aggregate_usage(usage_log: Iterable[dict]) -> dict:
    """Summarize a usage log into total calls/tokens/cost, broken down by role."""
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    by_role: Dict[str, dict] = {}

    for record in usage_log:
        total_calls += 1
        input_tokens = record.get("input_tokens", 0)
        output_tokens = record.get("output_tokens", 0)
        cost = record.get("estimated_cost_usd", 0.0)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += cost

        role = record.get("role", "unknown")
        role_bucket = by_role.setdefault(
            role, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}
        )
        role_bucket["calls"] += 1
        role_bucket["input_tokens"] += input_tokens
        role_bucket["output_tokens"] += output_tokens
        role_bucket["estimated_cost_usd"] = round(role_bucket["estimated_cost_usd"] + cost, 6)

    return {
        "total_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_estimated_cost_usd": round(total_cost, 6),
        "by_role": by_role,
    }
