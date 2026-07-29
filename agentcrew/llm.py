"""
Pluggable LLM used by every agent node.

Defaults to `FakeLLM`, a deterministic, template-driven generator, so the
whole multi-agent graph runs end-to-end offline with zero API cost -- handy
for demos, CI, and anyone cloning the repo without wanting to provision a
key just to see how the orchestration works. Point `AGENTCREW_LLM=openai`
(with OPENAI_API_KEY set) or `AGENTCREW_LLM=anthropic` at a real model for
production-quality output; every node calls the same `LLM.complete()`
interface so no orchestration code changes.
"""
from __future__ import annotations

import abc
import os
import textwrap


class LLM(abc.ABC):
    @abc.abstractmethod
    def complete(self, role: str, prompt: str) -> str:
        ...


class FakeLLM(LLM):
    """Deterministic per-role templates so the graph is fully testable offline."""

    def complete(self, role: str, prompt: str) -> str:
        topic = prompt.split("Topic:", 1)[-1].strip().splitlines()[0] if "Topic:" in prompt else prompt[:60]

        if role == "researcher":
            return textwrap.dedent(f"""\
                Key findings on "{topic}":
                1. Adoption of this area has grown significantly in production ML systems over the last two years.
                2. The main technical challenges are latency, cost control, and evaluation/observability.
                3. Leading approaches combine retrieval, orchestration frameworks, and human-in-the-loop review.
                4. Open questions remain around reliability at scale and standardized evaluation metrics.""")

        if role == "writer":
            return textwrap.dedent(f"""\
                # Report: {topic}

                ## Overview
                {topic} has become a core capability in modern AI systems, driven by
                the need for grounded, up-to-date, and explainable outputs.

                ## Key Findings
                {prompt.split('Research notes:', 1)[-1].strip()[:400]}

                ## Recommendation
                Teams adopting this should invest early in evaluation tooling and
                keep a human-in-the-loop review step for high-stakes outputs.""")

        if role == "critic":
            has_recommendation = "Recommendation" in prompt
            has_findings = "Key Findings" in prompt
            if has_recommendation and has_findings:
                return "APPROVE: The report has clear findings and an actionable recommendation. No revision needed."
            return "REVISE: Please add a concrete Recommendation section and ensure Key Findings are explicit."

        return f"[{role}] " + prompt[:200]


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4.1-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install openai to use OpenAILLM") from exc
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def complete(self, role: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": f"You are the {role} agent in a multi-agent research crew."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


class AnthropicLLM(LLM):
    def __init__(self, model: str = "claude-sonnet-5"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install anthropic to use AnthropicLLM") from exc
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def complete(self, role: str, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=f"You are the {role} agent in a multi-agent research crew.",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


def get_llm(kind: str | None = None) -> LLM:
    kind = kind or os.environ.get("AGENTCREW_LLM", "fake")
    if kind == "fake":
        return FakeLLM()
    if kind == "openai":
        return OpenAILLM()
    if kind == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"Unknown LLM kind: {kind}")
