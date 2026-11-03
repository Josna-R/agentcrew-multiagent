"""LLM backends used by AgentCrew agents.

``FakeLLM`` is the default, fully offline backend: it produces deterministic,
role-specific text so the whole graph can be built, tested, and demoed with
zero API keys and zero network access. ``OpenAILLM`` and ``AnthropicLLM`` are
thin real-API wrappers for production use; both are optional (their SDKs are
imported lazily) and both wrap their network call with
``@retry_with_backoff`` from :mod:`agentcrew.resilience`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from agentcrew.resilience import retry_with_backoff
from agentcrew.tools import SearchHit, WebSearchTool


class LLM(ABC):
    """Abstract base class for a text-generation backend."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """A short identifier for this model, used in usage-log records."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, role: Optional[str] = None) -> str:
        """Generate a completion for ``prompt``, optionally tailored to ``role``."""
        raise NotImplementedError


class FakeLLM(LLM):
    """A deterministic, offline stand-in for a real LLM.

    Behavior is dispatched on ``role`` so each agent node gets realistic,
    distinct output without any network access:

    - ``"planner"``: decomposes a topic into exactly 3 subtopics.
    - ``"researcher"``: writes findings for one subtopic using
      :class:`WebSearchTool` results.
    - ``"writer"``: drafts a structured report from research notes (and
      incorporates critique feedback on revisions).
    - ``"critic"``: returns ``APPROVE`` or ``REVISE`` with reasons.
    - ``"fact_checker"``: scores groundedness of a draft against research
      notes and flags unsupported sentences.
    """

    def __init__(self, search_tool: Optional[WebSearchTool] = None) -> None:
        self._search_tool = search_tool or WebSearchTool()

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def generate(self, prompt: str, role: Optional[str] = None) -> str:
        handler = {
            "planner": self._planner,
            "researcher": self._researcher,
            "writer": self._writer,
            "critic": self._critic,
            "fact_checker": self._fact_checker,
        }.get(role, self._default)
        return handler(prompt)

    # -- role handlers ----------------------------------------------------

    def _default(self, prompt: str) -> str:
        return f"[fake-llm] response to: {prompt[:120]}"

    def _planner(self, prompt: str) -> str:
        topic = prompt.strip() or "the topic"
        subtopics = [
            f"Background and definition of {topic}",
            f"Current state and key developments in {topic}",
            f"Challenges and future outlook for {topic}",
        ]
        return "\n".join(subtopics)

    def _researcher(self, prompt: str) -> str:
        # `prompt` is expected to be the subtopic itself.
        subtopic = prompt.strip()
        hits: List[SearchHit] = self._search_tool.run(subtopic, top_k=2)
        if not hits:
            return f"No sources were found for '{subtopic}'."
        lines = [f"Research findings on '{subtopic}':"]
        for hit in hits:
            lines.append(f"- According to '{hit.title}' ({hit.source_id}): {hit.snippet}")
        return "\n".join(lines)

    def _writer(self, prompt: str) -> str:
        # On a revision pass, `prompt` is prefixed with the critique text
        # ("Revise the report using this critique:\n...") before the actual
        # research notes. The summary must always be built from the research
        # notes themselves, never from that critique preamble -- otherwise a
        # revised draft's "Summary" ends up echoing the critic's instructions
        # ("Revise the report using this critique: REVISE Reasons: ...")
        # instead of the source material.
        marker = "Research notes:\n"
        idx = prompt.find(marker)
        notes = prompt[idx + len(marker):] if idx != -1 else prompt
        return (
            "# Report\n\n"
            "## Summary\n"
            f"{notes.strip()[:400]}\n\n"
            "## Conclusion\n"
            "This report synthesizes the research findings above into a "
            "coherent overview of the topic."
        )

    def _critic(self, prompt: str) -> str:
        # Deterministic rule used by tests/demo, driven by the real
        # `revision_count` that `make_critic` embeds in the prompt: the very
        # first pass (revision_count == 0) always comes back REVISE, and
        # every subsequent pass APPROVEs, as long as the draft is at least
        # minimally structured (has a heading). This exercises a genuine
        # "revise once, then approve" loop through real graph state, rather
        # than hardcoding the outcome.
        match = re.search(r"REVISION_COUNT:\s*(\d+)", prompt)
        revision_count = int(match.group(1)) if match else 0
        has_structure = "#" in prompt
        if revision_count == 0 or not has_structure:
            return "REVISE\nReasons: The draft is missing a clear conclusion or needs another pass."
        return "APPROVE\nReasons: The draft is well-structured and covers the research notes."

    def _fact_checker(self, prompt: str) -> str:
        # `prompt` contains both the draft and the research notes, joined by
        # make_fact_checker. We do a crude but deterministic groundedness
        # check: score = fraction of draft sentences that share at least one
        # non-trivial keyword with the research notes.
        return prompt  # actual scoring logic lives in agents.make_fact_checker


class OpenAILLM(LLM):
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> None:
        self._model = model
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model

    @retry_with_backoff(exceptions=(Exception,), max_attempts=3)
    def generate(self, prompt: str, role: Optional[str] = None) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise RuntimeError(
                "The 'openai' package is required to use OpenAILLM. Install it with "
                "`pip install openai`."
            ) from exc

        client = OpenAI(api_key=self._api_key)
        system = f"You are the {role} agent in a multi-agent research crew." if role else None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(model=self._model, messages=messages)
        return response.choices[0].message.content or ""


class AnthropicLLM(LLM):
    """Thin wrapper around the Anthropic messages API."""

    def __init__(self, model: str = "claude-3-5-sonnet", api_key: Optional[str] = None) -> None:
        self._model = model
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model

    @retry_with_backoff(exceptions=(Exception,), max_attempts=3)
    def generate(self, prompt: str, role: Optional[str] = None) -> str:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise RuntimeError(
                "The 'anthropic' package is required to use AnthropicLLM. Install it with "
                "`pip install anthropic`."
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        system = f"You are the {role} agent in a multi-agent research crew." if role else ""
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))
