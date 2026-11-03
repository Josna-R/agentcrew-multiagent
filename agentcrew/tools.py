"""Tools available to AgentCrew agents.

``WebSearchTool`` is deliberately *offline*: it is backed by a small embedded
knowledge base of short reference articles so that research queries return
real, deterministic content without any network access. ``CalculatorTool``
is a safe arithmetic evaluator built on Python's ``ast`` module -- it parses
an expression into an AST and only ever evaluates numeric/operator nodes,
rejecting names, calls, attribute access, subscripts, comprehensions, and any
other node type that could be used to escape the sandbox (e.g.
``__import__('os')`` or ``open('/etc/passwd')``).
"""

from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


class Tool(ABC):
    """Base class for all tools an agent can call."""

    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool and return its result."""
        raise NotImplementedError


@dataclass
class SearchHit:
    """A single result returned by :class:`WebSearchTool`."""

    title: str
    snippet: str
    source_id: str


# ---------------------------------------------------------------------------
# WebSearchTool: offline knowledge base + keyword-overlap ranking
# ---------------------------------------------------------------------------

# A small, varied, embedded "knowledge base" standing in for the open web.
# Each article is short and self-contained so FakeLLM can quote/summarize it
# deterministically in tests and in the demo.
_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "source_id": "kb-001",
        "title": "Photosynthesis: How Plants Convert Light to Energy",
        "text": (
            "Photosynthesis is the process by which green plants and some "
            "other organisms use sunlight to synthesize nutrients from "
            "carbon dioxide and water. Chlorophyll in chloroplasts absorbs "
            "light energy, driving reactions that produce glucose and "
            "release oxygen as a byproduct. This process underpins nearly "
            "all food chains on Earth and regulates atmospheric oxygen and "
            "carbon dioxide levels."
        ),
    },
    {
        "source_id": "kb-002",
        "title": "The Basics of Quantum Computing",
        "text": (
            "Quantum computing uses quantum bits, or qubits, which can "
            "represent both 0 and 1 simultaneously through superposition. "
            "Entanglement allows qubits to be correlated in ways classical "
            "bits cannot, enabling certain algorithms, such as Shor's "
            "algorithm for factoring and Grover's algorithm for search, to "
            "outperform classical computers on specific problems. Building "
            "stable, error-corrected qubits remains a central engineering "
            "challenge."
        ),
    },
    {
        "source_id": "kb-003",
        "title": "Renewable Energy: Solar and Wind Power Trends",
        "text": (
            "Solar photovoltaic and wind power have become the fastest "
            "growing sources of electricity generation worldwide, driven by "
            "falling manufacturing costs and supportive policy. Energy "
            "storage, particularly lithium-ion and emerging flow batteries, "
            "is increasingly paired with renewables to smooth out "
            "intermittent generation and stabilize electrical grids."
        ),
    },
    {
        "source_id": "kb-004",
        "title": "An Introduction to Neural Networks",
        "text": (
            "Artificial neural networks are computational models inspired "
            "loosely by biological neurons. Layers of weighted connections "
            "transform input data through nonlinear activation functions, "
            "and training via backpropagation adjusts weights to minimize a "
            "loss function. Deep networks with many layers have driven "
            "major advances in computer vision, natural language "
            "processing, and speech recognition."
        ),
    },
    {
        "source_id": "kb-005",
        "title": "The History of the Roman Aqueducts",
        "text": (
            "Roman aqueducts were engineering marvels that transported fresh "
            "water from distant sources into cities and towns using gravity "
            "and carefully calculated gradients. Built from stone, brick, "
            "and concrete, some aqueducts spanned over 50 miles, including "
            "elevated arched sections. They supplied public baths, "
            "fountains, and private households, supporting the sanitation "
            "and growth of Roman urban centers."
        ),
    },
    {
        "source_id": "kb-006",
        "title": "CRISPR Gene Editing Explained",
        "text": (
            "CRISPR-Cas9 is a gene-editing technology derived from a "
            "bacterial immune defense mechanism. A guide RNA directs the "
            "Cas9 enzyme to a specific DNA sequence, where it creates a "
            "precise cut that cells repair, allowing researchers to disable, "
            "correct, or insert genes. CRISPR has accelerated research into "
            "genetic diseases, agriculture, and potential therapies for "
            "inherited disorders."
        ),
    },
    {
        "source_id": "kb-007",
        "title": "Ocean Currents and Global Climate",
        "text": (
            "Ocean currents redistribute heat around the planet, with "
            "systems like the Gulf Stream carrying warm water from the "
            "tropics toward higher latitudes. The global thermohaline "
            "circulation, sometimes called the ocean conveyor belt, links "
            "surface and deep-water currents driven by temperature and "
            "salinity differences, and disruptions to this circulation are "
            "linked to significant regional climate shifts."
        ),
    },
    {
        "source_id": "kb-008",
        "title": "Blockchain and Distributed Ledgers",
        "text": (
            "A blockchain is a distributed ledger maintained across a "
            "network of nodes, where transactions are grouped into blocks "
            "and linked cryptographically to prior blocks, making the "
            "history tamper-evident. Consensus mechanisms such as proof of "
            "work or proof of stake allow decentralized networks to agree "
            "on a single shared state without a central authority."
        ),
    },
    {
        "source_id": "kb-009",
        "title": "The Human Immune System: Innate and Adaptive Defenses",
        "text": (
            "The human immune system combines innate defenses, such as "
            "skin barriers and inflammatory responses, with an adaptive "
            "system that learns to recognize specific pathogens via B cells "
            "and T cells. Memory cells formed after an infection or "
            "vaccination allow a faster, stronger response if the same "
            "pathogen is encountered again, which is the biological basis "
            "for vaccination."
        ),
    },
    {
        "source_id": "kb-010",
        "title": "Space Exploration: From Low Earth Orbit to Mars",
        "text": (
            "Modern space exploration spans low Earth orbit stations, "
            "reusable launch vehicles that have sharply reduced launch "
            "costs, robotic probes studying the outer solar system, and "
            "plans for crewed missions to Mars. Challenges include "
            "radiation exposure, life-support reliability, and the "
            "logistics of sustaining human life far from Earth's resources."
        ),
    },
]


class WebSearchTool(Tool):
    """An offline search tool over a small embedded knowledge base.

    Ranking is a simple keyword-overlap score: each article is scored by how
    many of the query's lowercased word tokens appear in its title and text,
    and the top ``top_k`` articles are returned as :class:`SearchHit`.
    """

    name = "web_search"
    description = "Search a small offline knowledge base of reference articles."

    # Common filler words excluded so ranking is driven by meaningful content
    # words rather than ties on "the", "and", "of", etc.
    _STOPWORDS = {
        "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are",
        "was", "were", "for", "with", "as", "by", "this", "that", "it",
        "its", "at", "be", "from", "into", "what", "how", "why", "about",
    }

    def __init__(self, knowledge_base: List[Dict[str, str]] | None = None) -> None:
        self.knowledge_base = knowledge_base if knowledge_base is not None else _KNOWLEDGE_BASE

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return [
            tok.strip(".,!?;:()'\"").lower()
            for tok in text.split()
            if tok.strip(".,!?;:()'\"") and tok.strip(".,!?;:()'\"").lower() not in cls._STOPWORDS
        ]

    def _score(self, query_tokens: List[str], article: Dict[str, str]) -> int:
        haystack_tokens = set(self._tokenize(article["title"] + " " + article["text"]))
        return sum(1 for tok in query_tokens if tok in haystack_tokens)

    def run(self, query: str, top_k: int = 3) -> List[SearchHit]:
        query_tokens = self._tokenize(query)
        scored = [
            (self._score(query_tokens, article), article) for article in self.knowledge_base
        ]
        # Keep only articles with at least one overlapping token when possible;
        # fall back to the whole base (still ranked) if nothing overlaps at all,
        # so a query never returns nothing.
        positive = [pair for pair in scored if pair[0] > 0]
        pool = positive if positive else scored
        pool.sort(key=lambda pair: pair[0], reverse=True)
        hits: List[SearchHit] = []
        for score, article in pool[:top_k]:
            snippet = article["text"][:220].rsplit(" ", 1)[0] + "..."
            hits.append(
                SearchHit(
                    title=article["title"],
                    snippet=snippet,
                    source_id=article["source_id"],
                )
            )
        return hits


# ---------------------------------------------------------------------------
# CalculatorTool: safe arithmetic evaluation via ast
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(Tool):
    """A safe arithmetic evaluator.

    Only numeric literals, unary +/-, and the binary operators +, -, *, /,
    //, %, ** are permitted. The expression is parsed with ``ast.parse`` in
    ``"eval"`` mode and walked node by node; any node that is not one of the
    allowed numeric/operator node types (names, calls, attributes,
    subscripts, comprehensions, imports, string/byte literals, etc.) raises
    :class:`CalculatorTool.CalculatorError` instead of being evaluated. This
    makes attacks like ``__import__('os').system('rm -rf /')`` or
    ``open('/etc/passwd').read()`` impossible: ``ast.Name`` and ``ast.Call``
    nodes are rejected outright, so no function of any kind -- builtin or
    otherwise -- can ever be invoked.
    """

    name = "calculator"
    description = "Evaluate a safe arithmetic expression."

    class CalculatorError(Exception):
        """Raised when an expression contains disallowed syntax."""

    def run(self, expression: str) -> float:
        return self.evaluate(expression)

    def evaluate(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise CalculatorTool.CalculatorError(f"invalid syntax: {exc}") from exc
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise CalculatorTool.CalculatorError(
                    f"disallowed constant type: {type(node.value).__name__}"
                )
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise CalculatorTool.CalculatorError(f"disallowed operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            try:
                return _ALLOWED_BINOPS[op_type](left, right)
            except ZeroDivisionError as exc:
                raise CalculatorTool.CalculatorError("division by zero") from exc
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise CalculatorTool.CalculatorError(f"disallowed unary operator: {op_type.__name__}")
            return _ALLOWED_UNARYOPS[op_type](self._eval_node(node.operand))
        # Anything else -- Name, Call, Attribute, Subscript, List, Dict, Set,
        # comprehensions, Lambda, IfExp, BoolOp, Compare, Str/Bytes (pre-3.8
        # AST nodes), f-strings, imports, etc. -- is explicitly rejected.
        raise CalculatorTool.CalculatorError(f"disallowed expression: {type(node).__name__}")
