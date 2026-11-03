"""Tests for agentcrew.tools: WebSearchTool ranking and CalculatorTool safety."""

from __future__ import annotations

import pytest

from agentcrew.tools import CalculatorTool, SearchHit, WebSearchTool


class TestWebSearchTool:
    def test_returns_search_hits(self):
        tool = WebSearchTool()
        hits = tool.run("computing and technology systems", top_k=2)
        assert len(hits) == 2
        assert all(isinstance(hit, SearchHit) for hit in hits)

    def test_returns_fewer_hits_than_top_k_when_only_one_article_overlaps(self):
        # Only one knowledge-base article actually overlaps on these keywords,
        # so the tool should not pad the result with unrelated articles.
        tool = WebSearchTool()
        hits = tool.run("quantum computing qubits", top_k=3)
        assert len(hits) == 1
        assert hits[0].source_id == "kb-002"

    def test_ranks_most_relevant_article_first(self):
        tool = WebSearchTool()
        hits = tool.run("CRISPR gene editing Cas9", top_k=3)
        assert hits[0].source_id == "kb-006"
        assert "CRISPR" in hits[0].title

    def test_different_queries_return_different_top_results(self):
        tool = WebSearchTool()
        solar_hits = tool.run("solar and wind renewable energy", top_k=1)
        immune_hits = tool.run("immune system T cells vaccination", top_k=1)
        assert solar_hits[0].source_id != immune_hits[0].source_id
        assert solar_hits[0].source_id == "kb-003"
        assert immune_hits[0].source_id == "kb-009"

    def test_never_returns_empty_even_for_unrelated_query(self):
        tool = WebSearchTool()
        hits = tool.run("xyzzy plugh completely unrelated nonsense query", top_k=2)
        assert len(hits) == 2

    def test_custom_knowledge_base_is_used(self):
        custom_kb = [
            {"source_id": "custom-1", "title": "Widgets", "text": "Widgets are small mechanical parts."}
        ]
        tool = WebSearchTool(knowledge_base=custom_kb)
        hits = tool.run("widgets", top_k=1)
        assert hits[0].source_id == "custom-1"


class TestCalculatorTool:
    def test_basic_arithmetic(self):
        calc = CalculatorTool()
        assert calc.evaluate("2 + 2") == 4
        assert calc.evaluate("10 / 4") == 2.5
        assert calc.evaluate("2 ** 10") == 1024
        assert calc.evaluate("(3 + 4) * 2 - 1") == 13

    def test_unary_operators(self):
        calc = CalculatorTool()
        assert calc.evaluate("-5 + 3") == -2
        assert calc.evaluate("+5") == 5

    def test_floor_div_and_mod(self):
        calc = CalculatorTool()
        assert calc.evaluate("17 // 5") == 3
        assert calc.evaluate("17 % 5") == 2

    def test_division_by_zero_raises_calculator_error(self):
        calc = CalculatorTool()
        with pytest.raises(CalculatorTool.CalculatorError):
            calc.evaluate("1 / 0")

    def test_invalid_syntax_raises_calculator_error(self):
        calc = CalculatorTool()
        with pytest.raises(CalculatorTool.CalculatorError):
            calc.evaluate("2 +")

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "__import__('os')",
            "eval('1+1')",
            "exec('print(1)')",
            "[1, 2, 3]",
            "{'a': 1}",
            "(lambda: 1)()",
            "1 if True else 2",
            "1 == 1",
            "os.system('ls')",
            "'a string'",
            "[x for x in range(10)]",
        ],
    )
    def test_unsafe_or_disallowed_expressions_are_rejected(self, expression):
        calc = CalculatorTool()
        with pytest.raises(CalculatorTool.CalculatorError):
            calc.evaluate(expression)

    def test_run_delegates_to_evaluate(self):
        calc = CalculatorTool()
        assert calc.run("3 * 3") == 9
