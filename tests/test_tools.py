"""Tests for the tool system: tools, router, and registry."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.base import ToolResult
from src.tools.calculator import CalculatorTool
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter
from src.tools.web_search import WebSearchTool
from src.tools.wikipedia import WikipediaTool

# -- ToolResult Tests --


class TestToolResult:
    def test_success_when_no_error(self) -> None:
        r = ToolResult(tool_name="test", query="q", data={"k": "v"})
        assert r.success is True

    def test_failure_when_error(self) -> None:
        r = ToolResult(tool_name="test", query="q", error="failed")
        assert r.success is False

    def test_format_includes_data(self) -> None:
        r = ToolResult(tool_name="Calc", query="2+2", data={"result": "4"})
        formatted = r.format()
        assert "[Calc]" in formatted
        assert "2+2" in formatted
        assert "4" in formatted

    def test_format_shows_error(self) -> None:
        r = ToolResult(tool_name="Calc", query="q", error="boom")
        assert "Error: boom" in r.format()


# -- Calculator Tests --


class TestCalculator:
    async def test_basic_math(self) -> None:
        tool = CalculatorTool()
        r = await tool.execute("2 + 3")
        assert r.success
        assert r.data["result"] == "5.0"

    async def test_power(self) -> None:
        tool = CalculatorTool()
        r = await tool.execute("2^10")
        assert r.success
        assert r.data["result"] == "1024.0"

    async def test_percentage(self) -> None:
        tool = CalculatorTool()
        r = await tool.execute("15% of 200")
        assert r.success
        assert r.data["result"] == "30.0"

    async def test_datetime(self) -> None:
        tool = CalculatorTool()
        r = await tool.execute("what time is it")
        assert r.success
        assert "utc_time" in r.data

    async def test_division_by_zero(self) -> None:
        tool = CalculatorTool()
        r = await tool.execute("10 / 0")
        assert not r.success
        assert "zero" in r.error.lower()

    async def test_invalid_expression(self) -> None:
        tool = CalculatorTool()
        r = await tool.execute("no math here")
        assert not r.success


# -- Web Search Tests --


class TestWebSearch:
    @patch("src.tools.web_search.WebSearchTool._search_sync")
    async def test_returns_results(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {"title": "Result 1", "url": "http://a.com", "snippet": "text"},
        ]
        tool = WebSearchTool()
        r = await tool.execute("test query")
        assert r.success
        assert len(r.data["results"]) == 1

    @patch("src.tools.web_search.WebSearchTool._search_sync")
    async def test_handles_empty_results(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        tool = WebSearchTool()
        r = await tool.execute("test")
        assert not r.success
        assert "No results" in r.error

    @patch("src.tools.web_search.WebSearchTool._search_sync")
    async def test_handles_exception(self, mock_search: MagicMock) -> None:
        mock_search.side_effect = Exception("network error")
        tool = WebSearchTool()
        r = await tool.execute("test")
        assert not r.success
        assert "network error" in r.error


# -- Wikipedia Tests --


class TestWikipedia:
    @patch("src.tools.wikipedia.WikipediaTool._lookup_sync")
    async def test_returns_summary(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = {
            "title": "Python",
            "summary": "A programming language.",
            "url": "https://en.wikipedia.org/wiki/Python",
        }
        tool = WikipediaTool()
        r = await tool.execute("what is Python")
        assert r.success
        assert r.data["title"] == "Python"

    @patch("src.tools.wikipedia.WikipediaTool._lookup_sync")
    async def test_handles_not_found(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = None
        tool = WikipediaTool()
        r = await tool.execute("asdfghjkl")
        assert not r.success

    @patch("src.tools.wikipedia.WikipediaTool._lookup_sync")
    async def test_handles_exception(self, mock_lookup: MagicMock) -> None:
        mock_lookup.side_effect = Exception("API down")
        tool = WikipediaTool()
        r = await tool.execute("test")
        assert not r.success


# -- Router Tests --


class TestToolRouter:
    def _make_tools(self) -> dict[str, MagicMock]:
        tools = {}
        for name, kws in [
            ("Web Search", ["search", "latest", "news"]),
            ("Wikipedia", ["what is", "who is", "explain"]),
            ("Calculator", ["calculate"]),
        ]:
            tool = MagicMock()
            tool.name = name
            tool.keywords = kws
            tool.patterns = []
            tools[name] = tool
        return tools

    def test_keyword_match(self) -> None:
        router = ToolRouter()
        tools = self._make_tools()
        assert router.route_keyword("search for Python", tools) == ["Web Search"]
        assert router.route_keyword("what is AI", tools) == ["Wikipedia"]
        assert router.route_keyword("calculate 2+2", tools) == ["Calculator"]

    def test_no_match(self) -> None:
        router = ToolRouter()
        tools = self._make_tools()
        assert router.route_keyword("hello", tools) == []

    async def test_hybrid_uses_llm_on_no_keyword_match(self) -> None:
        router = ToolRouter(api_key="test")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"tools": ["Web Search"]}'
        router._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        tools = self._make_tools()
        result = await router.route("look up the market today", tools)
        assert "Web Search" in result

    async def test_hybrid_skips_llm_when_keyword_matches(self) -> None:
        router = ToolRouter(api_key="test")
        router._client.chat.completions.create = AsyncMock()

        tools = self._make_tools()
        result = await router.route("search for SpaceX", tools)
        assert "Web Search" in result
        # LLM should NOT have been called
        router._client.chat.completions.create.assert_not_called()


# -- Registry Tests --


class TestToolRegistry:
    async def test_no_tools_returns_empty(self) -> None:
        router = ToolRouter()
        registry = ToolRegistry(router)
        result = await registry.route_and_execute("hello")
        assert result == ""

    async def test_calculator_integration(self) -> None:
        router = ToolRouter()
        registry = ToolRegistry(router)
        registry.register(CalculatorTool())

        result = await registry.route_and_execute("calculate 10 * 5")
        assert "50.0" in result
        assert "[Calculator]" in result

    async def test_no_match_returns_empty(self) -> None:
        router = ToolRouter()
        registry = ToolRegistry(router)
        registry.register(CalculatorTool())

        result = await registry.route_and_execute("hello how are you")
        assert result == ""

    async def test_timeout_handled_gracefully(self) -> None:
        router = ToolRouter()
        registry = ToolRegistry(router)

        # Create a slow tool
        slow_tool = MagicMock()
        slow_tool.name = "Slow"
        slow_tool.keywords = ["slow"]
        slow_tool.patterns = []

        async def slow_execute(query: str) -> ToolResult:
            await asyncio.sleep(10)
            return ToolResult(tool_name="Slow", query=query, data={})

        slow_tool.execute = slow_execute
        registry.register(slow_tool)

        result = await registry.route_and_execute("slow query")
        # Should return empty (tool timed out, retry timed out)
        assert result == ""

    async def test_last_results_tracked(self) -> None:
        router = ToolRouter()
        registry = ToolRegistry(router)
        registry.register(CalculatorTool())

        await registry.route_and_execute("calculate 1 + 1")
        assert len(registry.last_results) > 0
        assert registry.last_results[0].tool_name == "Calculator"


# -- Chitchat Detection Tests --


class TestChitchatDetection:
    """Verify chitchat queries skip LLM fallback."""

    def test_greetings_are_chitchat(self) -> None:
        router = ToolRouter()
        for q in ["hello", "hi", "hey", "howdy", "greetings"]:
            assert router._is_chitchat(q), f"'{q}' should be chitchat"

    def test_pleasantries_are_chitchat(self) -> None:
        router = ToolRouter()
        for q in ["thanks", "thank you", "bye", "goodbye", "ok", "sure"]:
            assert router._is_chitchat(q), f"'{q}' should be chitchat"

    def test_questions_are_not_chitchat(self) -> None:
        router = ToolRouter()
        for q in ["what is AI", "search for news", "2+2", "explain gravity"]:
            assert not router._is_chitchat(q), f"'{q}' should NOT be chitchat"

    def test_case_and_punctuation_insensitive(self) -> None:
        router = ToolRouter()
        assert router._is_chitchat("Hello!")
        assert router._is_chitchat("THANKS?")
        assert router._is_chitchat("  how are you  ")


# -- Edge Case Tests: False Positives --


class TestFalsePositives:
    """Queries that SHOULD NOT trigger tools despite containing trigger-like words."""

    def _make_real_tools(self) -> dict:
        return {
            "Web Search": WebSearchTool(),
            "Wikipedia": WikipediaTool(),
            "Calculator": CalculatorTool(),
        }

    def test_metaphorical_divided_no_tool(self) -> None:
        """'divided by borders' is not math."""
        router = ToolRouter()
        tools = self._make_real_tools()
        matched = router.route_keyword("nations divided by colonial borders", tools)
        assert "Calculator" not in matched

    def test_historical_divided_no_calculator(self) -> None:
        """'divided in 1947' is history, not math."""
        router = ToolRouter()
        tools = self._make_real_tools()
        matched = router.route_keyword("India and Pakistan divided in 1947", tools)
        assert "Calculator" not in matched

    def test_times_as_noun_not_calculator(self) -> None:
        """'New York Times' should not trigger Calculator."""
        router = ToolRouter()
        tools = self._make_real_tools()
        matched = router.route_keyword("New York Times article about AI", tools)
        assert "Calculator" not in matched

    def test_current_as_adjective_not_search(self) -> None:
        """'current in a circuit' is physics, not breaking news."""
        router = ToolRouter()
        tools = self._make_real_tools()
        matched = router.route_keyword("how does current flow in a circuit", tools)
        # "current" is not in web search keywords anymore (we removed it)
        assert "Web Search" not in matched


# -- Edge Case Tests: Should Trigger --


class TestShouldTrigger:
    """Queries that SHOULD trigger tools — including short ones."""

    def _make_real_tools(self) -> dict:
        return {
            "Web Search": WebSearchTool(),
            "Wikipedia": WikipediaTool(),
            "Calculator": CalculatorTool(),
        }

    def test_short_search_query(self) -> None:
        """'search AI' is only 2 words but should trigger Web Search."""
        router = ToolRouter()
        matched = router.route_keyword("search AI", self._make_real_tools())
        assert "Web Search" in matched

    def test_short_math(self) -> None:
        """'2+2' is 1 token but should trigger Calculator."""
        router = ToolRouter()
        matched = router.route_keyword("2+2", self._make_real_tools())
        assert "Calculator" in matched

    def test_explicit_math_expression(self) -> None:
        router = ToolRouter()
        matched = router.route_keyword("what is 147 * 38 + 92", self._make_real_tools())
        assert "Calculator" in matched

    def test_wikipedia_who_is(self) -> None:
        router = ToolRouter()
        matched = router.route_keyword("who is Alan Turing", self._make_real_tools())
        assert "Wikipedia" in matched


# -- Chain Pre-filter Tests --


class TestChainPreFilter:
    """Test that the pre-filter correctly gates LLM chain validation."""

    def test_skips_when_calculator_already_ran(self) -> None:
        from src.tools.registry import ToolRegistry

        results = [ToolResult(tool_name="Calculator", query="2+2", data={"result": "4"})]
        assert not ToolRegistry._should_consider_chaining(results)

    def test_skips_when_no_successful_results(self) -> None:
        from src.tools.registry import ToolRegistry

        results = [ToolResult(tool_name="Wikipedia", query="q", error="failed")]
        assert not ToolRegistry._should_consider_chaining(results)

    def test_skips_when_no_numbers_in_results(self) -> None:
        from src.tools.registry import ToolRegistry

        results = [ToolResult(
            tool_name="Wikipedia", query="q",
            data={"summary": "No numbers here at all."},
        )]
        assert not ToolRegistry._should_consider_chaining(results)

    def test_passes_when_numbers_in_results(self) -> None:
        from src.tools.registry import ToolRegistry

        results = [ToolResult(
            tool_name="Wikipedia", query="q",
            data={"summary": "Japan has a population of 125000000."},
        )]
        assert ToolRegistry._should_consider_chaining(results)


# -- LLM Chain Validator Tests --


class TestChainValidator:
    """Test LLM-driven chain validation with mocked LLM."""

    async def test_chains_when_llm_approves(self) -> None:
        router = ToolRouter(api_key="test")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "chain": True,
            "expression": "125000000 / 47",
            "reasoning": "User wants population divided by prefectures",
        })
        router._client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = [ToolResult(
            tool_name="Wikipedia", query="Japan",
            data={"summary": "Japan population 125000000"},
        )]
        decision = await router.validate_chain(
            "population of Japan divided by 47", results
        )
        assert decision.should_chain is True
        assert decision.expression == "125000000 / 47"

    async def test_rejects_metaphorical_divided(self) -> None:
        router = ToolRouter(api_key="test")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "chain": False,
            "reasoning": "divided is used metaphorically, not mathematically",
        })
        router._client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = [ToolResult(
            tool_name="Wikipedia", query="India",
            data={"summary": "India was divided in 1947"},
        )]
        decision = await router.validate_chain(
            "India and Pakistan were divided in 1947", results
        )
        assert decision.should_chain is False

    async def test_fallback_on_llm_failure(self) -> None:
        router = ToolRouter(api_key="test")
        router._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        results = [ToolResult(
            tool_name="Wikipedia", query="q",
            data={"summary": "population 125000000"},
        )]
        decision = await router.validate_chain("some query", results)
        assert decision.should_chain is False

    async def test_no_chain_without_api_key(self) -> None:
        router = ToolRouter(api_key="")
        results = [ToolResult(
            tool_name="Wikipedia", query="q",
            data={"summary": "population 125000000"},
        )]
        decision = await router.validate_chain("divided by 47", results)
        assert decision.should_chain is False
