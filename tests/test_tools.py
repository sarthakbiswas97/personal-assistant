"""Tests for the tool system: tools, router, and registry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        assert len(registry._last_results) > 0
        assert registry._last_results[0].tool_name == "Calculator"
