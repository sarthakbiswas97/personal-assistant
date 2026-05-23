"""DuckDuckGo web search tool.

Provides real-time information retrieval for queries about
current events, news, or topics beyond the model's training data.
Uses the free DuckDuckGo API — no API key required.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from src.tools.base import ToolResult

logger = logging.getLogger(__name__)

_KEYWORDS = [
    "search", "find", "look up", "lookup", "google",
    "latest", "news", "current", "recent", "today",
    "what happened", "trending",
]

_PATTERNS = [
    re.compile(r"search\s+(for\s+)?", re.IGNORECASE),
    re.compile(r"(latest|recent|current)\s+\w+", re.IGNORECASE),
]

_MAX_RESULTS = 3


class WebSearchTool:
    """DuckDuckGo web search for real-time information."""

    @property
    def name(self) -> str:
        return "Web Search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current events, news, or real-time information. "
            "Use when the query asks about recent events, live data, or topics "
            "that may have changed after the model's training cutoff."
        )

    @property
    def keywords(self) -> list[str]:
        return _KEYWORDS

    @property
    def patterns(self) -> list[re.Pattern[str]]:
        return _PATTERNS

    async def execute(self, query: str) -> ToolResult:
        """Search DuckDuckGo and return top results."""
        start = time.perf_counter()

        try:
            results = await asyncio.get_running_loop().run_in_executor(
                None, self._search_sync, query
            )
            latency = (time.perf_counter() - start) * 1000

            if not results:
                return ToolResult(
                    tool_name=self.name,
                    query=query,
                    error="No results found",
                    latency_ms=latency,
                )

            return ToolResult(
                tool_name=self.name,
                query=query,
                data={"results": results},
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.warning("Web search failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                query=query,
                error=str(e),
                latency_ms=latency,
            )

    @staticmethod
    def _search_sync(query: str) -> list[dict[str, str]]:
        """Synchronous DuckDuckGo search (run in executor)."""
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=_MAX_RESULTS))

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
