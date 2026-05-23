"""Tool registry: orchestrates routing, execution, chaining, and formatting.

The registry is the single entry point for the app. It owns all
registered tools, the router, and the execution pipeline:

    route_and_execute(query) -> str

Everything else is internal.
"""

from __future__ import annotations

import asyncio
import logging
import re

from src.tools.base import Tool, ToolResult
from src.tools.router import ToolRouter

logger = logging.getLogger(__name__)

_TOOL_TIMEOUT_SECONDS = 5
_MAX_CONTEXT_LENGTH = 1500
_RETRY_BACKOFF_SECONDS = 1

_CONTEXT_HEADER = (
    "[TOOL RESULTS - Use these to provide an accurate, grounded response]\n"
)
_CONTEXT_FOOTER = (
    "\nCite sources when relevant. "
    "If tool results don't answer the question, use your own knowledge."
)


class ToolRegistry:
    """Orchestrates tool routing, execution, chaining, and formatting."""

    def __init__(self, router: ToolRouter) -> None:
        self._tools: dict[str, Tool] = {}
        self._router = router
        self._last_results: list[ToolResult] = []

    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    async def route_and_execute(self, query: str) -> str:
        """Main entry point: route query → execute tools → format context.

        Returns formatted context string, or empty string if no tools matched.
        """
        if not self._tools:
            return ""

        # 1. Route
        matched = await self._router.route(query, self._tools)
        if not matched:
            return ""

        logger.info("Tools matched for query: %s", matched)

        # 2. Execute matched tools concurrently with timeout + retry
        results = await self._execute_all(query, matched)

        # 3. Chain: if wiki returned data and query has math intent, run calculator
        results = await self._chain(query, results)

        # Store for observability
        self._last_results = list(results)

        # 4. Filter successful results
        successful = [r for r in results if r.success]
        if not successful:
            logger.warning("All tools failed for query: %s", query)
            return ""

        # 5. Format context string
        return self._format_context(successful)

    async def _execute_all(
        self, query: str, tool_names: list[str]
    ) -> list[ToolResult]:
        """Execute multiple tools concurrently with timeout and retry."""
        tasks = [
            self._execute_with_retry(query, name)
            for name in tool_names
            if name in self._tools
        ]
        return list(await asyncio.gather(*tasks))

    async def _execute_with_retry(
        self, query: str, tool_name: str
    ) -> ToolResult:
        """Execute a single tool with timeout and one retry on failure."""
        result = await self._execute_one(query, tool_name)

        if not result.success:
            logger.info("Retrying tool %s after failure", tool_name)
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
            result = await self._execute_one(query, tool_name)

        return result

    async def _execute_one(self, query: str, tool_name: str) -> ToolResult:
        """Execute a single tool with timeout."""
        tool = self._tools[tool_name]

        try:
            result = await asyncio.wait_for(
                tool.execute(query),
                timeout=_TOOL_TIMEOUT_SECONDS,
            )
            return result
        except TimeoutError:
            logger.warning("Tool %s timed out after %ds", tool_name, _TOOL_TIMEOUT_SECONDS)
            return ToolResult(
                tool_name=tool_name,
                query=query,
                error=f"Timed out after {_TOOL_TIMEOUT_SECONDS}s",
                latency_ms=_TOOL_TIMEOUT_SECONDS * 1000,
            )
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return ToolResult(
                tool_name=tool_name,
                query=query,
                error=str(e),
            )

    async def _chain(
        self, query: str, results: list[ToolResult]
    ) -> list[ToolResult]:
        """Targeted chaining: wikipedia result + math intent → calculator.

        If Wikipedia returned a number and the query has math intent
        (division, multiplication, etc.), run calculator with the extracted data.
        """
        if "Calculator" in [r.tool_name for r in results]:
            # Calculator already ran, no chaining needed
            return results

        # Check if any result contains a number and query has math intent
        math_intent = bool(re.search(
            r"(?:divided|multiplied|times|plus|minus|per|ratio|average)",
            query,
            re.IGNORECASE,
        ))
        if not math_intent:
            return results

        # Look for numbers in successful results
        for result in results:
            if not result.success:
                continue
            # Extract numbers from result data
            numbers = _extract_numbers_from_result(result)
            if numbers and "Calculator" in self._tools:
                # Build a math expression from the query context
                calc_query = f"{query} (numbers from {result.tool_name}: {numbers})"
                calc_result = await self._execute_one(calc_query, "Calculator")
                if calc_result.success:
                    results = [*results, calc_result]
                    logger.info("Chained Calculator after %s", result.tool_name)
                break

        return results

    @staticmethod
    def _format_context(results: list[ToolResult]) -> str:
        """Format tool results into a context string for the model."""
        parts = [_CONTEXT_HEADER]

        for result in results:
            parts.append(result.format())
            parts.append("")  # blank line between tools

        parts.append(_CONTEXT_FOOTER)

        context = "\n".join(parts)

        # Cap total length
        if len(context) > _MAX_CONTEXT_LENGTH:
            context = context[:_MAX_CONTEXT_LENGTH] + "\n... (truncated)"

        return context


def _extract_numbers_from_result(result: ToolResult) -> list[float]:
    """Extract numeric values from a tool result's data."""
    numbers: list[float] = []
    text = str(result.data)
    for match in re.finditer(r"\b(\d{1,15}(?:\.\d+)?)\b", text):
        try:
            num = float(match.group(1))
            if num > 1:  # Skip trivial numbers like 0, 1
                numbers.append(num)
        except ValueError:
            continue
    return numbers[:3]  # Cap at 3 numbers
