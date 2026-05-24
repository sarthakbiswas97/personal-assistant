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
        """LLM-driven chaining: ask the LLM if tool results need computation.

        Pre-filter gates the LLM call to keep costs low. The LLM sees
        both the query and tool results, giving it semantic context to
        distinguish "divided by 47" (math) from "divided by borders" (metaphor).
        """
        if not self._should_consider_chaining(results):
            return results

        # Ask the LLM whether chaining is appropriate
        decision = await self._router.validate_chain(query, results)

        if not decision.should_chain or not decision.expression:
            logger.info("Chain validator: no chaining needed (%s)", decision.reasoning)
            return results

        # Execute Calculator with the LLM-provided expression
        if "Calculator" not in self._tools:
            return results

        calc_result = await self._execute_one(decision.expression, "Calculator")
        if calc_result.success:
            logger.info(
                "Chained Calculator: %s = %s",
                decision.expression,
                calc_result.data.get("result", "?"),
            )
            return [*results, calc_result]

        logger.warning("Chained Calculator failed on expression: %s", decision.expression)
        return results

    @staticmethod
    def _should_consider_chaining(results: list[ToolResult]) -> bool:
        """Cheap pre-filter: is chaining even worth considering?

        Only passes when:
        - Calculator didn't already run (no point chaining to itself)
        - At least one tool returned successful results
        - Results contain numeric data (nothing to compute otherwise)
        """
        tool_names = [r.tool_name for r in results]

        # Calculator already ran
        if "Calculator" in tool_names:
            return False

        # No successful results
        successful = [r for r in results if r.success]
        if not successful:
            return False

        # No numbers in any result
        has_numbers = any(
            _result_has_numbers(r) for r in successful
        )
        return has_numbers

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


def _result_has_numbers(result: ToolResult) -> bool:
    """Check if a tool result contains any numeric values."""
    text = str(result.data)
    return bool(re.search(r"\b\d{2,}\b", text))
