"""Tool protocol and result types.

Every tool implements the Tool protocol — a name, description,
routing hints (keywords + patterns), and an async execute method
that returns a structured ToolResult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolResult:
    """Immutable result of a tool execution."""

    tool_name: str
    query: str
    data: dict = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return not self.error

    def format(self) -> str:
        """Format result for context injection."""
        if self.error:
            return f"[{self.tool_name}] Error: {self.error}"

        lines = [f"[{self.tool_name}] Query: \"{self.query}\""]
        for key, value in self.data.items():
            if isinstance(value, list):
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        parts = " | ".join(f"{k}: {v}" for k, v in item.items())
                        lines.append(f"  {i}. {parts}")
                    else:
                        lines.append(f"  {i}. {item}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@runtime_checkable
class Tool(Protocol):
    """Protocol for all tools. Implement this to add a new tool."""

    @property
    def name(self) -> str:
        """Unique tool identifier."""
        ...

    @property
    def description(self) -> str:
        """When to use this tool (part of LLM routing prompt)."""
        ...

    @property
    def keywords(self) -> list[str]:
        """Trigger words for keyword-based routing."""
        ...

    @property
    def patterns(self) -> list[re.Pattern[str]]:
        """Regex patterns for pattern-based routing."""
        ...

    async def execute(self, query: str) -> ToolResult:
        """Execute the tool and return structured result."""
        ...
