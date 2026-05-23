"""Safe math evaluator and datetime tool.

Handles arithmetic expressions and datetime queries deterministically.
Uses ast module for safe math parsing — never calls eval().
"""

from __future__ import annotations

import ast
import logging
import operator
import re
import time
from datetime import UTC, datetime

from src.tools.base import ToolResult

logger = logging.getLogger(__name__)

_KEYWORDS = [
    "calculate", "compute", "math",
    "what time", "what date", "current time", "current date",
    "time in", "date today",
]

_PATTERNS = [
    # Math expressions: "147 * 38 + 92", "2^10", "15% of 200"
    re.compile(r"\d+\s*[\+\-\*\/\%\^]\s*\d+"),
    # Percentage: "what is 15% of 200"
    re.compile(r"\d+\s*%\s*of\s*\d+", re.IGNORECASE),
    # Time queries
    re.compile(r"(?:what|current)\s+(?:time|date)", re.IGNORECASE),
    re.compile(r"time\s+in\s+\w+", re.IGNORECASE),
]

# Allowed operators for safe math evaluation
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool:
    """Safe math evaluator and datetime lookup."""

    @property
    def name(self) -> str:
        return "Calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate mathematical expressions or get current date/time. "
            "Use for arithmetic, percentages, or when the user asks about "
            "the current time or date. Never use for non-numeric queries."
        )

    @property
    def keywords(self) -> list[str]:
        return _KEYWORDS

    @property
    def patterns(self) -> list[re.Pattern[str]]:
        return _PATTERNS

    async def execute(self, query: str) -> ToolResult:
        """Evaluate math expression or return datetime."""
        start = time.perf_counter()

        try:
            # Check if it's a datetime query
            if _is_datetime_query(query):
                result = _handle_datetime(query)
            else:
                result = _handle_math(query)

            latency = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                query=query,
                data=result,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.warning("Calculator failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                query=query,
                error=str(e),
                latency_ms=latency,
            )


def _is_datetime_query(query: str) -> bool:
    """Check if the query is asking about date or time."""
    q = query.lower()
    return any(kw in q for kw in ["time", "date", "today", "now"])


def _handle_datetime(query: str) -> dict[str, str]:
    """Return current UTC datetime."""
    now = datetime.now(UTC)
    return {
        "type": "datetime",
        "utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%H:%M:%S UTC"),
    }


def _handle_math(query: str) -> dict[str, str]:
    """Extract and safely evaluate a math expression from the query."""
    # Handle percentage: "15% of 200" → "200 * 0.15"
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", query)
    if pct_match:
        pct = float(pct_match.group(1))
        base = float(pct_match.group(2))
        result = base * (pct / 100)
        return {
            "type": "math",
            "expression": f"{pct}% of {base}",
            "result": str(result),
        }

    # Extract math expression from query
    expr = _extract_expression(query)
    if not expr:
        raise ValueError("No valid math expression found in query")

    # Replace ^ with ** for Python power operator
    expr = expr.replace("^", "**")

    result = _safe_eval(expr)
    return {
        "type": "math",
        "expression": expr,
        "result": str(result),
    }


def _extract_expression(query: str) -> str:
    """Extract a math expression from natural language."""
    # Find sequences of digits and operators
    match = re.search(
        r"([\d.]+(?:\s*[\+\-\*\/\%\^\*]{1,2}\s*[\d.]+)+)",
        query,
    )
    if match:
        return match.group(1).strip()

    # Try to find standalone number expressions
    match = re.search(r"([\d.]+\s*[\+\-\*\/\^]\s*[\d.]+)", query)
    if match:
        return match.group(1).strip()

    return ""


def _safe_eval(expr: str) -> float:
    """Safely evaluate a math expression using AST parsing.

    Only allows numeric literals and basic arithmetic operators.
    Never calls eval() or exec().
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {expr}") from e

    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> float:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type == ast.Div and right == 0:
            raise ValueError("Division by zero")
        return _OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _OPS[op_type](_eval_node(node.operand))

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")
