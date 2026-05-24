"""Hybrid tool router: keyword heuristic + LLM fallback.

First tries fast keyword/pattern matching against registered tools.
If no match, optionally asks the frontier LLM to classify the query.
This two-tier approach balances speed (most queries) with intelligence
(ambiguous queries).
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from src.tools.base import Tool

# Queries that never need tools — skip LLM fallback entirely.
# Keyword heuristic still runs (in case someone says "search" as a greeting).
_CHITCHAT = frozenset({
    "hello", "hi", "hey", "howdy", "greetings",
    "thanks", "thank you", "thx",
    "bye", "goodbye", "see you",
    "ok", "okay", "sure", "cool", "great", "nice",
    "yes", "no", "yeah", "nah", "yep", "nope",
    "help", "help me",
    "good morning", "good evening", "good night",
    "how are you", "what's up", "whats up",
})

logger = logging.getLogger(__name__)

_LLM_ROUTER_PROMPT = """\
You are a tool router. Given a user query, decide which tools (if any) should be called.

Available tools:
{tool_descriptions}

Respond with a JSON object: {{"tools": ["tool_name1", "tool_name2"]}}
If no tools are needed, respond with: {{"tools": []}}

Rules:
- Only select tools that are clearly needed for the query.
- Chitchat, opinions, and general knowledge do NOT need tools.
- Math expressions need Calculator.
- Questions about current events or recent news need Web Search.
- Questions seeking factual/encyclopedic knowledge need Wikipedia.
- A query can need 0, 1, or multiple tools.\
"""


class ToolRouter:
    """Routes user queries to the appropriate tools."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-4.1-mini",
    ) -> None:
        self._client: AsyncOpenAI | None = None
        if api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        self._model_name = model_name

    def route_keyword(self, query: str, tools: dict[str, Tool]) -> list[str]:
        """Fast keyword + pattern matching. Returns list of tool names."""
        q_lower = query.lower()
        matched: list[str] = []

        for name, tool in tools.items():
            # Check keywords
            if any(kw in q_lower for kw in tool.keywords):
                matched.append(name)
                continue

            # Check regex patterns
            if any(p.search(query) for p in tool.patterns):
                matched.append(name)

        return matched

    async def route_llm(
        self, query: str, tools: dict[str, Tool]
    ) -> list[str]:
        """LLM-based classification fallback. Returns list of tool names."""
        if self._client is None:
            return []

        tool_descriptions = "\n".join(
            f"- {name}: {tool.description}" for name, tool in tools.items()
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": _LLM_ROUTER_PROMPT.format(
                            tool_descriptions=tool_descriptions
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                max_tokens=50,
                temperature=0,
            )

            raw = response.choices[0].message.content or "{}"
            result = json.loads(raw)
            tool_names = result.get("tools", [])

            # Validate tool names exist
            valid = [n for n in tool_names if n in tools]
            if valid:
                logger.info("LLM router matched tools: %s", valid)
            return valid
        except Exception:
            logger.warning("LLM router failed, no tools selected", exc_info=True)
            return []

    async def route(self, query: str, tools: dict[str, Tool]) -> list[str]:
        """Hybrid routing: keyword first, LLM fallback if no match.

        Flow:
        1. Always run keyword heuristic (instant, catches "search AI" etc.)
        2. If no keyword match, check if query is chitchat → skip LLM
        3. Otherwise, fall back to LLM classifier for subtle queries
        """
        # Fast path: keyword heuristic (always runs, even for short queries)
        matched = self.route_keyword(query, tools)
        if matched:
            logger.debug("Keyword router matched: %s", matched)
            return matched

        # Skip LLM fallback for chitchat (no point classifying "hello")
        if self._is_chitchat(query):
            return []

        # Slow path: LLM classification for substantive queries
        matched = await self.route_llm(query, tools)
        return matched

    @staticmethod
    def _is_chitchat(query: str) -> bool:
        """Check if query is conversational filler that never needs tools."""
        normalized = query.strip().lower().rstrip("?!.")
        return normalized in _CHITCHAT

    async def cleanup(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()
