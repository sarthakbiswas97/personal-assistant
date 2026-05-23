"""Wikipedia lookup tool.

Provides factual grounding by retrieving Wikipedia summaries.
Reduces hallucination on knowledge-intensive queries by giving
the model authoritative source material before it responds.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from src.tools.base import ToolResult

logger = logging.getLogger(__name__)

_KEYWORDS = [
    "wikipedia", "wiki", "who is", "who was",
    "what is", "what are", "define", "explain",
    "tell me about", "history of", "biography",
]

_PATTERNS = [
    re.compile(r"(?:tell me|what do you know) about .+", re.IGNORECASE),
    re.compile(r"(?:who|what) (?:is|was|are|were) .+", re.IGNORECASE),
]

_MAX_SUMMARY_SENTENCES = 4


class WikipediaTool:
    """Wikipedia summary lookup for factual grounding."""

    @property
    def name(self) -> str:
        return "Wikipedia"

    @property
    def description(self) -> str:
        return (
            "Look up factual information from Wikipedia. Use for questions "
            "about people, places, concepts, history, science, or any topic "
            "where authoritative factual grounding reduces hallucination risk."
        )

    @property
    def keywords(self) -> list[str]:
        return _KEYWORDS

    @property
    def patterns(self) -> list[re.Pattern[str]]:
        return _PATTERNS

    async def execute(self, query: str) -> ToolResult:
        """Look up a Wikipedia summary for the query topic."""
        start = time.perf_counter()

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._lookup_sync, query
            )
            latency = (time.perf_counter() - start) * 1000

            if not result:
                return ToolResult(
                    tool_name=self.name,
                    query=query,
                    error="No Wikipedia article found",
                    latency_ms=latency,
                )

            return ToolResult(
                tool_name=self.name,
                query=query,
                data=result,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.warning("Wikipedia lookup failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                query=query,
                error=str(e),
                latency_ms=latency,
            )

    @staticmethod
    def _lookup_sync(query: str) -> dict[str, str] | None:
        """Synchronous Wikipedia lookup (run in executor)."""
        import wikipedia

        # Extract the topic from common query patterns
        topic = _extract_topic(query)

        try:
            page = wikipedia.page(topic, auto_suggest=True)
            summary = wikipedia.summary(
                topic,
                sentences=_MAX_SUMMARY_SENTENCES,
                auto_suggest=True,
            )
            return {
                "title": page.title,
                "summary": summary,
                "url": page.url,
            }
        except wikipedia.DisambiguationError as e:
            # Take the first option from disambiguation
            if e.options:
                first = e.options[0]
                try:
                    summary = wikipedia.summary(
                        first, sentences=_MAX_SUMMARY_SENTENCES
                    )
                    page = wikipedia.page(first)
                    return {
                        "title": page.title,
                        "summary": summary,
                        "url": page.url,
                        "note": f"Disambiguated to '{first}'",
                    }
                except Exception:
                    return None
            return None
        except wikipedia.PageError:
            return None


def _extract_topic(query: str) -> str:
    """Extract the likely topic from a natural language query."""
    # Remove common question prefixes
    prefixes = [
        r"^(?:tell me|what do you know) about\s+",
        r"^(?:who|what) (?:is|was|are|were)\s+",
        r"^(?:define|explain)\s+",
        r"^(?:search|find|look up|lookup)\s+(?:for\s+)?",
        r"^(?:history of|biography of)\s+",
    ]
    topic = query.strip()
    for prefix in prefixes:
        topic = re.sub(prefix, "", topic, flags=re.IGNORECASE).strip()

    # Remove trailing punctuation
    topic = topic.rstrip("?.!")
    return topic if topic else query
