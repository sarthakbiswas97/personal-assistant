"""Layer 2: LLM-based conversation summarization.

When the working memory window overflows, evicted turns are compressed
into a running summary. The summary is updated incrementally — each
time turns are evicted, the existing summary + new turns are condensed
into an updated summary.

Falls back to simple extractive compression if no LLM is available.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from src.models.base import Message

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
You are a conversation summarizer. Given the existing summary and new \
conversation turns, produce an updated summary that preserves:
- Key facts, names, numbers, and decisions
- User preferences and requests
- Important context for continuing the conversation

Be concise (3-5 sentences max). Do not include greetings or filler.\
"""

_SUMMARIZE_USER_TEMPLATE = """\
EXISTING SUMMARY:
{existing_summary}

NEW TURNS TO INCORPORATE:
{new_turns}

Write the updated summary:\
"""


class ConversationSummarizer:
    """Compresses evicted conversation turns into a running summary."""

    def __init__(self, api_key: str = "", model_name: str = "gpt-4.1-mini") -> None:
        self._client: AsyncOpenAI | None = None
        if api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        self._model_name = model_name

    async def summarize(
        self, existing_summary: str, evicted_messages: list[Message]
    ) -> str:
        """Update the running summary with newly evicted turns.

        Args:
            existing_summary: The current conversation summary (may be empty).
            evicted_messages: Messages that were evicted from working memory.

        Returns:
            Updated summary string.
        """
        if not evicted_messages:
            return existing_summary

        new_turns = _format_turns(evicted_messages)

        if self._client is not None:
            return await self._llm_summarize(existing_summary, new_turns)

        # Fallback: extractive compression (no LLM available)
        return _extractive_fallback(existing_summary, new_turns)

    async def _llm_summarize(self, existing_summary: str, new_turns: str) -> str:
        """Use frontier model to generate a compressed summary."""
        user_content = _SUMMARIZE_USER_TEMPLATE.format(
            existing_summary=existing_summary or "(none)",
            new_turns=new_turns,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _SUMMARIZE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            summary = response.choices[0].message.content or ""
            logger.info("Summarized %d chars of turns into %d char summary",
                        len(new_turns), len(summary))
            return summary.strip()
        except Exception:
            logger.warning("LLM summarization failed, using fallback", exc_info=True)
            return _extractive_fallback(existing_summary, new_turns)

    async def cleanup(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()


def _format_turns(messages: list[Message]) -> str:
    """Format messages into a readable transcript."""
    lines = []
    for msg in messages:
        role = msg.role.capitalize()
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _extractive_fallback(existing_summary: str, new_turns: str) -> str:
    """Simple fallback: append first line of each turn to the summary."""
    parts = [existing_summary] if existing_summary else []
    for line in new_turns.split("\n"):
        trimmed = line.strip()
        if trimmed:
            # Take first 100 chars of each turn
            parts.append(trimmed[:100])
    # Cap total summary length
    combined = " | ".join(parts)
    return combined[:500]
