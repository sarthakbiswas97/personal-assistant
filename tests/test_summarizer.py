"""Tests for conversation summarizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.memory.summarizer import (
    ConversationSummarizer,
    _extractive_fallback,
    _format_turns,
)
from src.models.base import Message


class TestFormatTurns:
    """Tests for turn formatting helper."""

    def test_formats_user_and_assistant(self) -> None:
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        result = _format_turns(messages)
        assert result == "User: Hello\nAssistant: Hi there!"

    def test_empty_messages(self) -> None:
        assert _format_turns([]) == ""


class TestExtractiveFallback:
    """Tests for the no-LLM fallback summarization."""

    def test_creates_summary_from_turns(self) -> None:
        result = _extractive_fallback("", "User: Hello\nAssistant: Hi")
        assert "User: Hello" in result
        assert "Assistant: Hi" in result

    def test_appends_to_existing_summary(self) -> None:
        result = _extractive_fallback("Previous context.", "User: New message")
        assert result.startswith("Previous context.")
        assert "User: New message" in result

    def test_caps_at_500_chars(self) -> None:
        long_turn = "User: " + "x" * 600
        result = _extractive_fallback("", long_turn)
        assert len(result) <= 500


class TestConversationSummarizer:
    """Tests for the LLM-based summarizer."""

    async def test_returns_existing_summary_when_no_evictions(self) -> None:
        summarizer = ConversationSummarizer()
        result = await summarizer.summarize("Existing context.", [])
        assert result == "Existing context."

    async def test_uses_fallback_when_no_api_key(self) -> None:
        summarizer = ConversationSummarizer(api_key="")
        messages = [
            Message(role="user", content="My name is Sarthak"),
            Message(role="assistant", content="Nice to meet you!"),
        ]
        result = await summarizer.summarize("", messages)
        assert "Sarthak" in result

    async def test_calls_llm_when_api_key_present(self) -> None:
        summarizer = ConversationSummarizer(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "User introduced themselves as Sarthak."
        summarizer._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [
            Message(role="user", content="My name is Sarthak"),
            Message(role="assistant", content="Nice to meet you!"),
        ]
        result = await summarizer.summarize("", messages)

        assert result == "User introduced themselves as Sarthak."
        summarizer._client.chat.completions.create.assert_called_once()

    async def test_falls_back_on_llm_error(self) -> None:
        summarizer = ConversationSummarizer(api_key="test-key")
        summarizer._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        messages = [Message(role="user", content="Hello")]
        result = await summarizer.summarize("", messages)

        # Should use extractive fallback, not crash
        assert "Hello" in result
