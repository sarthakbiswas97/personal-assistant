"""Tests for frontier model backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.base import Message
from src.models.frontier_model import FrontierModel


@pytest.fixture
def model() -> FrontierModel:
    return FrontierModel(api_key="test-key", model_name="gpt-4.1-mini")


def _make_completion_response(content: str) -> MagicMock:
    """Create a mock chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_stream_chunks(tokens: list[str]) -> list[MagicMock]:
    """Create mock streaming chunks."""
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = token
        chunks.append(chunk)
    # Final chunk with None content
    final = MagicMock()
    final.choices = [MagicMock()]
    final.choices[0].delta.content = None
    chunks.append(final)
    return chunks


class TestFrontierModel:
    """Tests for FrontierModel OpenAI backend."""

    @pytest.mark.asyncio
    async def test_generate_returns_response(self, model: FrontierModel) -> None:
        mock_response = _make_completion_response("Hello there!")
        model._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        result = await model.generate(messages)

        assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_generate_passes_correct_model(self, model: FrontierModel) -> None:
        mock_response = _make_completion_response("Ok")
        model._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await model.generate([Message(role="user", content="test")])

        call_kwargs = model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1-mini"

    @pytest.mark.asyncio
    async def test_generate_formats_messages(self, model: FrontierModel) -> None:
        mock_response = _make_completion_response("Ok")
        model._client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]
        await model.generate(messages)

        call_kwargs = model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]

    @pytest.mark.asyncio
    async def test_generate_handles_none_content(self, model: FrontierModel) -> None:
        mock_response = _make_completion_response(None)
        mock_response.choices[0].message.content = None
        model._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await model.generate([Message(role="user", content="test")])
        assert result == ""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, model: FrontierModel) -> None:
        chunks = _make_stream_chunks(["Hello", " ", "world"])

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        model._client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [Message(role="user", content="Hi")]
        tokens = [token async for token in model.stream(messages)]

        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_cleanup_closes_client(self, model: FrontierModel) -> None:
        model._client.close = AsyncMock()
        await model.cleanup()
        model._client.close.assert_called_once()
