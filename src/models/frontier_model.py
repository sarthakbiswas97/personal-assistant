"""Frontier model backend using OpenAI API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from src.models.base import BaseModel, Message

logger = logging.getLogger(__name__)


class FrontierModel(BaseModel):
    """OpenAI GPT-4.1 / GPT-4.1-mini backend.

    Uses the async OpenAI client for non-blocking inference.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4.1-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model_name = model_name

    async def generate(self, messages: list[Message]) -> str:
        """Generate a complete response via OpenAI chat completions."""
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[m.to_dict() for m in messages],
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response tokens via OpenAI chat completions."""
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[m.to_dict() for m in messages],
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta

    async def cleanup(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
