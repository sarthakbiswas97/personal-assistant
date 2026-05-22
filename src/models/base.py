"""Abstract base class for model backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """Immutable chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dict format expected by model APIs."""
        return {"role": self.role, "content": self.content}


class BaseModel(ABC):
    """Abstract interface for all model backends.

    Every model backend (OSS or frontier) must implement this interface
    to ensure consistent behavior across the application.
    """

    @abstractmethod
    async def generate(self, messages: list[Message]) -> str:
        """Generate a complete response for the given message history.

        Args:
            messages: Ordered conversation history including system prompt.

        Returns:
            The model's complete response text.
        """

    @abstractmethod
    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response tokens for the given message history.

        Args:
            messages: Ordered conversation history including system prompt.

        Yields:
            Individual text chunks as they become available.
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Release any resources held by the model backend."""
