"""Conversation memory with sliding window over message history."""

from dataclasses import dataclass, field

from src.models.base import Message


@dataclass(frozen=True)
class ConversationSnapshot:
    """Immutable snapshot of conversation state for model input."""

    system_message: Message | None
    messages: tuple[Message, ...]

    def to_message_list(self) -> list[Message]:
        """Return full message list including system prompt."""
        if self.system_message is None:
            return list(self.messages)
        return [self.system_message, *self.messages]


class ConversationMemory:
    """Sliding window memory that maintains recent conversation turns.

    Keeps the system prompt separate from the conversation history.
    When the window is exceeded, oldest user-assistant pairs are dropped
    while the system prompt is always preserved.
    """

    def __init__(self, max_turns: int = 10, system_prompt: str = "") -> None:
        """Initialize memory.

        Args:
            max_turns: Maximum number of user-assistant turn pairs to retain.
            system_prompt: Optional system prompt, always prepended to output.
        """
        self._max_turns = max_turns
        self._system_message: Message | None = (
            Message(role="system", content=system_prompt)
            if system_prompt
            else None
        )
        self._messages: list[Message] = []

    @property
    def turn_count(self) -> int:
        """Number of complete user-assistant turn pairs stored."""
        return sum(1 for m in self._messages if m.role == "user")

    @property
    def message_count(self) -> int:
        """Total number of messages (excluding system prompt)."""
        return len(self._messages)

    def add_user_message(self, content: str) -> None:
        """Record a user message."""
        self._messages = [*self._messages, Message(role="user", content=content)]
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant response."""
        self._messages = [*self._messages, Message(role="assistant", content=content)]

    def get_snapshot(self) -> ConversationSnapshot:
        """Return an immutable snapshot of the current conversation state."""
        return ConversationSnapshot(
            system_message=self._system_message,
            messages=tuple(self._messages),
        )

    def reset(self) -> None:
        """Clear conversation history, preserving system prompt."""
        self._messages = []

    def _trim(self) -> None:
        """Drop oldest turns if over the max_turns limit.

        Trims from the front, always removing complete user-assistant pairs.
        If the oldest message is an assistant response without a preceding
        user message (edge case), it is also dropped.
        """
        while self.turn_count > self._max_turns:
            if not self._messages:
                break
            # Drop the oldest message (should be a user message)
            rest = self._messages[1:]
            # If next message is an assistant response, drop it too (complete pair)
            if rest and rest[0].role == "assistant":
                rest = rest[1:]
            self._messages = rest
