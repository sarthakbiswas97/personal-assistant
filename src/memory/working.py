"""Layer 1: Working memory with sliding window over recent turns.

When the window overflows, evicted turns are returned to the caller
(the MemoryManager) for summarization instead of being dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.base import Message


@dataclass(frozen=True)
class ConversationSnapshot:
    """Immutable snapshot of conversation state for model input."""

    system_message: Message | None
    summary_message: Message | None
    messages: tuple[Message, ...]

    def to_message_list(self) -> list[Message]:
        """Return full message list: system + summary + recent turns."""
        result: list[Message] = []
        if self.system_message is not None:
            result.append(self.system_message)
        if self.summary_message is not None:
            result.append(self.summary_message)
        result.extend(self.messages)
        return result


class WorkingMemory:
    """Sliding window over recent conversation turns.

    Unlike the old ConversationMemory, this class returns evicted
    messages on overflow instead of dropping them. The caller is
    responsible for compressing evicted turns (e.g., via summarization).
    """

    def __init__(self, max_turns: int = 10, system_prompt: str = "") -> None:
        self._max_turns = max_turns
        self._system_message: Message | None = (
            Message(role="system", content=system_prompt)
            if system_prompt
            else None
        )
        self._messages: list[Message] = []
        self._summary: str = ""

    @property
    def turn_count(self) -> int:
        """Number of complete user-assistant turn pairs stored."""
        return sum(1 for m in self._messages if m.role == "user")

    @property
    def message_count(self) -> int:
        """Total number of messages (excluding system prompt and summary)."""
        return len(self._messages)

    @property
    def summary(self) -> str:
        """Current conversation summary."""
        return self._summary

    @summary.setter
    def summary(self, value: str) -> None:
        self._summary = value

    @property
    def messages(self) -> list[Message]:
        """Current messages in the working window."""
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._messages = list(value)

    def add_user_message(self, content: str) -> list[Message]:
        """Record a user message and return any evicted turns.

        Returns:
            List of evicted Message objects (empty if no overflow).
        """
        self._messages = [*self._messages, Message(role="user", content=content)]
        return self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Record an assistant response."""
        self._messages = [*self._messages, Message(role="assistant", content=content)]

    def get_snapshot(self) -> ConversationSnapshot:
        """Return an immutable snapshot of the current conversation state."""
        summary_msg = (
            Message(role="system", content=f"Conversation context so far:\n{self._summary}")
            if self._summary
            else None
        )
        return ConversationSnapshot(
            system_message=self._system_message,
            summary_message=summary_msg,
            messages=tuple(self._messages),
        )

    def reset(self) -> None:
        """Clear conversation history and summary, preserving system prompt."""
        self._messages = []
        self._summary = ""

    def _trim(self) -> list[Message]:
        """Remove oldest turns if over the limit. Return evicted messages."""
        evicted: list[Message] = []
        while self.turn_count > self._max_turns:
            if not self._messages:
                break
            # Evict the oldest message (should be a user message)
            evicted.append(self._messages[0])
            rest = self._messages[1:]
            # If next is an assistant response, evict it too (complete pair)
            if rest and rest[0].role == "assistant":
                evicted.append(rest[0])
                rest = rest[1:]
            self._messages = rest
        return evicted
