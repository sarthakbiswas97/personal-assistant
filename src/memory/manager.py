"""MemoryManager: orchestrates all 3 memory layers.

Layer 1 (WorkingMemory)  — recent turns, verbatim
Layer 2 (Summarizer)     — compresses evicted turns into running summary
Layer 3 (Redis)          — persists state across restarts

The manager exposes the same simple interface the app uses:
    add_user_message, add_assistant_message, get_snapshot, reset
"""

from __future__ import annotations

import asyncio
import logging

# Optional import — avoids circular dependency
from typing import TYPE_CHECKING

from src.memory.persistence import RedisSessionStore
from src.memory.summarizer import ConversationSummarizer
from src.memory.working import ConversationSnapshot, WorkingMemory
from src.models.base import Message

if TYPE_CHECKING:
    from src.observability import MetricsCollector

logger = logging.getLogger(__name__)


class MemoryManager:
    """Tiered context manager composing working memory, summarizer, and persistence."""

    def __init__(
        self,
        session_id: str,
        working: WorkingMemory,
        summarizer: ConversationSummarizer,
        store: RedisSessionStore,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._session_id = session_id
        self._working = working
        self._summarizer = summarizer
        self._store = store
        self._metrics = metrics
        self._initialized = False

    async def initialize(self) -> None:
        """Load existing session from Redis if available."""
        if self._initialized:
            return

        state = await self._store.load_session(self._session_id)
        if state is not None:
            self._working.messages = [
                Message(role=m["role"], content=m["content"])
                for m in state.messages
            ]
            self._working.summary = state.summary
            if self._metrics:
                asyncio.create_task(self._metrics.record_session_restore())
            logger.info(
                "Restored session %s: %d messages, summary=%d chars",
                self._session_id,
                len(state.messages),
                len(state.summary),
            )

        self._initialized = True

    async def add_user_message(self, content: str) -> None:
        """Add a user message, summarize if overflow, persist to Redis."""
        await self.initialize()

        evicted = self._working.add_user_message(content)

        if evicted:
            updated_summary = await self._summarizer.summarize(
                self._working.summary, evicted
            )
            self._working.summary = updated_summary
            if self._metrics:
                asyncio.create_task(self._metrics.record_summarization())
            logger.info(
                "Summarized %d evicted messages, summary now %d chars",
                len(evicted),
                len(updated_summary),
            )

        # Persist to Redis in background (non-blocking)
        asyncio.create_task(self._persist())

    async def add_assistant_message(self, content: str) -> None:
        """Add an assistant message and persist to Redis."""
        self._working.add_assistant_message(content)
        asyncio.create_task(self._persist())

    def get_snapshot(self) -> ConversationSnapshot:
        """Get the current conversation snapshot for model input."""
        return self._working.get_snapshot()

    async def reset(self) -> None:
        """Clear all memory and delete the Redis session."""
        self._working.reset()
        await self._store.delete_session(self._session_id)

    @property
    def turn_count(self) -> int:
        return self._working.turn_count

    @property
    def summary(self) -> str:
        return self._working.summary

    async def _persist(self) -> None:
        """Save current state to Redis."""
        await self._store.save_session(
            self._session_id,
            self._working.messages,
            self._working.summary,
        )
