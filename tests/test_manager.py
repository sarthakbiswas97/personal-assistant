"""Tests for MemoryManager orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.manager import MemoryManager
from src.memory.persistence import RedisSessionStore, SessionState
from src.memory.summarizer import ConversationSummarizer
from src.memory.working import WorkingMemory
from src.models.base import Message


@pytest.fixture
def mock_store() -> RedisSessionStore:
    store = MagicMock(spec=RedisSessionStore)
    store.load_session = AsyncMock(return_value=None)
    store.save_session = AsyncMock()
    store.delete_session = AsyncMock()
    return store


@pytest.fixture
def mock_summarizer() -> ConversationSummarizer:
    summarizer = MagicMock(spec=ConversationSummarizer)
    summarizer.summarize = AsyncMock(return_value="Summarized context.")
    return summarizer


@pytest.fixture
def manager(mock_store, mock_summarizer) -> MemoryManager:
    working = WorkingMemory(max_turns=2, system_prompt="Be helpful.")
    return MemoryManager(
        session_id="test_session",
        working=working,
        summarizer=mock_summarizer,
        store=mock_store,
    )


class TestMemoryManager:
    """Tests for MemoryManager orchestration."""

    async def test_add_message_and_get_snapshot(self, manager: MemoryManager) -> None:
        await manager.add_user_message("Hello")
        await manager.add_assistant_message("Hi!")

        snapshot = manager.get_snapshot()
        messages = snapshot.to_message_list()

        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Hi!"

    async def test_summarizes_on_overflow(
        self, manager: MemoryManager, mock_summarizer: MagicMock
    ) -> None:
        # Fill up 2 turns
        await manager.add_user_message("Turn 1")
        await manager.add_assistant_message("Response 1")
        await manager.add_user_message("Turn 2")
        await manager.add_assistant_message("Response 2")

        # Third turn should trigger eviction + summarization
        await manager.add_user_message("Turn 3")

        mock_summarizer.summarize.assert_called_once()
        assert manager.summary == "Summarized context."

    async def test_summary_appears_in_snapshot(
        self, manager: MemoryManager
    ) -> None:
        await manager.add_user_message("Turn 1")
        await manager.add_assistant_message("Response 1")
        await manager.add_user_message("Turn 2")
        await manager.add_assistant_message("Response 2")
        await manager.add_user_message("Turn 3")

        snapshot = manager.get_snapshot()
        messages = snapshot.to_message_list()

        # Should have: system + summary + recent turns
        roles = [m.role for m in messages]
        assert roles[0] == "system"
        assert roles[1] == "system"  # summary as system message
        assert "Summarized context." in messages[1].content

    async def test_persists_on_message(
        self, manager: MemoryManager, mock_store: MagicMock
    ) -> None:
        await manager.add_user_message("Hello")
        # Give background task a moment to run
        await asyncio.sleep(0.1)

        mock_store.save_session.assert_called()

    async def test_restores_from_redis(
        self, mock_store: MagicMock, mock_summarizer: MagicMock
    ) -> None:
        # Simulate existing session in Redis
        state = SessionState(
            messages=[
                {"role": "user", "content": "Previous msg"},
                {"role": "assistant", "content": "Previous response"},
            ],
            summary="User discussed ML earlier.",
            created_at=1000.0,
            updated_at=1000.0,
        )
        mock_store.load_session = AsyncMock(return_value=state)

        working = WorkingMemory(max_turns=5, system_prompt="System.")
        mgr = MemoryManager(
            session_id="restored",
            working=working,
            summarizer=mock_summarizer,
            store=mock_store,
        )

        await mgr.add_user_message("New message")

        snapshot = mgr.get_snapshot()
        messages = snapshot.to_message_list()

        # Should have: system + summary + restored + new
        contents = [m.content for m in messages]
        assert "System." in contents[0]
        assert "ML earlier" in contents[1]  # summary
        assert "Previous msg" in contents[2]
        assert "New message" in contents[-1]

    async def test_reset_clears_everything(
        self, manager: MemoryManager, mock_store: MagicMock
    ) -> None:
        await manager.add_user_message("Hello")
        await manager.reset()

        assert manager.turn_count == 0
        assert manager.summary == ""
        mock_store.delete_session.assert_called_once_with("test_session")


import asyncio
