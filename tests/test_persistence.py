"""Tests for Redis session persistence."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.persistence import RedisSessionStore, SessionState
from src.models.base import Message


class TestSessionState:
    """Tests for SessionState serialization."""

    def test_roundtrip_json(self) -> None:
        messages = [Message(role="user", content="Hello")]
        state = SessionState.from_memory(messages, "Some summary")
        json_str = state.to_json()
        restored = SessionState.from_json(json_str)

        assert restored.summary == "Some summary"
        assert len(restored.messages) == 1
        assert restored.messages[0]["content"] == "Hello"

    def test_from_memory_sets_timestamps(self) -> None:
        state = SessionState.from_memory([], "")
        assert state.created_at > 0
        assert state.updated_at > 0


class TestRedisSessionStore:
    """Tests for RedisSessionStore with mocked Redis."""

    def test_unavailable_when_no_url(self) -> None:
        store = RedisSessionStore(redis_url="")
        assert not store.is_available

    @patch("src.memory.persistence.aioredis")
    def test_available_when_url_provided(self, mock_aioredis: MagicMock) -> None:
        store = RedisSessionStore(redis_url="redis://localhost:6379")
        assert store.is_available

    async def test_save_noop_when_unavailable(self) -> None:
        store = RedisSessionStore(redis_url="")
        # Should not raise
        await store.save_session("test", [Message(role="user", content="Hi")], "")

    async def test_load_returns_none_when_unavailable(self) -> None:
        store = RedisSessionStore(redis_url="")
        result = await store.load_session("test")
        assert result is None

    @patch("src.memory.persistence.aioredis")
    async def test_save_and_load(self, mock_aioredis: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_aioredis.from_url.return_value = mock_client

        store = RedisSessionStore(redis_url="redis://localhost:6379")

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]

        # Save
        await store.save_session("sess_1", messages, "User said hello")
        mock_client.set.assert_called_once()

        # Verify key format
        call_args = mock_client.set.call_args
        assert call_args[0][0] == "session:sess_1"

    @patch("src.memory.persistence.aioredis")
    async def test_load_refreshes_ttl(self, mock_aioredis: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_aioredis.from_url.return_value = mock_client

        state = SessionState.from_memory(
            [Message(role="user", content="Hi")], "summary"
        )
        mock_client.get.return_value = state.to_json()

        store = RedisSessionStore(redis_url="redis://localhost:6379")
        result = await store.load_session("sess_1")

        assert result is not None
        assert result.summary == "summary"
        # Verify TTL was refreshed
        mock_client.expire.assert_called_once()

    @patch("src.memory.persistence.aioredis")
    async def test_delete_session(self, mock_aioredis: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_aioredis.from_url.return_value = mock_client

        store = RedisSessionStore(redis_url="redis://localhost:6379")
        await store.delete_session("sess_1")
        mock_client.delete.assert_called_once_with("session:sess_1")

    @patch("src.memory.persistence.aioredis")
    async def test_save_failure_does_not_raise(self, mock_aioredis: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.set.side_effect = Exception("Connection lost")
        mock_aioredis.from_url.return_value = mock_client

        store = RedisSessionStore(redis_url="redis://localhost:6379")
        # Should log warning but not crash
        await store.save_session("sess_1", [], "")
