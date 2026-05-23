"""Layer 3: Redis-backed session persistence.

Stores conversation state (messages, summary, metadata) in Redis
with TTL-based expiration. Operates as a no-op when Redis is
unavailable, allowing the app to run in memory-only mode.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass

import redis.asyncio as aioredis

from src.models.base import Message

logger = logging.getLogger(__name__)

_SESSION_PREFIX = "session:"
_DEFAULT_TTL_SECONDS = 3600  # 1 hour


@dataclass(frozen=True)
class SessionState:
    """Serializable conversation state for persistence."""

    messages: list[dict[str, str]]
    summary: str
    created_at: float
    updated_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: str) -> SessionState:
        parsed = json.loads(data)
        return SessionState(**parsed)

    @staticmethod
    def from_memory(messages: list[Message], summary: str) -> SessionState:
        now = time.time()
        return SessionState(
            messages=[m.to_dict() for m in messages],
            summary=summary,
            created_at=now,
            updated_at=now,
        )


class RedisSessionStore:
    """Async Redis session store with graceful degradation.

    When Redis is unavailable, all operations silently no-op.
    The app continues to work in memory-only mode.
    """

    def __init__(self, redis_url: str = "", ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._client: aioredis.Redis | None = None
        self._ttl = ttl_seconds
        self._available = False

        if redis_url:
            try:
                self._client = aioredis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                self._available = True
                logger.info("Redis session store initialized")
            except Exception:
                logger.warning("Failed to create Redis client, running in memory-only mode",
                               exc_info=True)

    @property
    def is_available(self) -> bool:
        return self._available

    async def ping(self) -> bool:
        """Test Redis connectivity."""
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except Exception:
            self._available = False
            logger.warning("Redis ping failed, switching to memory-only mode")
            return False

    async def save_session(
        self, session_id: str, messages: list[Message], summary: str
    ) -> None:
        """Persist conversation state to Redis.

        Non-blocking — failures are logged but don't propagate.
        """
        if not self._available:
            return

        key = f"{_SESSION_PREFIX}{session_id}"
        state = SessionState.from_memory(messages, summary)

        try:
            await self._client.set(key, state.to_json(), ex=self._ttl)
            logger.debug("Saved session %s (%d messages)", session_id, len(messages))
        except Exception:
            logger.warning("Failed to save session %s", session_id, exc_info=True)

    async def load_session(self, session_id: str) -> SessionState | None:
        """Load conversation state from Redis.

        Returns None if session doesn't exist or Redis is unavailable.
        """
        if not self._available:
            return None

        key = f"{_SESSION_PREFIX}{session_id}"

        try:
            data = await self._client.get(key)
            if data is None:
                return None

            state = SessionState.from_json(data)
            logger.debug("Loaded session %s (%d messages)", session_id, len(state.messages))
            # Refresh TTL on read (sliding expiration)
            await self._client.expire(key, self._ttl)
            return state
        except Exception:
            logger.warning("Failed to load session %s", session_id, exc_info=True)
            return None

    async def delete_session(self, session_id: str) -> None:
        """Remove a session from Redis."""
        if not self._available:
            return

        try:
            await self._client.delete(f"{_SESSION_PREFIX}{session_id}")
        except Exception:
            logger.warning("Failed to delete session %s", session_id, exc_info=True)

    async def cleanup(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            logger.info("Redis connection closed")
