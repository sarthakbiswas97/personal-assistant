"""Runtime observability with Redis-backed metrics collection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_PREFIX = "metrics:"
_MAX_LATENCY_SAMPLES = 100


@dataclass(frozen=True)
class MetricsSummary:
    """Snapshot of all collected metrics."""

    requests: dict[str, int]
    latency_avg: dict[str, float]
    latency_p50: dict[str, float]
    latency_p95: dict[str, float]
    guardrail_blocks: int
    summarizations: int
    session_restores: int
    errors: int


class MetricsCollector:
    """Collects and queries runtime metrics via Redis.

    All operations are fire-and-forget — failures are logged
    but never propagate to the caller.
    """

    def __init__(self, redis_url: str = "") -> None:
        self._client: aioredis.Redis | None = None
        self._available = False

        if redis_url:
            try:
                self._client = aioredis.from_url(
                    redis_url, decode_responses=True, socket_connect_timeout=5
                )
                self._available = True
            except Exception:
                logger.warning("Metrics collector: Redis unavailable", exc_info=True)

    @property
    def is_available(self) -> bool:
        return self._available

    async def record_request(self, model: str, latency_ms: float) -> None:
        """Record a completed model request with latency."""
        if not self._available:
            return
        try:
            pipe = self._client.pipeline()
            pipe.incr(f"{_PREFIX}requests:{model}")
            pipe.lpush(f"{_PREFIX}latency:{model}", latency_ms)
            pipe.ltrim(f"{_PREFIX}latency:{model}", 0, _MAX_LATENCY_SAMPLES - 1)
            await pipe.execute()
        except Exception:
            logger.warning("Failed to record request metric", exc_info=True)

    async def record_guardrail_block(self) -> None:
        if not self._available:
            return
        try:
            await self._client.incr(f"{_PREFIX}guardrail_blocks")
        except Exception:
            logger.warning("Failed to record guardrail metric", exc_info=True)

    async def record_summarization(self) -> None:
        if not self._available:
            return
        try:
            await self._client.incr(f"{_PREFIX}summarizations")
        except Exception:
            logger.warning("Failed to record summarization metric", exc_info=True)

    async def record_session_restore(self) -> None:
        if not self._available:
            return
        try:
            await self._client.incr(f"{_PREFIX}session_restores")
        except Exception:
            logger.warning("Failed to record session restore metric", exc_info=True)

    async def record_error(self) -> None:
        if not self._available:
            return
        try:
            await self._client.incr(f"{_PREFIX}errors")
        except Exception:
            logger.warning("Failed to record error metric", exc_info=True)

    async def get_summary(self) -> MetricsSummary | None:
        """Retrieve aggregated metrics snapshot."""
        if not self._available:
            return None

        try:
            models = set()
            keys = []
            async for key in self._client.scan_iter(f"{_PREFIX}requests:*"):
                model = key.replace(f"{_PREFIX}requests:", "")
                models.add(model)
                keys.append(key)

            requests: dict[str, int] = {}
            latency_avg: dict[str, float] = {}
            latency_p50: dict[str, float] = {}
            latency_p95: dict[str, float] = {}

            for model in models:
                req_count = await self._client.get(f"{_PREFIX}requests:{model}")
                requests[model] = int(req_count) if req_count else 0

                raw_latencies = await self._client.lrange(
                    f"{_PREFIX}latency:{model}", 0, -1
                )
                if raw_latencies:
                    latencies = sorted(float(v) for v in raw_latencies)
                    latency_avg[model] = sum(latencies) / len(latencies)
                    latency_p50[model] = latencies[len(latencies) // 2]
                    idx_95 = min(int(len(latencies) * 0.95), len(latencies) - 1)
                    latency_p95[model] = latencies[idx_95]

            guardrail_blocks = int(
                await self._client.get(f"{_PREFIX}guardrail_blocks") or 0
            )
            summarizations = int(
                await self._client.get(f"{_PREFIX}summarizations") or 0
            )
            session_restores = int(
                await self._client.get(f"{_PREFIX}session_restores") or 0
            )
            errors = int(await self._client.get(f"{_PREFIX}errors") or 0)

            return MetricsSummary(
                requests=requests,
                latency_avg=latency_avg,
                latency_p50=latency_p50,
                latency_p95=latency_p95,
                guardrail_blocks=guardrail_blocks,
                summarizations=summarizations,
                session_restores=session_restores,
                errors=errors,
            )
        except Exception:
            logger.warning("Failed to read metrics", exc_info=True)
            return None

    async def cleanup(self) -> None:
        if self._client is not None:
            await self._client.aclose()
