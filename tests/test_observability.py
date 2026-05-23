"""Tests for observability metrics collector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.observability import MetricsCollector


class TestMetricsCollector:
    """Tests for MetricsCollector with mocked Redis."""

    def test_unavailable_when_no_url(self) -> None:
        collector = MetricsCollector(redis_url="")
        assert not collector.is_available

    @patch("src.observability.aioredis")
    def test_available_when_url_provided(self, mock_aioredis: MagicMock) -> None:
        collector = MetricsCollector(redis_url="redis://localhost:6379")
        assert collector.is_available

    async def test_record_request_noop_when_unavailable(self) -> None:
        collector = MetricsCollector(redis_url="")
        await collector.record_request("test_model", 100.0)

    async def test_record_guardrail_block_noop_when_unavailable(self) -> None:
        collector = MetricsCollector(redis_url="")
        await collector.record_guardrail_block()

    @patch("src.observability.aioredis")
    async def test_record_request_uses_pipeline(self, mock_aioredis: MagicMock) -> None:
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock()

        # pipeline() is sync in redis.asyncio, returns a Pipeline object
        mock_client = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_aioredis.from_url.return_value = mock_client

        collector = MetricsCollector(redis_url="redis://localhost:6379")
        await collector.record_request("OSS", 150.0)

        mock_client.pipeline.assert_called_once()
        mock_pipe.incr.assert_called_once()
        mock_pipe.lpush.assert_called_once()
        mock_pipe.ltrim.assert_called_once()
        mock_pipe.execute.assert_called_once()

    @patch("src.observability.aioredis")
    async def test_record_guardrail_block(self, mock_aioredis: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_aioredis.from_url.return_value = mock_client

        collector = MetricsCollector(redis_url="redis://localhost:6379")
        await collector.record_guardrail_block()

        mock_client.incr.assert_called_once_with("metrics:guardrail_blocks")

    @patch("src.observability.aioredis")
    async def test_get_summary_returns_none_when_unavailable(
        self, mock_aioredis: MagicMock
    ) -> None:
        collector = MetricsCollector(redis_url="")
        result = await collector.get_summary()
        assert result is None

    @patch("src.observability.aioredis")
    async def test_record_error_does_not_raise_on_failure(
        self, mock_aioredis: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_client.incr.side_effect = Exception("Redis down")
        mock_aioredis.from_url.return_value = mock_client

        collector = MetricsCollector(redis_url="redis://localhost:6379")
        await collector.record_error()  # should not raise
