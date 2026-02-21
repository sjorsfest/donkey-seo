"""Redis client helpers."""

import logging
from typing import Any, Awaitable, cast

from redis.asyncio import Redis

from app.config import settings

logger = logging.getLogger(__name__)

_redis_client: Redis | None = None


class RedisQueueClient:
    """Typed queue operations over the shared Redis client."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def llen(self, key: str) -> int:
        """Return queue length."""
        value = await cast(Awaitable[int], self._client.llen(key))
        return int(value)

    async def rpush(self, key: str, payload: str) -> int:
        """Push payload to the queue tail."""
        value = await cast(Awaitable[int], self._client.rpush(key, payload))
        return int(value)

    async def blpop(self, key: str, *, timeout: int) -> tuple[str, str] | None:
        """Pop one payload from queue head."""
        raw = await cast(
            Awaitable[list[Any] | None],
            self._client.blpop([key], timeout=timeout),
        )
        if not raw or len(raw) < 2:
            return None
        queue_name = str(raw[0])
        payload = str(raw[1])
        return queue_name, payload


def get_redis_client() -> Redis:
    """Get a shared Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def get_redis_queue_client() -> RedisQueueClient:
    """Get typed queue operations on the shared Redis client."""
    return RedisQueueClient(get_redis_client())


async def close_redis() -> None:
    """Close Redis client connections."""
    global _redis_client
    if _redis_client is None:
        return

    await _redis_client.aclose()
    _redis_client = None
    logger.info("Redis connection closed")
