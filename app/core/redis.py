"""Redis client helpers."""

import logging

from redis.asyncio import Redis

from app.config import settings

logger = logging.getLogger(__name__)

_redis_client: Redis | None = None


def get_redis_client() -> Redis:
    """Get a shared Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


async def close_redis() -> None:
    """Close Redis client connections."""
    global _redis_client
    if _redis_client is None:
        return

    await _redis_client.aclose()
    _redis_client = None
    logger.info("Redis connection closed")
