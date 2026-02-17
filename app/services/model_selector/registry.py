"""Snapshot registry for runtime model selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from redis.asyncio import Redis

from app.config import settings
from app.core.redis import get_redis_client
from app.services.model_selector.types import SelectionSnapshot

logger = logging.getLogger(__name__)

SNAPSHOT_REDIS_KEY = "model_selector:snapshot:latest"
MODEL_REDIS_KEY_PREFIX = "model_selector:selected"

_cached_snapshot: SelectionSnapshot | None = None
_cached_snapshot_path: Path | None = None
_cached_snapshot_mtime_ns: int | None = None


def get_agent_model(environment: str, agent_class_name: str) -> str | None:
    """Resolve selected model from local snapshot file."""
    snapshot = load_snapshot()
    if snapshot is None:
        return None
    return snapshot.get_agent_model(environment, agent_class_name)


def load_snapshot(snapshot_path: str | Path | None = None) -> SelectionSnapshot | None:
    """Load snapshot from disk with mtime-based in-process cache."""
    global _cached_snapshot
    global _cached_snapshot_mtime_ns
    global _cached_snapshot_path

    path = Path(snapshot_path or settings.model_selector_snapshot_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    if (
        _cached_snapshot is not None
        and _cached_snapshot_path == path
        and _cached_snapshot_mtime_ns == stat.st_mtime_ns
    ):
        return _cached_snapshot

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to load model selector snapshot",
            extra={"path": str(path), "error": str(exc)},
        )
        return None

    if not isinstance(payload, dict):
        logger.warning("Model selector snapshot has invalid root type", extra={"path": str(path)})
        return None

    snapshot = SelectionSnapshot.from_dict(payload)
    _cached_snapshot = snapshot
    _cached_snapshot_path = path
    _cached_snapshot_mtime_ns = stat.st_mtime_ns
    return snapshot


async def write_snapshot_to_redis(
    snapshot: SelectionSnapshot,
    redis_client: Redis | None = None,
    ttl_seconds: int | None = None,
) -> None:
    """Write snapshot plus per-agent keys to Redis."""
    redis = redis_client or get_redis_client()
    ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
    payload = snapshot.to_dict()
    serialized = json.dumps(payload)

    await redis.set(SNAPSHOT_REDIS_KEY, serialized, ex=ttl)

    environments = payload.get("environments", {})
    if not isinstance(environments, dict):
        return

    for environment, env_payload in environments.items():
        if not isinstance(env_payload, dict):
            continue
        agents_payload = env_payload.get("agents")
        if not isinstance(agents_payload, dict):
            continue

        for agent_class, agent_data in agents_payload.items():
            if not isinstance(agent_data, dict):
                continue
            model = agent_data.get("model")
            if not isinstance(model, str) or not model:
                continue
            key = redis_key_for_agent(environment, agent_class)
            await redis.set(key, model, ex=ttl)


async def get_agent_model_from_redis(
    environment: str,
    agent_class_name: str,
    redis_client: Redis | None = None,
) -> str | None:
    """Resolve selected model from Redis overlay."""
    redis = redis_client or get_redis_client()
    key = redis_key_for_agent(environment, agent_class_name)
    model = await redis.get(key)
    if isinstance(model, str) and model:
        return model
    return None


def redis_key_for_agent(environment: str, agent_class_name: str) -> str:
    """Build Redis key for per-agent model selection."""
    return f"{MODEL_REDIS_KEY_PREFIX}:{environment}:{agent_class_name}"


def reset_snapshot_cache() -> None:
    """Test helper to clear in-process snapshot cache."""
    global _cached_snapshot
    global _cached_snapshot_mtime_ns
    global _cached_snapshot_path

    _cached_snapshot = None
    _cached_snapshot_mtime_ns = None
    _cached_snapshot_path = None
