"""Task status manager backed by Redis."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from app.config import settings
from app.core.redis import get_redis_client

logger = logging.getLogger(__name__)

TASK_KEY_PREFIX = "task"
UNSET: object = object()


class TaskManager:
    """Store and fetch task progress from Redis."""

    def __init__(self, redis_client: Redis | None = None) -> None:
        self.redis = redis_client or get_redis_client()
        self.ttl_seconds = settings.cache_ttl_seconds

    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get task status by task ID."""
        raw = await self.redis.get(self._task_key(task_id))
        if raw is None:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid task payload in Redis", extra={"task_id": task_id})
            return None

    async def set_task_state(
        self,
        task_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        project_id: str | None = None,
        pipeline_module: str | None = None,
        source_topic_id: str | None = None,
        current_step: int | None = None,
        current_step_name: str | None = None,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        progress_percent: float | None = None,
        error_message: str | None | object = UNSET,
    ) -> dict[str, Any]:
        """Create or update task state."""
        now = self._now_iso()
        payload = await self.get_task_status(task_id) or {"task_id": task_id, "created_at": now}
        payload["updated_at"] = now
        effective_pipeline_module = pipeline_module or payload.get("pipeline_module")
        display_current_step = self._display_step_number(
            pipeline_module=str(effective_pipeline_module) if effective_pipeline_module else None,
            current_step=current_step,
        )

        updates = {
            "status": status,
            "stage": stage,
            "project_id": project_id,
            "pipeline_module": pipeline_module,
            "source_topic_id": source_topic_id,
            "current_step": display_current_step,
            "current_step_name": current_step_name,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "progress_percent": progress_percent,
        }
        for key, value in updates.items():
            if value is not None:
                payload[key] = value
        if error_message is not UNSET:
            payload["error_message"] = error_message

        await self.redis.set(
            self._task_key(task_id),
            json.dumps(payload),
            ex=self.ttl_seconds,
        )
        return payload

    async def mark_step_completed(
        self,
        task_id: str,
        step_number: int,
        step_name: str,
    ) -> dict[str, Any]:
        """Increment completion counters for a task."""
        payload = await self.get_task_status(task_id) or {"task_id": task_id}
        pipeline_module = payload.get("pipeline_module")
        display_step = self._display_step_number(
            pipeline_module=str(pipeline_module) if pipeline_module else None,
            current_step=step_number,
        ) or step_number
        completed_steps = self._to_int(payload.get("completed_steps")) + 1
        total_steps = self._to_int(payload.get("total_steps"), default=0)
        progress_percent = (
            round((completed_steps / total_steps) * 100, 2) if total_steps > 0 else None
        )

        return await self.set_task_state(
            task_id=task_id,
            status="running",
            stage=f"Completed step {display_step}: {step_name}",
            current_step=step_number,
            current_step_name=step_name,
            completed_steps=completed_steps,
            progress_percent=progress_percent,
            error_message=None,
        )

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"{TASK_KEY_PREFIX}:{task_id}"

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _display_step_number(
        *,
        pipeline_module: str | None,
        current_step: int | None,
    ) -> int | None:
        if current_step is None:
            return None
        if pipeline_module == "discovery":
            return max(1, current_step - 1)
        if pipeline_module == "setup":
            return current_step + 1
        return current_step

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
