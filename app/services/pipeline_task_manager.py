"""Redis-backed queueing for setup/discovery/content pipeline jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.core.database import get_session_context
from app.core.exceptions import PipelineAlreadyRunningError
from app.core.redis import get_redis_client

logger = logging.getLogger(__name__)

PipelineModule = Literal["setup", "discovery", "content"]
JobKind = Literal["start", "resume"]


@dataclass(slots=True)
class PipelineTaskJob:
    """Queued pipeline execution job."""

    kind: JobKind
    project_id: str
    run_id: str


class PipelineQueueFullError(RuntimeError):
    """Raised when a pipeline queue is full."""


class PipelineTaskManager:
    """Redis-backed enqueue/dequeue manager for one pipeline module."""

    def __init__(
        self,
        *,
        pipeline_module: PipelineModule,
        worker_count: int,
        queue_size: int,
    ) -> None:
        self.pipeline_module = pipeline_module
        self._worker_count = max(1, worker_count)
        self._queue_size_limit = max(1, queue_size)
        self._redis = get_redis_client()

    @property
    def worker_count(self) -> int:
        """Configured worker count for this module."""
        return self._worker_count

    @property
    def queue_size_limit(self) -> int:
        """Configured queue size cap for this module."""
        return self._queue_size_limit

    async def get_queue_size(self) -> int:
        """Get current queue length from Redis."""
        return int(await self._redis.llen(self._queue_key()))

    async def start(self) -> None:
        """Compatibility no-op for producer-only API process."""
        return

    async def stop(self) -> None:
        """Compatibility no-op for producer-only API process."""
        return

    async def enqueue_start(
        self,
        *,
        project_id: str,
        run_id: str,
    ) -> None:
        """Queue a start pipeline job."""
        await self.start()
        await self._enqueue(
            PipelineTaskJob(
                kind="start",
                project_id=project_id,
                run_id=run_id,
            ),
            enforce_capacity=True,
        )

    async def enqueue_resume(self, *, project_id: str, run_id: str) -> None:
        """Queue a resume pipeline job."""
        await self.start()
        await self._enqueue(
            PipelineTaskJob(
                kind="resume",
                project_id=project_id,
                run_id=run_id,
            ),
            enforce_capacity=True,
        )

    async def requeue(self, job: PipelineTaskJob) -> None:
        """Requeue a job without hard-failing on configured queue caps."""
        await self._enqueue(job, enforce_capacity=False)

    async def _enqueue(
        self,
        job: PipelineTaskJob,
        *,
        enforce_capacity: bool,
    ) -> None:
        if enforce_capacity:
            pending = int(await self._redis.llen(self._queue_key()))
            if pending >= self._queue_size_limit:
                raise PipelineQueueFullError("Pipeline task queue is full")
        payload = self._serialize_job(job)
        queue_size = int(await self._redis.rpush(self._queue_key(), payload))
        logger.info(
            "Pipeline task queued",
            extra={
                "pipeline_module": self.pipeline_module,
                "kind": job.kind,
                "project_id": job.project_id,
                "run_id": job.run_id,
                "queue_size": queue_size,
            },
        )

    async def pop_next(self, *, timeout_seconds: int = 5) -> PipelineTaskJob | None:
        """Pop the next queued job for this module."""
        timeout = max(1, int(timeout_seconds))
        popped = await self._redis.blpop(self._queue_key(), timeout=timeout)
        if not popped:
            return None
        _, payload = popped
        try:
            return self._deserialize_job(payload)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(
                "Dropping malformed pipeline job payload",
                extra={"pipeline_module": self.pipeline_module, "payload": payload},
            )
            return None

    def _queue_key(self) -> str:
        return f"pipeline:queue:{self.pipeline_module}"

    @staticmethod
    def _serialize_job(job: PipelineTaskJob) -> str:
        return json.dumps(
            {"kind": job.kind, "project_id": job.project_id, "run_id": job.run_id},
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize_job(payload: str) -> PipelineTaskJob:
        data = json.loads(payload)
        kind = data["kind"]
        project_id = data["project_id"]
        run_id = data["run_id"]
        if kind not in {"start", "resume"}:
            raise ValueError("Invalid pipeline job kind")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("Invalid project_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Invalid run_id")
        return PipelineTaskJob(kind=kind, project_id=project_id, run_id=run_id)


class PipelineTaskWorker:
    """Async worker pool consuming Redis jobs for a pipeline module."""

    def __init__(
        self,
        *,
        manager: PipelineTaskManager,
        poll_timeout_seconds: int = 5,
        requeue_delay_seconds: float = 1.0,
    ) -> None:
        self.manager = manager
        self.poll_timeout_seconds = max(1, int(poll_timeout_seconds))
        self.requeue_delay_seconds = max(0.1, float(requeue_delay_seconds))
        self._workers: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        """Start worker tasks if not already running."""
        if self._workers:
            return
        self._stopping.clear()
        self._workers = [
            asyncio.create_task(
                self._worker_loop(index + 1),
                name=f"{self.manager.pipeline_module}-pipeline-worker-{index + 1}",
            )
            for index in range(self.manager.worker_count)
        ]
        logger.info(
            "Pipeline worker pool started",
            extra={
                "pipeline_module": self.manager.pipeline_module,
                "worker_count": self.manager.worker_count,
            },
        )

    async def stop(self) -> None:
        """Stop worker tasks."""
        if not self._workers:
            return
        self._stopping.set()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        logger.info(
            "Pipeline worker pool stopped",
            extra={"pipeline_module": self.manager.pipeline_module},
        )

    async def _worker_loop(self, worker_index: int) -> None:
        logger.info(
            "Pipeline worker started",
            extra={
                "pipeline_module": self.manager.pipeline_module,
                "worker_index": worker_index,
            },
        )
        try:
            while not self._stopping.is_set():
                job = await self.manager.pop_next(timeout_seconds=self.poll_timeout_seconds)
                if job is None:
                    continue
                try:
                    await self._execute(job)
                except PipelineAlreadyRunningError:
                    logger.info(
                        "Pipeline module busy, requeueing job",
                        extra={
                            "pipeline_module": self.manager.pipeline_module,
                            "project_id": job.project_id,
                            "run_id": job.run_id,
                            "worker_index": worker_index,
                        },
                    )
                    await asyncio.sleep(self.requeue_delay_seconds)
                    await self.manager.requeue(job)
                except Exception:
                    logger.exception(
                        "Pipeline worker job failed",
                        extra={
                            "pipeline_module": self.manager.pipeline_module,
                            "worker_index": worker_index,
                            "kind": job.kind,
                            "project_id": job.project_id,
                            "run_id": job.run_id,
                        },
                    )
        except asyncio.CancelledError:
            pass
        finally:
            logger.info(
                "Pipeline worker stopped",
                extra={
                    "pipeline_module": self.manager.pipeline_module,
                    "worker_index": worker_index,
                },
            )

    async def _execute(self, job: PipelineTaskJob) -> None:
        async with get_session_context() as session:
            # Local import avoids a circular dependency at module import time.
            from app.services.pipeline_orchestrator import PipelineOrchestrator

            orchestrator = PipelineOrchestrator(session, job.project_id)
            if job.kind == "start":
                await orchestrator.start_pipeline(
                    run_id=job.run_id,
                    pipeline_module=self.manager.pipeline_module,
                )
                return
            await orchestrator.resume_pipeline(
                job.run_id,
                pipeline_module=self.manager.pipeline_module,
            )


_discovery_pipeline_task_manager: PipelineTaskManager | None = None
_content_pipeline_task_manager: PipelineTaskManager | None = None
_setup_pipeline_task_manager: PipelineTaskManager | None = None


def get_discovery_pipeline_task_manager() -> PipelineTaskManager:
    """Get singleton discovery task manager."""
    global _discovery_pipeline_task_manager
    if _discovery_pipeline_task_manager is None:
        _discovery_pipeline_task_manager = PipelineTaskManager(
            pipeline_module="discovery",
            worker_count=settings.discovery_pipeline_task_workers,
            queue_size=settings.discovery_pipeline_task_queue_size,
        )
    return _discovery_pipeline_task_manager


def get_setup_pipeline_task_manager() -> PipelineTaskManager:
    """Get singleton setup task manager."""
    global _setup_pipeline_task_manager
    if _setup_pipeline_task_manager is None:
        _setup_pipeline_task_manager = PipelineTaskManager(
            pipeline_module="setup",
            worker_count=settings.setup_pipeline_task_workers,
            queue_size=settings.setup_pipeline_task_queue_size,
        )
    return _setup_pipeline_task_manager


def get_content_pipeline_task_manager() -> PipelineTaskManager:
    """Get singleton content task manager."""
    global _content_pipeline_task_manager
    if _content_pipeline_task_manager is None:
        _content_pipeline_task_manager = PipelineTaskManager(
            pipeline_module="content",
            worker_count=settings.content_pipeline_task_workers,
            queue_size=settings.content_pipeline_task_queue_size,
        )
    return _content_pipeline_task_manager


def get_pipeline_task_manager(pipeline_module: PipelineModule) -> PipelineTaskManager:
    """Get singleton task manager by module."""
    if pipeline_module == "setup":
        return get_setup_pipeline_task_manager()
    if pipeline_module == "discovery":
        return get_discovery_pipeline_task_manager()
    return get_content_pipeline_task_manager()
