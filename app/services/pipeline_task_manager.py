"""In-process task queues for discovery/content pipeline jobs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.core.exceptions import PipelineAlreadyRunningError
from app.core.database import get_session_context

logger = logging.getLogger(__name__)

PipelineModule = Literal["discovery", "content"]
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
    """Async queue + worker pool for a single pipeline module."""

    def __init__(
        self,
        *,
        pipeline_module: PipelineModule,
        worker_count: int,
        queue_size: int,
    ) -> None:
        self.pipeline_module = pipeline_module
        self._worker_count = max(1, worker_count)
        self._queue: asyncio.Queue[PipelineTaskJob | None] = asyncio.Queue(
            maxsize=max(1, queue_size)
        )
        self._workers: list[asyncio.Task[None]] = []
        self._lifecycle_lock = asyncio.Lock()

    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start worker tasks if not already running."""
        async with self._lifecycle_lock:
            if self._workers:
                return
            self._workers = [
                asyncio.create_task(
                    self._worker_loop(index + 1),
                    name=f"{self.pipeline_module}-pipeline-worker-{index + 1}",
                )
                for index in range(self._worker_count)
            ]
            logger.info(
                "Pipeline task manager started",
                extra={
                    "pipeline_module": self.pipeline_module,
                    "worker_count": self._worker_count,
                    "queue_maxsize": self._queue.maxsize,
                },
            )

    async def stop(self) -> None:
        """Stop worker tasks."""
        async with self._lifecycle_lock:
            if not self._workers:
                return
            workers = self._workers
            self._workers = []

        for _ in workers:
            await self._queue.put(None)

        results = await asyncio.gather(*workers, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "Pipeline worker exited with error",
                    extra={
                        "pipeline_module": self.pipeline_module,
                        "error": str(result),
                    },
                )

        logger.info("Pipeline task manager stopped", extra={"pipeline_module": self.pipeline_module})

    async def enqueue_start(
        self,
        *,
        project_id: str,
        run_id: str,
    ) -> None:
        """Queue a start pipeline job."""
        await self.start()
        self._enqueue(
            PipelineTaskJob(
                kind="start",
                project_id=project_id,
                run_id=run_id,
            )
        )

    async def enqueue_resume(self, *, project_id: str, run_id: str) -> None:
        """Queue a resume pipeline job."""
        await self.start()
        self._enqueue(
            PipelineTaskJob(
                kind="resume",
                project_id=project_id,
                run_id=run_id,
            )
        )

    def _enqueue(self, job: PipelineTaskJob) -> None:
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise PipelineQueueFullError("Pipeline task queue is full") from exc

        logger.info(
            "Pipeline task queued",
            extra={
                "pipeline_module": self.pipeline_module,
                "kind": job.kind,
                "project_id": job.project_id,
                "run_id": job.run_id,
                "queue_size": self.queue_size,
            },
        )

    async def _worker_loop(self, worker_index: int) -> None:
        logger.info(
            "Pipeline worker started",
            extra={"pipeline_module": self.pipeline_module, "worker_index": worker_index},
        )
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                break

            try:
                await self._execute(job)
            except PipelineAlreadyRunningError:
                logger.info(
                    "Pipeline module busy, requeueing job",
                    extra={
                        "pipeline_module": self.pipeline_module,
                        "project_id": job.project_id,
                        "run_id": job.run_id,
                    },
                )
                await asyncio.sleep(1.0)
                self._enqueue(job)
            except Exception:
                logger.exception(
                    "Pipeline worker job failed",
                    extra={
                        "pipeline_module": self.pipeline_module,
                        "worker_index": worker_index,
                        "kind": job.kind,
                        "project_id": job.project_id,
                        "run_id": job.run_id,
                    },
                )
            finally:
                self._queue.task_done()

        logger.info(
            "Pipeline worker stopped",
            extra={"pipeline_module": self.pipeline_module, "worker_index": worker_index},
        )

    async def _execute(self, job: PipelineTaskJob) -> None:
        async with get_session_context() as session:
            # Local import avoids a circular dependency at module import time.
            from app.services.pipeline_orchestrator import PipelineOrchestrator

            orchestrator = PipelineOrchestrator(session, job.project_id)
            if job.kind == "start":
                await orchestrator.start_pipeline(
                    run_id=job.run_id,
                    pipeline_module=self.pipeline_module,
                )
                return

            await orchestrator.resume_pipeline(job.run_id, pipeline_module=self.pipeline_module)


_discovery_pipeline_task_manager: PipelineTaskManager | None = None
_content_pipeline_task_manager: PipelineTaskManager | None = None


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
