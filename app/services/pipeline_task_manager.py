"""In-process task queue for pipeline orchestration jobs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.core.database import get_session_context
from app.services.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

JobKind = Literal["start", "resume"]


@dataclass(slots=True)
class PipelineTaskJob:
    """Queued pipeline execution job."""

    kind: JobKind
    project_id: str
    run_id: str
    start_step: int | None = None
    end_step: int | None = None
    skip_steps: list[int] | None = None


class PipelineQueueFullError(RuntimeError):
    """Raised when the pipeline queue is full."""


class PipelineTaskManager:
    """Async queue + worker pool for pipeline jobs."""

    def __init__(self, *, worker_count: int, queue_size: int) -> None:
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
                    name=f"pipeline-worker-{index + 1}",
                )
                for index in range(self._worker_count)
            ]
            logger.info(
                "Pipeline task manager started",
                extra={
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
                logger.warning("Pipeline worker exited with error: %s", result)

        logger.info("Pipeline task manager stopped")

    async def enqueue_start(
        self,
        *,
        project_id: str,
        run_id: str,
        start_step: int,
        end_step: int,
        skip_steps: list[int],
    ) -> None:
        """Queue a start pipeline job."""
        await self.start()
        self._enqueue(
            PipelineTaskJob(
                kind="start",
                project_id=project_id,
                run_id=run_id,
                start_step=start_step,
                end_step=end_step,
                skip_steps=skip_steps,
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
                "kind": job.kind,
                "project_id": job.project_id,
                "run_id": job.run_id,
                "queue_size": self.queue_size,
            },
        )

    async def _worker_loop(self, worker_index: int) -> None:
        logger.info("Pipeline worker started", extra={"worker_index": worker_index})
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                break

            try:
                await self._execute(job)
            except Exception:
                logger.exception(
                    "Pipeline worker job failed",
                    extra={
                        "worker_index": worker_index,
                        "kind": job.kind,
                        "project_id": job.project_id,
                        "run_id": job.run_id,
                    },
                )
            finally:
                self._queue.task_done()

        logger.info("Pipeline worker stopped", extra={"worker_index": worker_index})

    async def _execute(self, job: PipelineTaskJob) -> None:
        async with get_session_context() as session:
            orchestrator = PipelineOrchestrator(session, job.project_id)
            if job.kind == "start":
                start_step = job.start_step if job.start_step is not None else 0
                end_step = job.end_step if job.end_step is not None else 14
                await orchestrator.start_pipeline(
                    run_id=job.run_id,
                    start_step=start_step,
                    end_step=end_step,
                    skip_steps=job.skip_steps or [],
                )
                return

            await orchestrator.resume_pipeline(job.run_id)


_pipeline_task_manager: PipelineTaskManager | None = None


def get_pipeline_task_manager() -> PipelineTaskManager:
    """Get singleton pipeline task manager."""
    global _pipeline_task_manager
    if _pipeline_task_manager is None:
        _pipeline_task_manager = PipelineTaskManager(
            worker_count=settings.pipeline_task_workers,
            queue_size=settings.pipeline_task_queue_size,
        )
    return _pipeline_task_manager
