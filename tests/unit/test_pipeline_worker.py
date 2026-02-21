"""Unit tests for pipeline worker module selection."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.services.pipeline_task_manager import PipelineTaskJob, PipelineTaskWorker
from app.workers.pipeline_worker import resolve_modules


def test_resolve_modules_defaults_to_all() -> None:
    assert resolve_modules(None) == ["setup", "discovery", "content"]


def test_resolve_modules_all_wins() -> None:
    assert resolve_modules(["discovery", "all"]) == ["setup", "discovery", "content"]


def test_resolve_modules_deduplicates_preserving_order() -> None:
    assert resolve_modules(["content", "setup", "content"]) == ["content", "setup"]


class _FakeManager:
    pipeline_module = "setup"
    worker_count = 1

    def __init__(self, job: PipelineTaskJob) -> None:
        self._job = job
        self.requeued: list[PipelineTaskJob] = []
        self._popped = False

    async def pop_next(self, *, timeout_seconds: int = 5) -> PipelineTaskJob | None:
        if self._popped:
            return None
        self._popped = True
        return self._job

    async def requeue(self, job: PipelineTaskJob) -> None:
        self.requeued.append(job)


class _FlakyDequeueManager(_FakeManager):
    def __init__(self, job: PipelineTaskJob) -> None:
        super().__init__(job)
        self._raised = False

    async def pop_next(self, *, timeout_seconds: int = 5) -> PipelineTaskJob | None:
        if not self._raised:
            self._raised = True
            raise RuntimeError("temporary dequeue failure")
        return await super().pop_next(timeout_seconds=timeout_seconds)


@pytest.mark.asyncio
async def test_pipeline_task_worker_requeues_when_slice_incomplete(monkeypatch: Any) -> None:
    manager = _FakeManager(
        PipelineTaskJob(kind="start", project_id="project-1", run_id="run-1")
    )
    worker = PipelineTaskWorker(manager=manager, poll_timeout_seconds=1)

    async def _fake_execute(_job: PipelineTaskJob) -> bool:
        worker._stopping.set()
        return True

    monkeypatch.setattr(worker, "_execute", _fake_execute)

    await worker._worker_loop(worker_index=1)

    assert len(manager.requeued) == 1
    assert manager.requeued[0].run_id == "run-1"


@pytest.mark.asyncio
async def test_pipeline_task_worker_recovers_from_dequeue_error(monkeypatch: Any) -> None:
    manager = _FlakyDequeueManager(
        PipelineTaskJob(kind="start", project_id="project-1", run_id="run-1")
    )
    worker = PipelineTaskWorker(manager=manager, poll_timeout_seconds=1)
    calls = {"count": 0}

    async def _fake_execute(_job: PipelineTaskJob) -> bool:
        calls["count"] += 1
        worker._stopping.set()
        return False

    monkeypatch.setattr(worker, "_execute", _fake_execute)

    await worker._worker_loop(worker_index=1)

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_pipeline_task_worker_drops_duplicate_run_job_when_lock_busy(
    monkeypatch: Any,
) -> None:
    job = PipelineTaskJob(kind="resume", project_id="project-1", run_id="run-1")
    manager = _FakeManager(job)
    worker = PipelineTaskWorker(manager=manager, poll_timeout_seconds=1)

    busy_lock = worker._run_locks.setdefault(job.run_id, asyncio.Lock())
    await busy_lock.acquire()

    calls = {"count": 0}

    async def _pop_once_then_stop(*, timeout_seconds: int = 5) -> PipelineTaskJob | None:
        if calls["count"] == 0:
            calls["count"] += 1
            worker._stopping.set()
            return job
        return None

    manager.pop_next = _pop_once_then_stop  # type: ignore[method-assign]
    execute_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(worker, "_execute", execute_mock)

    await worker._worker_loop(worker_index=1)

    assert execute_mock.await_count == 0
    assert manager.requeued == []
