"""Unit tests for pipeline worker module selection."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, call

import pytest

from app.core.exceptions import PipelineAlreadyRunningError, PipelineDelayedResumeRequested
from app.services.pipeline_task_manager import PipelineTaskJob, PipelineTaskWorker
from app.workers.pipeline_worker import (
    _recover_stale_pipeline_runs_on_startup,
    _run_discovery_auto_halt_reconciliation_loop,
    resolve_modules,
)


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


class _RowsResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[object, ...]]:
        return self._rows


class _RowsSession:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    async def execute(self, _query: object) -> _RowsResult:
        return _RowsResult(self._rows)


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


@pytest.mark.asyncio
async def test_pipeline_task_worker_requeues_resume_after_delayed_resume_request(
    monkeypatch: Any,
) -> None:
    manager = _FakeManager(
        PipelineTaskJob(kind="start", project_id="project-1", run_id="run-1")
    )
    worker = PipelineTaskWorker(manager=manager, poll_timeout_seconds=1)

    async def _fake_execute(_job: PipelineTaskJob) -> bool:
        worker._stopping.set()
        raise PipelineDelayedResumeRequested(
            delay_seconds=120,
            reason="discovery_max_iterations_exhausted",
        )

    sleep_mock = AsyncMock()
    monkeypatch.setattr(worker, "_execute", _fake_execute)
    monkeypatch.setattr("app.services.pipeline_task_manager.asyncio.sleep", sleep_mock)

    await worker._worker_loop(worker_index=1)

    assert len(manager.requeued) == 1
    assert manager.requeued[0].run_id == "run-1"
    assert manager.requeued[0].kind == "resume"
    sleep_mock.assert_awaited()


@pytest.mark.asyncio
async def test_pipeline_task_worker_requeues_when_project_lock_busy(
    monkeypatch: Any,
) -> None:
    job = PipelineTaskJob(kind="start", project_id="project-1", run_id="run-1")
    manager = _FakeManager(job)
    worker = PipelineTaskWorker(manager=manager, poll_timeout_seconds=1)

    busy_lock = worker._project_locks.setdefault(job.project_id, asyncio.Lock())
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
    sleep_mock = AsyncMock()
    monkeypatch.setattr(worker, "_execute", execute_mock)
    monkeypatch.setattr("app.services.pipeline_task_manager.asyncio.sleep", sleep_mock)

    await worker._worker_loop(worker_index=1)

    assert execute_mock.await_count == 0
    assert len(manager.requeued) == 1
    assert manager.requeued[0].run_id == "run-1"
    sleep_mock.assert_awaited()


@pytest.mark.asyncio
async def test_pipeline_task_worker_uses_backoff_when_project_lock_stays_busy(
    monkeypatch: Any,
) -> None:
    job = PipelineTaskJob(kind="start", project_id="project-1", run_id="run-1")
    manager = _FakeManager(job)
    worker = PipelineTaskWorker(manager=manager, poll_timeout_seconds=1)

    busy_lock = worker._project_locks.setdefault(job.project_id, asyncio.Lock())
    await busy_lock.acquire()

    calls = {"count": 0}

    async def _pop_twice_then_stop(*, timeout_seconds: int = 5) -> PipelineTaskJob | None:
        calls["count"] += 1
        if calls["count"] <= 2:
            return job
        worker._stopping.set()
        return None

    manager.pop_next = _pop_twice_then_stop  # type: ignore[method-assign]
    execute_mock = AsyncMock(return_value=False)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(worker, "_execute", execute_mock)
    monkeypatch.setattr("app.services.pipeline_task_manager.asyncio.sleep", sleep_mock)

    await worker._worker_loop(worker_index=1)

    assert execute_mock.await_count == 0
    assert len(manager.requeued) == 2
    assert manager.requeued[0].run_id == "run-1"
    assert manager.requeued[1].run_id == "run-1"
    assert sleep_mock.await_args_list == [call(1.0), call(2.0)]


@pytest.mark.asyncio
async def test_pipeline_task_worker_auto_pauses_blocked_run_after_busy_limit(
    monkeypatch: Any,
) -> None:
    job = PipelineTaskJob(kind="resume", project_id="project-1", run_id="run-1")
    manager = _FakeManager(job)
    worker = PipelineTaskWorker(
        manager=manager,
        poll_timeout_seconds=1,
        requeue_delay_seconds=0.1,
        max_requeue_delay_seconds=0.1,
        max_module_busy_requeues=1,
    )

    calls = {"count": 0}

    async def _pop_once_then_stop(*, timeout_seconds: int = 5) -> PipelineTaskJob | None:
        if calls["count"] == 0:
            calls["count"] += 1
            worker._stopping.set()
            return job
        return None

    async def _busy_execute(_job: PipelineTaskJob) -> bool:
        raise PipelineAlreadyRunningError(
            "project-1",
            blocking_run_id="blocking-run-1",
        )

    manager.pop_next = _pop_once_then_stop  # type: ignore[method-assign]
    pause_mock = AsyncMock()
    sleep_mock = AsyncMock()
    monkeypatch.setattr(worker, "_execute", _busy_execute)
    monkeypatch.setattr(worker, "_pause_blocked_run", pause_mock)
    monkeypatch.setattr("app.services.pipeline_task_manager.asyncio.sleep", sleep_mock)

    await worker._worker_loop(worker_index=1)

    pause_mock.assert_awaited_once()
    sleep_mock.assert_not_awaited()
    assert manager.requeued == []


@pytest.mark.asyncio
async def test_discovery_auto_halt_reconciliation_loop_runs_sweep_once(
    monkeypatch: Any,
) -> None:
    stop_event = asyncio.Event()
    calls = {"count": 0}

    async def _fake_reconcile() -> int:
        calls["count"] += 1
        stop_event.set()
        return 1

    monkeypatch.setattr(
        "app.workers.pipeline_worker.reconcile_discovery_auto_halted_runs",
        _fake_reconcile,
    )
    metrics_mock = AsyncMock()
    monkeypatch.setattr(
        "app.workers.pipeline_worker.write_discovery_reconciliation_metrics",
        metrics_mock,
    )

    await _run_discovery_auto_halt_reconciliation_loop(stop_event)
    assert calls["count"] == 1
    metrics_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_recover_stale_runs_enqueues_missing_pending_and_running_jobs(
    monkeypatch: Any,
) -> None:
    rows = [
        ("run-pending", "project-1", "content", "pending", None, None),
        ("run-running", "project-1", "content", "running", 5, "topic-1"),
    ]
    session = _RowsSession(rows)

    async def _fake_lrange(_self: object, _key: str, _start: int, _end: int) -> list[str]:
        return []

    class _FakeRedis:
        lrange = _fake_lrange

    queue = type(
        "_Queue",
        (),
        {
            "enqueue_start": AsyncMock(),
            "enqueue_resume": AsyncMock(),
        },
    )()
    task_manager = type("_TaskManager", (), {"set_task_state": AsyncMock()})()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session_context(*, commit_on_exit: bool = True):
        _ = commit_on_exit
        yield session

    monkeypatch.setattr("app.workers.pipeline_worker.get_session_context", _fake_session_context)
    monkeypatch.setattr("app.workers.pipeline_worker.get_redis_client", lambda: _FakeRedis())
    monkeypatch.setattr("app.workers.pipeline_worker.get_pipeline_task_manager", lambda _module: queue)
    monkeypatch.setattr("app.workers.pipeline_worker.TaskManager", lambda: task_manager)

    recovered = await _recover_stale_pipeline_runs_on_startup(modules=["content"])

    assert recovered == 2
    queue.enqueue_start.assert_awaited_once_with(
        project_id="project-1",
        run_id="run-pending",
    )
    queue.enqueue_resume.assert_awaited_once_with(
        project_id="project-1",
        run_id="run-running",
    )
    assert task_manager.set_task_state.await_count == 2


@pytest.mark.asyncio
async def test_recover_stale_runs_skips_runs_already_present_in_queue(
    monkeypatch: Any,
) -> None:
    rows = [("run-queued", "project-1", "content", "pending", None, None)]
    session = _RowsSession(rows)

    async def _fake_lrange(_self: object, _key: str, _start: int, _end: int) -> list[str]:
        return ['{"kind":"start","project_id":"project-1","run_id":"run-queued"}']

    class _FakeRedis:
        lrange = _fake_lrange

    queue = type(
        "_Queue",
        (),
        {
            "enqueue_start": AsyncMock(),
            "enqueue_resume": AsyncMock(),
        },
    )()
    task_manager = type("_TaskManager", (), {"set_task_state": AsyncMock()})()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session_context(*, commit_on_exit: bool = True):
        _ = commit_on_exit
        yield session

    monkeypatch.setattr("app.workers.pipeline_worker.get_session_context", _fake_session_context)
    monkeypatch.setattr("app.workers.pipeline_worker.get_redis_client", lambda: _FakeRedis())
    monkeypatch.setattr("app.workers.pipeline_worker.get_pipeline_task_manager", lambda _module: queue)
    monkeypatch.setattr("app.workers.pipeline_worker.TaskManager", lambda: task_manager)

    recovered = await _recover_stale_pipeline_runs_on_startup(modules=["content"])

    assert recovered == 0
    queue.enqueue_start.assert_not_awaited()
    queue.enqueue_resume.assert_not_awaited()
    task_manager.set_task_state.assert_not_awaited()
