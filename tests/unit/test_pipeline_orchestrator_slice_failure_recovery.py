"""Unit tests for queued-slice failure recovery behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.pipeline_orchestrator import PipelineOrchestrator


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _FragileRun:
    def __init__(self) -> None:
        self._id = "run-1"
        self.status = "running"
        self.start_step = 0
        self.end_step = 10
        self.skip_steps: list[int] = []
        self.steps_config: dict[str, object] = {}
        self.paused_at_step: int | None = None
        self._pipeline_module = "discovery"
        self._source_topic_id = "topic-1"
        self._explode_on_attr_access = False

    @property
    def id(self) -> str:
        if self._explode_on_attr_access:
            raise RuntimeError("id lazy-load attempted on broken session")
        return self._id

    @property
    def pipeline_module(self) -> str:
        if self._explode_on_attr_access:
            raise RuntimeError("pipeline_module lazy-load attempted on broken session")
        return self._pipeline_module

    @property
    def source_topic_id(self) -> str:
        if self._explode_on_attr_access:
            raise RuntimeError("source_topic_id lazy-load attempted on broken session")
        return self._source_topic_id


class _SessionWithRun:
    def __init__(self, run: _FragileRun) -> None:
        self._run = run
        self.rollback = AsyncMock()

    async def execute(self, _query: object) -> _ScalarResult:
        return _ScalarResult(self._run)


@pytest.mark.asyncio
async def test_slice_failure_uses_cached_run_fields_and_pauses_run() -> None:
    run = _FragileRun()
    session = _SessionWithRun(run)
    orchestrator = PipelineOrchestrator(session, "project-1")
    orchestrator._update_run_status = AsyncMock()  # type: ignore[method-assign]
    orchestrator.task_manager = SimpleNamespace(set_task_state=AsyncMock())

    async def _fail_slice(_run: _FragileRun) -> bool:
        _run.paused_at_step = 5
        _run._explode_on_attr_access = True
        raise RuntimeError("connection dropped during commit")

    orchestrator._run_discovery_slice = AsyncMock(side_effect=_fail_slice)  # type: ignore[method-assign]

    should_requeue = await orchestrator.run_queued_job_slice(
        run_id="run-1",
        pipeline_module="discovery",
        job_kind="resume",
    )

    assert should_requeue is False
    session.rollback.assert_awaited_once()
    orchestrator._update_run_status.assert_awaited_once()
    update_call = orchestrator._update_run_status.await_args
    assert update_call.args[0] == "run-1"
    assert update_call.kwargs["status"] == "paused"
    assert update_call.kwargs["paused_at_step"] == 5
    assert "connection dropped during commit" in update_call.kwargs["error_message"]

    set_state_call = orchestrator.task_manager.set_task_state.await_args
    assert set_state_call.kwargs["pipeline_module"] == "discovery"
    assert set_state_call.kwargs["source_topic_id"] == "topic-1"
    assert set_state_call.kwargs["current_step"] == 5
