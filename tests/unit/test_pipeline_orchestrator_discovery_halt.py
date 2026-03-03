"""Unit tests for discovery auto-halt behavior inside queued slice execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.discovery_pipeline_halt import DiscoveryPipelineHaltState
from app.services.pipeline_orchestrator import PipelineOrchestrator


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _SessionWithRun:
    def __init__(self, run: object) -> None:
        self._run = run
        self.rollback = AsyncMock()

    async def execute(self, _query: object) -> _ScalarResult:
        return _ScalarResult(self._run)


@pytest.mark.asyncio
async def test_run_queued_discovery_slice_auto_halts_when_backlog_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = SimpleNamespace(
        id="run-1",
        pipeline_module="discovery",
        status="running",
        start_step=1,
        end_step=7,
        skip_steps=[],
        steps_config={},
        paused_at_step=None,
        source_topic_id=None,
    )
    session = _SessionWithRun(run)
    orchestrator = PipelineOrchestrator(session, "project-1")
    orchestrator._update_run_status = AsyncMock()  # type: ignore[method-assign]
    orchestrator.task_manager = SimpleNamespace(set_task_state=AsyncMock())
    orchestrator._run_discovery_slice = AsyncMock(return_value=True)  # type: ignore[method-assign]

    async def _fake_halt_state(**_kwargs: object) -> DiscoveryPipelineHaltState:
        return DiscoveryPipelineHaltState(
            is_paid_user=True,
            upcoming_scheduled_items=12,
            halt_threshold=10,
            window_days=60,
        )

    monkeypatch.setattr(
        "app.services.pipeline_orchestrator.resolve_discovery_pipeline_halt_state",
        _fake_halt_state,
    )

    should_requeue = await orchestrator.run_queued_job_slice(
        run_id="run-1",
        pipeline_module="discovery",
        job_kind="resume",
    )

    assert should_requeue is False
    orchestrator._run_discovery_slice.assert_not_awaited()
    update_call = orchestrator._update_run_status.await_args
    assert update_call.kwargs["status"] == "paused"
    assert update_call.kwargs["paused_at_step"] == 1
    assert "discovery_auto_halt_scheduled_content_backlog" in str(update_call.kwargs["error_message"])
