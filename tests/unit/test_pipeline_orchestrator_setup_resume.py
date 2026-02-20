"""Unit tests for setup-module resume behavior in pipeline orchestrator."""

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


class _SessionWithPausedSetupRun:
    def __init__(self, run: object) -> None:
        self._run = run

    async def execute(self, _query: object) -> _ScalarResult:
        return _ScalarResult(self._run)


@pytest.mark.asyncio
async def test_resume_setup_pipeline_keeps_step_zero_when_no_completed_steps() -> None:
    run = SimpleNamespace(
        id="run-1",
        pipeline_module="setup",
        start_step=0,
        end_step=1,
        skip_steps=[],
        steps_config={},
    )
    orchestrator = PipelineOrchestrator(_SessionWithPausedSetupRun(run), "project-1")
    orchestrator._get_last_completed_step = AsyncMock(return_value=None)  # type: ignore[method-assign]
    orchestrator.start_pipeline = AsyncMock(return_value=run)  # type: ignore[method-assign]

    await orchestrator.resume_pipeline("run-1", pipeline_module="setup")

    kwargs = orchestrator.start_pipeline.await_args.kwargs
    assert kwargs["start_step"] == 0
    assert kwargs["end_step"] == 1
    assert kwargs["pipeline_module"] == "setup"
