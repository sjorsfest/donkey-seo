"""Unit tests for pipeline step completion checks."""

from __future__ import annotations

import pytest

from app.services.pipeline_orchestrator import PipelineOrchestrator


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar(self) -> object:
        return self._value

    def scalar_one_or_none(self) -> object:
        raise AssertionError("scalar_one_or_none should not be called for completion checks")


class _SessionWithScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    async def execute(self, _query: object) -> _ScalarResult:
        return _ScalarResult(self._value)


@pytest.mark.asyncio
async def test_is_step_completed_true_with_multiple_rows_tolerated() -> None:
    orchestrator = PipelineOrchestrator(_SessionWithScalarResult("step-exec-id"), "project-1")

    is_completed = await orchestrator._is_step_completed(run_id="run-1", step_number=5)

    assert is_completed is True


@pytest.mark.asyncio
async def test_is_step_completed_false_when_no_rows() -> None:
    orchestrator = PipelineOrchestrator(_SessionWithScalarResult(None), "project-1")

    is_completed = await orchestrator._is_step_completed(run_id="run-1", step_number=5)

    assert is_completed is False
