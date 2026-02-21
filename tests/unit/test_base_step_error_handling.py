"""Unit tests for BaseStepService error handling robustness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.steps.base_step import BaseStepService


class _StubSession:
    def __init__(
        self,
        *,
        fail_commit_calls: set[int] | None = None,
        fail_rollback: bool = False,
    ) -> None:
        self.fail_commit_calls = fail_commit_calls or set()
        self.fail_rollback = fail_rollback
        self.commit_calls = 0
        self.rollback_calls = 0
        self.add_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1
        if self.commit_calls in self.fail_commit_calls:
            raise RuntimeError("commit failed")

    async def rollback(self) -> None:
        self.rollback_calls += 1
        if self.fail_rollback:
            raise RuntimeError("rollback failed")

    def add(self, _obj: object) -> None:
        self.add_calls += 1


class _FailingStep(BaseStepService[dict, dict]):
    step_number = 14
    step_name = "article_generation"

    async def _validate_preconditions(self, input_data: dict) -> None:
        return None

    async def _execute(self, input_data: dict) -> dict:
        raise ValueError("boom")

    async def _persist_results(self, result: dict) -> None:
        return None


def _execution_stub() -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_data=None,
        pipeline_run_id=None,
        status="pending",
        progress_percent=0.0,
        progress_message=None,
        items_processed=0,
        items_total=0,
        started_at=None,
        completed_at=None,
        error_message=None,
        error_traceback=None,
    )


@pytest.mark.asyncio
async def test_run_returns_failure_even_if_error_persist_commit_fails() -> None:
    session = _StubSession(fail_commit_calls={2})
    step = _FailingStep(
        session=session,  # type: ignore[arg-type]
        project_id="project-1",
        execution=_execution_stub(),  # type: ignore[arg-type]
    )

    result = await step.run({})

    assert result.success is False
    assert result.error == "boom"
    assert session.rollback_calls == 1
    assert session.add_calls == 1


@pytest.mark.asyncio
async def test_run_still_attempts_error_commit_when_rollback_fails() -> None:
    session = _StubSession(fail_rollback=True)
    execution = _execution_stub()
    step = _FailingStep(
        session=session,  # type: ignore[arg-type]
        project_id="project-1",
        execution=execution,  # type: ignore[arg-type]
    )

    result = await step.run({})

    assert result.success is False
    assert session.rollback_calls == 1
    assert session.commit_calls == 2
    assert execution.status == "failed"
    assert execution.error_message == "boom"
