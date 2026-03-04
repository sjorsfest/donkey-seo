"""Unit tests for Stripe upgrade-driven discovery resume helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.v1.billing.routes import (
    _enqueue_resume_for_halted_discovery_runs,
    _is_free_to_paid_upgrade,
)
from app.services.pipeline_task_manager import PipelineQueueFullError


class _FakeScalars:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def all(self) -> list[object]:
        return list(self._values)


class _FakeResult:
    def __init__(
        self,
        *,
        rows: list[tuple[object, object, object, object]] | None = None,
        scalars: list[object] | None = None,
    ) -> None:
        self._rows = rows or []
        self._scalars = scalars or []

    def all(self) -> list[tuple[object, object, object, object]]:
        return list(self._rows)

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._scalars)


class _FakeSession:
    def __init__(self, *, execute_results: list[_FakeResult]) -> None:
        self._execute_results = iter(execute_results)

    async def execute(self, _query: object) -> _FakeResult:
        return next(self._execute_results)


def test_is_free_to_paid_upgrade_only_true_on_free_to_paid_transition() -> None:
    assert _is_free_to_paid_upgrade(previous_plan=None, current_plan="starter") is True
    assert _is_free_to_paid_upgrade(previous_plan=None, current_plan=None) is False
    assert _is_free_to_paid_upgrade(previous_plan="starter", current_plan="growth") is False
    assert _is_free_to_paid_upgrade(previous_plan="starter", current_plan=None) is False


@pytest.mark.asyncio
async def test_enqueue_resume_uses_latest_halted_run_per_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        execute_results=[
            _FakeResult(
                rows=[
                    ("run-2", "project-1", 5, None),
                    ("run-1", "project-1", 2, None),
                    ("run-3", "project-2", 4, "topic-3"),
                ],
            ),
            _FakeResult(scalars=["project-2"]),
        ]
    )
    queue = SimpleNamespace(enqueue_resume=AsyncMock())
    task_manager = SimpleNamespace(set_task_state=AsyncMock())

    monkeypatch.setattr(
        "app.api.v1.billing.routes.get_discovery_pipeline_task_manager",
        lambda: queue,
    )
    monkeypatch.setattr(
        "app.api.v1.billing.routes.TaskManager",
        lambda: task_manager,
    )

    resumed_count = await _enqueue_resume_for_halted_discovery_runs(
        session=session,  # type: ignore[arg-type]
        user=SimpleNamespace(id="user-1"),  # type: ignore[arg-type]
    )

    assert resumed_count == 1
    queue.enqueue_resume.assert_awaited_once_with(
        project_id="project-1",
        run_id="run-2",
    )
    task_manager.set_task_state.assert_awaited_once()
    assert task_manager.set_task_state.await_args.kwargs["task_id"] == "run-2"
    assert task_manager.set_task_state.await_args.kwargs["status"] == "queued"


@pytest.mark.asyncio
async def test_enqueue_resume_handles_queue_full_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        execute_results=[
            _FakeResult(rows=[("run-1", "project-1", 3, None)]),
            _FakeResult(scalars=[]),
        ]
    )
    queue = SimpleNamespace(
        enqueue_resume=AsyncMock(side_effect=PipelineQueueFullError("full"))
    )
    task_manager = SimpleNamespace(set_task_state=AsyncMock())

    monkeypatch.setattr(
        "app.api.v1.billing.routes.get_discovery_pipeline_task_manager",
        lambda: queue,
    )
    monkeypatch.setattr(
        "app.api.v1.billing.routes.TaskManager",
        lambda: task_manager,
    )

    resumed_count = await _enqueue_resume_for_halted_discovery_runs(
        session=session,  # type: ignore[arg-type]
        user=SimpleNamespace(id="user-1"),  # type: ignore[arg-type]
    )

    assert resumed_count == 0
    assert task_manager.set_task_state.await_count == 2
    first_call = task_manager.set_task_state.await_args_list[0]
    second_call = task_manager.set_task_state.await_args_list[1]
    assert first_call.kwargs["status"] == "queued"
    assert second_call.kwargs["status"] == "paused"
