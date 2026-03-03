"""Unit tests for discovery auto-halt and daily resume reconciliation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.services.discovery_pipeline_halt import (
    DiscoveryPipelineHaltState,
    build_discovery_auto_halt_reason,
    is_discovery_auto_halt_reason,
    reconcile_discovery_auto_halted_runs,
    resolve_discovery_pipeline_halt_state,
)


class _FakeResult:
    def __init__(
        self,
        *,
        rows: list[tuple[object, object]] | None = None,
        one: tuple[object, object] | None = None,
        scalar: object | None = None,
    ) -> None:
        self._rows = rows or []
        self._one = one
        self._scalar = scalar

    def all(self) -> list[tuple[object, object]]:
        return self._rows

    def one_or_none(self) -> tuple[object, object] | None:
        return self._one

    def scalar_one_or_none(self) -> object | None:
        return self._scalar


class _FakeSession:
    def __init__(
        self,
        *,
        execute_results: list[_FakeResult],
        scalar_values: list[int] | None = None,
    ) -> None:
        self._execute_results = iter(execute_results)
        self._scalar_values = iter(scalar_values or [])

    async def execute(self, _query: object) -> _FakeResult:
        return next(self._execute_results)

    async def scalar(self, _query: object) -> int:
        return next(self._scalar_values)


def test_discovery_auto_halt_reason_roundtrip() -> None:
    state = DiscoveryPipelineHaltState(
        is_paid_user=True,
        upcoming_scheduled_items=12,
        halt_threshold=10,
        window_days=60,
    )
    reason = build_discovery_auto_halt_reason(state)

    assert is_discovery_auto_halt_reason(reason)
    assert not is_discovery_auto_halt_reason("something_else")


@pytest.mark.asyncio
async def test_resolve_discovery_halt_state_paid_user_counts_upcoming_items() -> None:
    session = _FakeSession(
        execute_results=[_FakeResult(one=("project-1", "growth"))],
        scalar_values=[12],
    )

    state = await resolve_discovery_pipeline_halt_state(
        session=session,  # type: ignore[arg-type]
        project_id="project-1",
    )

    assert state.is_paid_user is True
    assert state.upcoming_scheduled_items == 12
    assert state.should_halt is True
    assert state.should_resume is False


@pytest.mark.asyncio
async def test_resolve_discovery_halt_state_free_user_skips_schedule_count() -> None:
    session = _FakeSession(
        execute_results=[_FakeResult(one=("project-1", None))],
        scalar_values=[],
    )

    state = await resolve_discovery_pipeline_halt_state(
        session=session,  # type: ignore[arg-type]
        project_id="project-1",
    )

    assert state.is_paid_user is False
    assert state.upcoming_scheduled_items == 0
    assert state.should_halt is False
    assert state.should_resume is False


@pytest.mark.asyncio
async def test_reconcile_discovery_auto_halted_runs_enqueues_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        execute_results=[
            _FakeResult(rows=[("run-1", "project-1"), ("run-2", "project-2")]),
            _FakeResult(scalar=None),
        ],
    )

    @asynccontextmanager
    async def _fake_session_context(*, commit_on_exit: bool = True):
        _ = commit_on_exit
        yield session

    async def _fake_halt_state(*, session: object, project_id: str, as_of_date: object = None):
        _ = session, as_of_date
        if project_id == "project-1":
            return DiscoveryPipelineHaltState(
                is_paid_user=True,
                upcoming_scheduled_items=9,
                halt_threshold=10,
                window_days=60,
            )
        return DiscoveryPipelineHaltState(
            is_paid_user=True,
            upcoming_scheduled_items=12,
            halt_threshold=10,
            window_days=60,
        )

    queue = type("_Queue", (), {"enqueue_resume": AsyncMock()})()

    monkeypatch.setattr(
        "app.services.discovery_pipeline_halt.get_session_context",
        _fake_session_context,
    )
    monkeypatch.setattr(
        "app.services.discovery_pipeline_halt.resolve_discovery_pipeline_halt_state",
        _fake_halt_state,
    )
    monkeypatch.setattr(
        "app.services.discovery_pipeline_halt.get_discovery_pipeline_task_manager",
        lambda: queue,
    )

    resumed_count = await reconcile_discovery_auto_halted_runs()

    assert resumed_count == 1
    queue.enqueue_resume.assert_awaited_once_with(
        project_id="project-1",
        run_id="run-1",
    )
