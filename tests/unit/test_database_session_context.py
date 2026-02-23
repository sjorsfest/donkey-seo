"""Unit tests for database session context finalization behavior."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.database import get_session_context, rollback_read_only_transaction


class _FakeSessionContextManager:
    def __init__(self, session: "_FakeSession") -> None:
        self._session = session

    async def __aenter__(self) -> "_FakeSession":
        return self._session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakeSessionMaker:
    def __init__(self, session: "_FakeSession") -> None:
        self._session = session

    def __call__(self) -> _FakeSessionContextManager:
        return _FakeSessionContextManager(self._session)


class _FakeSession:
    def __init__(self) -> None:
        self.new: set[object] = set()
        self.dirty: set[object] = set()
        self.deleted: set[object] = set()
        self._in_transaction = True
        self.commit_calls = 0
        self.rollback_calls = 0
        self.commit_error: Exception | None = None
        self.rollback_error: Exception | None = None

    def in_transaction(self) -> bool:
        return self._in_transaction

    async def commit(self) -> None:
        self.commit_calls += 1
        if self.commit_error is not None:
            raise self.commit_error
        self._in_transaction = False

    async def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_error is not None:
            raise self.rollback_error
        self._in_transaction = False


@pytest.mark.asyncio
async def test_get_session_context_non_committing_mode_rolls_back_clean_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()
    monkeypatch.setattr(
        "app.core.database.async_session_maker",
        _FakeSessionMaker(session),
    )

    async with get_session_context(commit_on_exit=False) as yielded:
        assert yielded is session

    assert session.commit_calls == 1
    assert session.rollback_calls == 0


@pytest.mark.asyncio
async def test_get_session_context_non_committing_mode_rejects_pending_orm_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()
    session.dirty.add(object())
    monkeypatch.setattr(
        "app.core.database.async_session_maker",
        _FakeSessionMaker(session),
    )

    with pytest.raises(RuntimeError, match="pending ORM changes"):
        async with get_session_context(commit_on_exit=False):
            pass

    assert session.commit_calls == 0
    assert session.rollback_calls == 1


@pytest.mark.asyncio
async def test_rollback_read_only_transaction_ignores_transient_closed_connection() -> None:
    session = _FakeSession()
    session.commit_error = RuntimeError("connection is closed")

    await rollback_read_only_transaction(session, context="unit_test")
