"""Unit tests for DB kernel helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.db_kernel import (
    ConflictError,
    PermanentDbError,
    TransientDbError,
    db_read,
    db_write,
    db_write_no_retry,
)


class _FakeSession:
    def __init__(self) -> None:
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


@pytest.mark.asyncio
async def test_db_read_uses_short_lived_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession()

    @asynccontextmanager
    async def _fake_context(*, commit_on_exit: bool = True):
        assert commit_on_exit is False
        yield session

    monkeypatch.setattr("app.core.db_kernel.get_session_context", _fake_context)

    result = await db_read(lambda s: _echo("ok", s), operation_name="unit_read")

    assert result == "ok"
    assert session.commit_calls == 0


@pytest.mark.asyncio
async def test_db_write_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession()
    calls = {"count": 0}

    @asynccontextmanager
    async def _fake_context(*, commit_on_exit: bool = True):
        assert commit_on_exit is False
        yield session

    async def _operation(_session: _FakeSession) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("connection is closed")
        return "done"

    monkeypatch.setattr("app.core.db_kernel.get_session_context", _fake_context)

    result = await db_write(
        _operation,
        operation_name="unit_write",
        attempts=2,
        base_delay_seconds=0.0,
    )

    assert result == "done"
    assert calls["count"] == 2
    assert session.commit_calls == 1


@pytest.mark.asyncio
async def test_db_write_raises_permanent_on_non_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()

    @asynccontextmanager
    async def _fake_context(*, commit_on_exit: bool = True):
        assert commit_on_exit is False
        yield session

    monkeypatch.setattr("app.core.db_kernel.get_session_context", _fake_context)

    async def _operation(_session: _FakeSession) -> None:
        raise ValueError("bad payload")

    with pytest.raises(PermanentDbError):
        await db_write(_operation, operation_name="unit_write_perm", attempts=2, base_delay_seconds=0.0)


@pytest.mark.asyncio
async def test_db_write_no_retry_translates_integrity_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()

    @asynccontextmanager
    async def _fake_context(*, commit_on_exit: bool = True):
        assert commit_on_exit is False
        yield session

    monkeypatch.setattr("app.core.db_kernel.get_session_context", _fake_context)

    async def _operation(_session: _FakeSession) -> None:
        raise IntegrityError("INSERT ...", {}, Exception("duplicate key"))

    with pytest.raises(ConflictError):
        await db_write_no_retry(_operation, operation_name="unit_write_conflict")


@pytest.mark.asyncio
async def test_db_write_exhausted_transient_retry_raises_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()

    @asynccontextmanager
    async def _fake_context(*, commit_on_exit: bool = True):
        assert commit_on_exit is False
        yield session

    monkeypatch.setattr("app.core.db_kernel.get_session_context", _fake_context)

    async def _operation(_session: _FakeSession) -> None:
        raise RuntimeError("connection is closed")

    with pytest.raises(TransientDbError):
        await db_write(_operation, operation_name="unit_write_transient", attempts=2, base_delay_seconds=0.0)


async def _echo(value: str, _session: Any) -> str:
    return value
