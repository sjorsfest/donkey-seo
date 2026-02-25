"""Unit tests for integration API key dependency behavior."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from app.api.integration.dependencies import require_integration_api_key
from app.config import settings


class _FakeExecuteResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class _FakeSession:
    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls = 0

    async def execute(self, _query: Any) -> _FakeExecuteResult:
        self.calls += 1
        if not self._results:
            raise AssertionError("unexpected execute call")
        return _FakeExecuteResult(self._results.pop(0))


@pytest.mark.asyncio
async def test_require_integration_api_key_accepts_env_key_without_db_lookup() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"
    session = _FakeSession([])

    try:
        result = await require_integration_api_key(
            api_key="valid-key",
            project_id="project_1",
            session=session,
        )
    finally:
        settings.integration_api_keys = original_keys

    assert result is None
    assert session.calls == 0


@pytest.mark.asyncio
async def test_require_integration_api_key_accepts_project_key_hash_match() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = ""
    session = _FakeSession(["project_1"])

    try:
        result = await require_integration_api_key(
            api_key="dseo_key",
            project_id="project_1",
            session=session,
        )
    finally:
        settings.integration_api_keys = original_keys

    assert result is None
    assert session.calls == 1


@pytest.mark.asyncio
async def test_require_integration_api_key_rejects_missing_when_unconfigured() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = ""
    session = _FakeSession([None])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await require_integration_api_key(
                api_key=None,
                project_id="project_1",
                session=session,
            )
    finally:
        settings.integration_api_keys = original_keys

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Integration API keys are not configured"
    assert session.calls == 1


@pytest.mark.asyncio
async def test_require_integration_api_key_rejects_invalid_key() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"
    session = _FakeSession([None])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await require_integration_api_key(
                api_key="invalid-key",
                project_id="project_1",
                session=session,
            )
    finally:
        settings.integration_api_keys = original_keys

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or missing API key"
    assert session.calls == 1
