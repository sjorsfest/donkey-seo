"""Unit tests for Step 8 orchestrator wiring."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.services.pipeline_orchestrator import PipelineOrchestrator
from app.services.steps.discovery.step_08_serp import (
    SerpValidationInput,
    Step08SerpValidationService,
)


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _FakeSession:
    async def execute(self, _query: object) -> _ScalarResult:
        return _ScalarResult(SimpleNamespace(id=str(uuid.uuid4()), steps_config={}))


@pytest.mark.asyncio
async def test_get_step_service_returns_step08_service() -> None:
    orchestrator = PipelineOrchestrator(_FakeSession(), str(uuid.uuid4()))
    execution = SimpleNamespace()

    service = await orchestrator._get_step_service(
        module="discovery",
        step_number=8,
        execution=execution,
        session=_FakeSession(),
    )

    assert isinstance(service, Step08SerpValidationService)


@pytest.mark.asyncio
async def test_get_step_input_returns_step08_input() -> None:
    project_id = str(uuid.uuid4())
    session = _FakeSession()
    orchestrator = PipelineOrchestrator(session, project_id)

    input_data = await orchestrator._get_step_input(
        module="discovery",
        step_number=8,
        pipeline_run_id=str(uuid.uuid4()),
        session=session,
    )

    assert isinstance(input_data, SerpValidationInput)
    assert input_data.project_id == project_id
