"""Unit tests for module-local step numbering contracts."""

from __future__ import annotations

from app.services.pipeline_orchestrator import PipelineOrchestrator


class _NoopSession:
    async def execute(self, _query: object) -> object:  # pragma: no cover - not used in these tests
        raise AssertionError("execute should not be called")


def test_setup_module_contract_uses_steps_1_to_5() -> None:
    orchestrator = PipelineOrchestrator(_NoopSession(), "project-1")

    cfg = orchestrator._module_config("setup")

    assert cfg["default_start"] == 1
    assert cfg["default_end"] == 5
    assert sorted(cfg["step_names"].keys()) == [1, 2, 3, 4, 5]


def test_discovery_module_contract_uses_steps_1_to_7() -> None:
    orchestrator = PipelineOrchestrator(_NoopSession(), "project-1")

    cfg = orchestrator._module_config("discovery")

    assert cfg["default_start"] == 1
    assert cfg["default_end"] == 7
    assert sorted(cfg["step_names"].keys()) == [1, 2, 3, 4, 5, 6, 7]
    assert cfg["optional_steps"] == {7}


def test_content_module_contract_uses_steps_1_to_5() -> None:
    orchestrator = PipelineOrchestrator(_NoopSession(), "project-1")

    cfg = orchestrator._module_config("content")

    assert cfg["default_start"] == 1
    assert cfg["default_end"] == 5
    assert sorted(cfg["step_names"].keys()) == [1, 2, 3, 4, 5]
