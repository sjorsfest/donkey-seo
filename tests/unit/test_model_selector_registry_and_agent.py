"""Tests for selector registry and BaseAgent integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.services.model_selector.registry import get_agent_model, reset_snapshot_cache


class DummyInput(BaseModel):
    """Minimal input model for BaseAgent tests."""

    text: str


class DummyOutput(BaseModel):
    """Minimal output model for BaseAgent tests."""

    value: str


class DummyAgent(BaseAgent[DummyInput, DummyOutput]):
    """Test agent used to validate model resolution behavior."""

    @property
    def system_prompt(self) -> str:
        return "test"

    @property
    def output_type(self) -> type[DummyOutput]:
        return DummyOutput

    def _build_prompt(self, input_data: DummyInput) -> str:
        return input_data.text


class _FakeUsage:
    request_tokens = 1
    response_tokens = 1
    total_tokens = 2


class _FakeResult:
    def __init__(self, output: DummyOutput) -> None:
        self.output = output

    def usage(self) -> _FakeUsage:
        return _FakeUsage()


def write_snapshot(path: Path, model: str) -> None:
    """Write a minimal selector snapshot for one agent."""
    payload = {
        "version": "1",
        "generated_at": "2026-02-16T00:00:00+00:00",
        "environments": {
            "development": {
                "max_price": 0.0,
                "agents": {
                    "DummyAgent": {
                        "model": model,
                        "max_price": 0.0,
                        "score_breakdown": {},
                        "source_metadata": {},
                        "fallback_used": False,
                    }
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_registry_get_agent_model_from_snapshot(tmp_path: Path) -> None:
    """Registry returns model from snapshot file."""
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot(snapshot_path, "openrouter:test/model")

    reset_snapshot_cache()
    model = get_agent_model("development", "DummyAgent")
    assert model is None

    original_path = settings.model_selector_snapshot_path
    try:
        settings.model_selector_snapshot_path = str(snapshot_path)
        model = get_agent_model("development", "DummyAgent")
        assert model == "openrouter:test/model"
    finally:
        settings.model_selector_snapshot_path = original_path
        reset_snapshot_cache()


def test_base_agent_uses_model_selector_when_enabled(tmp_path: Path) -> None:
    """BaseAgent resolves model from selector snapshot before tier fallback."""
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot(snapshot_path, "openrouter:selected/model")

    original_enabled = settings.model_selector_enabled
    original_env = settings.environment
    original_snapshot = settings.model_selector_snapshot_path

    try:
        settings.model_selector_enabled = True
        settings.environment = "development"
        settings.model_selector_snapshot_path = str(snapshot_path)

        reset_snapshot_cache()
        agent = DummyAgent()
        assert agent._model == "openrouter:selected/model"
    finally:
        settings.model_selector_enabled = original_enabled
        settings.environment = original_env
        settings.model_selector_snapshot_path = original_snapshot
        reset_snapshot_cache()


def test_base_agent_falls_back_to_tier_model_when_snapshot_missing(tmp_path: Path) -> None:
    """Tier defaults remain fallback when snapshot is missing or unreadable."""
    missing_snapshot_path = tmp_path / "missing.json"

    original_enabled = settings.model_selector_enabled
    original_env = settings.environment
    original_snapshot = settings.model_selector_snapshot_path
    original_dev_standard = settings.dev_model_standard

    try:
        settings.model_selector_enabled = True
        settings.environment = "development"
        settings.model_selector_snapshot_path = str(missing_snapshot_path)
        settings.dev_model_standard = "tier-fallback-model"

        reset_snapshot_cache()
        agent = DummyAgent()
        assert agent._model == "tier-fallback-model"
    finally:
        settings.model_selector_enabled = original_enabled
        settings.environment = original_env
        settings.model_selector_snapshot_path = original_snapshot
        settings.dev_model_standard = original_dev_standard
        reset_snapshot_cache()


def test_base_agent_keeps_tier_resolution_when_selector_disabled(tmp_path: Path) -> None:
    """Selector disabled preserves existing tier-based behavior."""
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot(snapshot_path, "openrouter:selected/model")

    original_enabled = settings.model_selector_enabled
    original_env = settings.environment
    original_snapshot = settings.model_selector_snapshot_path
    original_dev_standard = settings.dev_model_standard

    try:
        settings.model_selector_enabled = False
        settings.environment = "development"
        settings.model_selector_snapshot_path = str(snapshot_path)
        settings.dev_model_standard = "tier-only-model"

        reset_snapshot_cache()
        agent = DummyAgent()
        assert agent._model == "tier-only-model"
    finally:
        settings.model_selector_enabled = original_enabled
        settings.environment = original_env
        settings.model_selector_snapshot_path = original_snapshot
        settings.dev_model_standard = original_dev_standard
        reset_snapshot_cache()


def test_base_agent_falls_back_when_snapshot_is_invalid_json(tmp_path: Path) -> None:
    """Invalid snapshot payload falls back to tier model instead of crashing."""
    invalid_snapshot_path = tmp_path / "invalid_snapshot.json"
    invalid_snapshot_path.write_text("{invalid-json", encoding="utf-8")

    original_enabled = settings.model_selector_enabled
    original_env = settings.environment
    original_snapshot = settings.model_selector_snapshot_path
    original_dev_standard = settings.dev_model_standard

    try:
        settings.model_selector_enabled = True
        settings.environment = "development"
        settings.model_selector_snapshot_path = str(invalid_snapshot_path)
        settings.dev_model_standard = "tier-invalid-fallback"

        reset_snapshot_cache()
        agent = DummyAgent()
        assert agent._model == "tier-invalid-fallback"
    finally:
        settings.model_selector_enabled = original_enabled
        settings.environment = original_env
        settings.model_selector_snapshot_path = original_snapshot
        settings.dev_model_standard = original_dev_standard
        reset_snapshot_cache()


def test_base_agent_system_prompt_includes_runtime_date_context() -> None:
    agent = DummyAgent(model_override="openrouter:test/model")
    prompt = agent._build_system_prompt()

    assert "Runtime Date Context" in prompt
    assert "Current year" in prompt
    assert "outdated years" in prompt


@pytest.mark.asyncio
async def test_base_agent_retries_with_fallback_model_on_429() -> None:
    """429 responses trigger a retry with configured fallback model."""

    class FlakyAgent:
        def __init__(self) -> None:
            self.models_seen: list[str | None] = []

        async def run(self, prompt: str, **kwargs: str) -> _FakeResult:
            self.models_seen.append(kwargs.get("model"))
            if len(self.models_seen) == 1:
                raise ModelHTTPError(
                    status_code=429,
                    model_name="openrouter:primary/model",
                    body={"message": "rate limited"},
                )
            return _FakeResult(DummyOutput(value=prompt))

    original_fallback = settings.model_selector_fallback_model
    try:
        settings.model_selector_fallback_model = "openrouter:fallback/model"
        agent = DummyAgent(model_override="openrouter:primary/model")
        fake_agent = FlakyAgent()
        agent._agent = fake_agent

        output = await agent.run(DummyInput(text="ok"))

        assert output.value == "ok"
        assert fake_agent.models_seen == [None, "openrouter:fallback/model"]
    finally:
        settings.model_selector_fallback_model = original_fallback


@pytest.mark.asyncio
async def test_base_agent_does_not_retry_when_fallback_matches_primary() -> None:
    """No fallback retry occurs when fallback model equals current model."""

    class Always429Agent:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, prompt: str, **kwargs: str) -> _FakeResult:
            self.calls += 1
            raise ModelHTTPError(
                status_code=429,
                model_name="openrouter:primary/model",
                body={"message": "rate limited"},
            )

    original_fallback = settings.model_selector_fallback_model
    try:
        settings.model_selector_fallback_model = "openrouter:primary/model"
        agent = DummyAgent(model_override="openrouter:primary/model")
        fake_agent = Always429Agent()
        agent._agent = fake_agent

        with pytest.raises(ModelHTTPError):
            await agent.run(DummyInput(text="ok"))

        assert fake_agent.calls == 1
    finally:
        settings.model_selector_fallback_model = original_fallback
