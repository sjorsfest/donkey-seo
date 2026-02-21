"""Unit tests for setup brand step split behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.steps.setup.step_02_brand_core import BrandCoreInput, Step02BrandCoreService
from app.services.steps.setup.step_04_brand_assets import (
    BrandAssetsInput,
    Step04BrandAssetsService,
)
from app.services.steps.setup.step_05_brand_visual import (
    BrandVisualInput,
    Step05BrandVisualService,
)


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one(self) -> object:
        return self._value


class _SingleResultSession:
    def __init__(self, value: object) -> None:
        self._value = value

    async def execute(self, _query: object) -> _ScalarResult:
        return _ScalarResult(self._value)


class _SequenceSession:
    def __init__(self, values: list[object]) -> None:
        self._values = values
        self._idx = 0

    async def execute(self, _query: object) -> _ScalarResult:
        value = self._values[self._idx]
        self._idx += 1
        return _ScalarResult(value)


@pytest.mark.asyncio
async def test_step02_brand_core_truncates_asset_candidates_and_stores_state(monkeypatch: pytest.MonkeyPatch) -> None:
    project = SimpleNamespace(domain="example.com")
    session = _SingleResultSession(project)

    class _Agent:
        async def run(self, _input: object) -> object:
            return SimpleNamespace(
                company_name="Acme",
                tagline="Automation for support",
                products_services=[
                    SimpleNamespace(
                        name="Acme Flow",
                        description="Automates support handoffs",
                        category="automation",
                        target_audience="support",
                        core_benefits=["faster response"],
                    )
                ],
                money_pages=["https://example.com/pricing"],
                unique_value_props=["fast setup"],
                differentiators=["workflow-native"],
                target_audience=SimpleNamespace(
                    target_roles=["Support Lead"],
                    target_industries=["SaaS"],
                    company_sizes=["SMB"],
                    primary_pains=["ticket backlog"],
                    desired_outcomes=["faster response times"],
                    common_objections=["migration effort"],
                ),
                tone_attributes=["pragmatic"],
                allowed_claims=["reduces manual work"],
                restricted_claims=["guaranteed ROI"],
                in_scope_topics=["support automation"],
                out_of_scope_topics=["medical advice"],
                extraction_confidence=0.83,
            )

    monkeypatch.setattr(
        "app.services.steps.setup.step_02_brand_core.BrandExtractorAgent",
        lambda: _Agent(),
    )
    monkeypatch.setattr(
        "app.services.steps.setup.step_02_brand_core.scrape_website",
        AsyncMock(
            return_value={
                "combined_content": "content",
                "source_urls": ["https://example.com", "https://example.com/pricing"],
                "asset_candidates": [
                    {
                        "url": f"https://example.com/assets/{idx}.png",
                        "role": "logo",
                        "role_confidence": 0.9,
                        "origin": "img_logo_signal",
                    }
                    for idx in range(100)
                ],
            }
        ),
    )

    service = Step02BrandCoreService.__new__(Step02BrandCoreService)
    service.session = session
    service.project_id = "project-1"
    service.execution = SimpleNamespace(result_summary=None)
    service._update_progress = AsyncMock()
    service.update_steps_config = AsyncMock()

    output = await service._execute(BrandCoreInput(project_id="project-1"))

    assert output.company_name == "Acme"
    assert len(output.asset_candidates) == 80
    service.update_steps_config.assert_awaited_once()
    payload = service.update_steps_config.await_args.args[0]
    assert len(payload["setup_state"]["asset_candidates"]) == 80


@pytest.mark.asyncio
async def test_step04_brand_assets_filters_low_quality_icons(monkeypatch: pytest.MonkeyPatch) -> None:
    project = SimpleNamespace(domain="example.com")
    brand = SimpleNamespace(brand_assets=[])
    session = _SequenceSession([project, brand])

    ingested_payload: dict[str, object] = {}

    class _Store:
        async def ingest_asset_candidates(self, **kwargs: object) -> list[dict[str, object]]:
            ingested_payload.update(kwargs)
            return [{"asset_id": "a1", "role": "logo"}]

    monkeypatch.setattr(
        "app.services.steps.setup.step_04_brand_assets.BrandAssetStore",
        lambda: _Store(),
    )

    service = Step04BrandAssetsService.__new__(Step04BrandAssetsService)
    service.session = session
    service.project_id = "project-1"
    service.execution = SimpleNamespace(result_summary=None)
    service._update_progress = AsyncMock()
    service.get_steps_config = AsyncMock(
        return_value={
            "setup_state": {
                "brand_source_pages": ["https://example.com"],
                "asset_candidates": [
                    {
                        "url": "https://example.com/favicon.ico",
                        "role": "icon",
                        "origin": "link_icon",
                        "role_confidence": 0.9,
                    },
                    {
                        "url": "https://example.com/logo.svg",
                        "role": "logo",
                        "origin": "img_logo_signal",
                        "role_confidence": 0.95,
                    },
                ],
            }
        }
    )
    service.update_steps_config = AsyncMock()

    output = await service._execute(BrandAssetsInput(project_id="project-1"))

    assert output.skipped_assets == 1
    assert output.brand_assets == [{"asset_id": "a1", "role": "logo"}]
    assert isinstance(ingested_payload.get("asset_candidates"), list)
    assert len(ingested_payload["asset_candidates"]) == 1
    service.update_steps_config.assert_awaited_once()


@pytest.mark.asyncio
async def test_step05_brand_visual_fallback_when_agent_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    brand = SimpleNamespace(
        company_name="Acme",
        tagline="Automation for support",
        tone_attributes=["pragmatic"],
        unique_value_props=["fast setup"],
        differentiators=["workflow-native"],
        target_roles=["Support Lead"],
        target_industries=["SaaS"],
        brand_assets=[{"asset_id": "a1", "role": "logo"}],
        extraction_confidence=0.1,
    )
    session = _SingleResultSession(brand)

    class _FailingAgent:
        async def run(self, _input: object) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.services.steps.setup.step_05_brand_visual.BrandVisualGuideAgent",
        lambda: _FailingAgent(),
    )

    service = Step05BrandVisualService.__new__(Step05BrandVisualService)
    service.session = session
    service.project_id = "project-1"
    service.execution = SimpleNamespace(result_summary=None)
    service._update_progress = AsyncMock()

    output = await service._execute(BrandVisualInput(project_id="project-1"))

    assert output.visual_extraction_confidence == 0.2
    assert "{article_topic}" in output.visual_prompt_contract["template"]
    assert "required_variables" in output.visual_prompt_contract
