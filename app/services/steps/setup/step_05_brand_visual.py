"""Step 5: Generate visual style guide and prompt contract."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.agents.brand_visual_guide import BrandVisualGuideAgent, BrandVisualGuideInput
from app.models.brand import BrandProfile
from app.models.generated_dtos import BrandProfilePatchDTO
from app.models.project import Project
from app.services.steps.base_step import BaseStepService
from app.services.steps.setup.brand_shared import (
    default_visual_prompt_contract,
    default_visual_style_guide,
    fallback_visual_confidence,
    normalize_prompt_contract,
)

logger = logging.getLogger(__name__)


@dataclass
class BrandVisualInput:
    """Input for setup step 5."""

    project_id: str


@dataclass
class BrandVisualOutput:
    """Output for setup step 5."""

    visual_style_guide: dict[str, Any]
    visual_prompt_contract: dict[str, Any]
    visual_extraction_confidence: float
    visual_last_synced_at: datetime


class Step05BrandVisualService(BaseStepService[BrandVisualInput, BrandVisualOutput]):
    """Step 5: Produce reusable visual brand context artifacts."""

    step_number = 5
    step_name = "brand_visual"
    is_optional = False

    async def _validate_preconditions(self, input_data: BrandVisualInput) -> None:
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = result.scalar_one_or_none()
        if not brand:
            raise ValueError("Brand profile not found. Run setup steps 2-4 first.")

    async def _execute(self, input_data: BrandVisualInput) -> BrandVisualOutput:
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one()
        try:
            steps_config = await self.get_steps_config()
        except Exception:
            logger.warning(
                "Failed to load setup_state for visual guide; continuing without visual signals",
                extra={"project_id": input_data.project_id},
            )
            steps_config = {}
        setup_state_raw = steps_config.get("setup_state")
        setup_state = setup_state_raw if isinstance(setup_state_raw, dict) else {}
        homepage_visual_signals_raw = setup_state.get("homepage_visual_signals")
        site_visual_signals_raw = setup_state.get("site_visual_signals")
        homepage_visual_signals = (
            homepage_visual_signals_raw
            if isinstance(homepage_visual_signals_raw, dict)
            else {}
        )
        site_visual_signals = (
            site_visual_signals_raw if isinstance(site_visual_signals_raw, dict) else {}
        )
        brand_assets = [
            item
            for item in list(brand.brand_assets or [])
            if isinstance(item, dict)
        ]

        await self._update_progress(20, "Generating visual prompt style guide...")

        visual_style_guide = default_visual_style_guide(
            tone_attributes=[str(item) for item in list(brand.tone_attributes or [])],
            differentiators=[str(item) for item in list(brand.differentiators or [])],
            homepage_visual_signals=homepage_visual_signals,
            site_visual_signals=site_visual_signals,
            brand_assets=brand_assets,
        )
        visual_prompt_contract = default_visual_prompt_contract()
        extraction_confidence = float(brand.extraction_confidence or 0.0)
        visual_extraction_confidence = fallback_visual_confidence(
            extraction_confidence=extraction_confidence,
            has_assets=bool(brand_assets),
        )

        try:
            visual_agent = BrandVisualGuideAgent()
            visual_output = await visual_agent.run(
                BrandVisualGuideInput(
                    company_name=brand.company_name or "Company",
                    tagline=brand.tagline,
                    tone_attributes=[str(item) for item in list(brand.tone_attributes or [])],
                    unique_value_props=[
                        str(item) for item in list(brand.unique_value_props or [])
                    ],
                    differentiators=[str(item) for item in list(brand.differentiators or [])],
                    target_roles=[str(item) for item in list(brand.target_roles or [])],
                    target_industries=[
                        str(item) for item in list(brand.target_industries or [])
                    ],
                    brand_assets=brand_assets,
                    homepage_visual_signals=homepage_visual_signals,
                    site_visual_signals=site_visual_signals,
                )
            )
            visual_style_guide = visual_output.visual_style_guide.model_dump()
            visual_prompt_contract = normalize_prompt_contract(
                visual_output.visual_prompt_contract.model_dump()
            )
            visual_extraction_confidence = float(visual_output.extraction_confidence)
        except Exception:
            logger.exception(
                "Visual style guide generation failed; storing deterministic fallback",
                extra={"project_id": input_data.project_id},
            )

        visual_last_synced_at = datetime.now(timezone.utc)

        await self._update_progress(100, "Visual style guide ready")

        return BrandVisualOutput(
            visual_style_guide=visual_style_guide,
            visual_prompt_contract=visual_prompt_contract,
            visual_extraction_confidence=visual_extraction_confidence,
            visual_last_synced_at=visual_last_synced_at,
        )

    async def _persist_results(self, result: BrandVisualOutput) -> None:
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == self.project_id)
        )
        brand = brand_result.scalar_one()

        brand.patch(
            self.session,
            BrandProfilePatchDTO.from_partial(
                {
                    "visual_style_guide": result.visual_style_guide,
                    "visual_prompt_contract": result.visual_prompt_contract,
                    "visual_extraction_confidence": result.visual_extraction_confidence,
                    "visual_last_synced_at": result.visual_last_synced_at,
                }
            ),
        )

        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, self.step_number)

        self.set_result_summary(
            {
                "visual_extraction_confidence": result.visual_extraction_confidence,
                "visual_last_synced_at": result.visual_last_synced_at.isoformat(),
            }
        )

        await self.session.commit()
