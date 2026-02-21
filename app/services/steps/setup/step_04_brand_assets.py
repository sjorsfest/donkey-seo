"""Step 4: Ingest brand assets for visual generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.integrations.asset_store import BrandAssetStore
from app.integrations.scraper import scrape_website
from app.models.brand import BrandProfile
from app.models.generated_dtos import BrandProfilePatchDTO
from app.models.project import Project
from app.services.steps.base_step import BaseStepService
from app.services.steps.setup.brand_shared import is_low_quality_icon_candidate

logger = logging.getLogger(__name__)


@dataclass
class BrandAssetsInput:
    """Input for setup step 4."""

    project_id: str


@dataclass
class BrandAssetsOutput:
    """Output for setup step 4."""

    brand_assets: list[dict[str, Any]]
    source_pages: list[str]
    skipped_assets: int


class Step04BrandAssetsService(BaseStepService[BrandAssetsInput, BrandAssetsOutput]):
    """Step 4: Ingest and persist brand image asset inventory."""

    step_number = 4
    step_name = "brand_assets"
    is_optional = False

    async def _validate_preconditions(self, input_data: BrandAssetsInput) -> None:
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = result.scalar_one_or_none()
        if not brand:
            raise ValueError("Brand profile not found. Run setup step 2 first.")

    async def _execute(self, input_data: BrandAssetsInput) -> BrandAssetsOutput:
        project_result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = project_result.scalar_one()

        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one()

        steps_config = await self.get_steps_config()
        setup_state_raw = steps_config.get("setup_state")
        setup_state = setup_state_raw if isinstance(setup_state_raw, dict) else {}

        source_pages = [
            str(url)
            for url in setup_state.get("brand_source_pages", [])
            if str(url).strip()
        ]
        raw_asset_candidates = [
            candidate
            for candidate in setup_state.get("asset_candidates", [])
            if isinstance(candidate, dict)
        ]

        if not raw_asset_candidates:
            await self._update_progress(20, "No cached asset candidates found, rescanning website...")
            scraped_data = await scrape_website(project.domain, max_pages=10)
            if not scraped_data.get("error"):
                raw_asset_candidates = [
                    candidate
                    for candidate in scraped_data.get("asset_candidates", [])
                    if isinstance(candidate, dict)
                ]
                source_pages = [
                    str(url)
                    for url in scraped_data.get("source_urls", [])
                    if str(url).strip()
                ]

        await self._update_progress(60, "Filtering and ingesting brand image assets...")
        asset_candidates = [
            candidate
            for candidate in raw_asset_candidates
            if not is_low_quality_icon_candidate(candidate)
        ]
        skipped_assets = len(raw_asset_candidates) - len(asset_candidates)

        existing_assets = [
            item
            for item in list(brand.brand_assets or [])
            if isinstance(item, dict)
        ]
        brand_assets = existing_assets
        if asset_candidates:
            try:
                store = BrandAssetStore()
                brand_assets = await store.ingest_asset_candidates(
                    project_id=input_data.project_id,
                    asset_candidates=asset_candidates,
                    existing_assets=existing_assets,
                    origin="setup_step_04_auto",
                )
            except Exception:
                logger.exception(
                    "Brand asset ingestion failed; continuing with existing assets",
                    extra={"project_id": input_data.project_id},
                )

        await self.update_steps_config(
            {
                "setup_state": {
                    "asset_candidates": [],
                }
            }
        )

        await self._update_progress(100, "Brand assets synchronized")

        return BrandAssetsOutput(
            brand_assets=brand_assets,
            source_pages=source_pages,
            skipped_assets=skipped_assets,
        )

    async def _persist_results(self, result: BrandAssetsOutput) -> None:
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == self.project_id)
        )
        brand = brand_result.scalar_one()

        payload: dict[str, Any] = {
            "brand_assets": result.brand_assets,
        }
        if result.source_pages:
            payload["source_pages"] = result.source_pages

        brand.patch(self.session, BrandProfilePatchDTO.from_partial(payload))

        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, self.step_number)

        self.set_result_summary(
            {
                "brand_assets_count": len(result.brand_assets),
                "source_pages_count": len(result.source_pages),
                "skipped_assets": result.skipped_assets,
            }
        )

        await self.session.commit()
