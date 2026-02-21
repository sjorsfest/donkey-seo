"""Step 2: Scrape website and extract core brand profile."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.agents.brand_extractor import BrandExtractorAgent, BrandExtractorInput
from app.integrations.scraper import scrape_website
from app.models.brand import BrandProfile
from app.models.generated_dtos import BrandProfileCreateDTO, BrandProfilePatchDTO
from app.models.project import Project
from app.services.steps.base_step import BaseStepService
from app.services.steps.setup.brand_shared import build_extracted_target_audience

logger = logging.getLogger(__name__)


@dataclass
class BrandCoreInput:
    """Input for setup step 2."""

    project_id: str


@dataclass
class BrandCoreOutput:
    """Output for setup step 2."""

    company_name: str
    tagline: str | None
    products_services: list[dict[str, Any]]
    money_pages: list[dict[str, Any]]
    unique_value_props: list[str]
    differentiators: list[str]
    extracted_target_audience: dict[str, list[str]]
    tone_attributes: list[str]
    allowed_claims: list[str]
    restricted_claims: list[str]
    in_scope_topics: list[str]
    out_of_scope_topics: list[str]
    source_pages: list[str]
    extraction_confidence: float
    asset_candidates: list[dict[str, Any]]


class Step02BrandCoreService(BaseStepService[BrandCoreInput, BrandCoreOutput]):
    """Step 2: Scrape and extract core brand attributes."""

    step_number = 2
    step_name = "brand_core"
    is_optional = False
    _MAX_STORED_ASSET_CANDIDATES = 80

    async def _validate_preconditions(self, input_data: BrandCoreInput) -> None:
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

    async def _execute(self, input_data: BrandCoreInput) -> BrandCoreOutput:
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()
        domain = project.domain

        await self._update_progress(10, f"Scraping website: {domain}...")
        scraped_data = await scrape_website(domain, max_pages=10)
        if scraped_data.get("error"):
            raise ValueError(f"Failed to scrape website: {scraped_data['error']}")

        await self._update_progress(45, "Analyzing brand content with AI...")
        agent_input = BrandExtractorInput(
            domain=domain,
            scraped_content=scraped_data.get("combined_content", ""),
            additional_context=None,
        )

        agent = BrandExtractorAgent()
        brand_profile = await agent.run(agent_input)

        extracted_target_audience = build_extracted_target_audience(
            brand_profile.target_audience
        )

        raw_asset_candidates = [
            candidate
            for candidate in scraped_data.get("asset_candidates", [])
            if isinstance(candidate, dict)
        ]
        stored_asset_candidates = raw_asset_candidates[: self._MAX_STORED_ASSET_CANDIDATES]
        await self.update_steps_config(
            {
                "setup_state": {
                    "brand_source_pages": [
                        str(url)
                        for url in scraped_data.get("source_urls", [])
                        if str(url).strip()
                    ],
                    "asset_candidates": stored_asset_candidates,
                }
            }
        )

        await self._update_progress(100, "Brand core extracted")

        return BrandCoreOutput(
            company_name=brand_profile.company_name,
            tagline=brand_profile.tagline,
            products_services=[
                {
                    "name": p.name,
                    "description": p.description,
                    "category": p.category,
                    "target_audience": p.target_audience,
                    "core_benefits": p.core_benefits,
                }
                for p in brand_profile.products_services
            ],
            money_pages=[
                {"url": url, "purpose": "conversion"}
                for url in brand_profile.money_pages
            ],
            unique_value_props=brand_profile.unique_value_props,
            differentiators=brand_profile.differentiators,
            extracted_target_audience=extracted_target_audience,
            tone_attributes=brand_profile.tone_attributes,
            allowed_claims=brand_profile.allowed_claims,
            restricted_claims=brand_profile.restricted_claims,
            in_scope_topics=brand_profile.in_scope_topics,
            out_of_scope_topics=brand_profile.out_of_scope_topics,
            source_pages=[
                str(url)
                for url in scraped_data.get("source_urls", [])
                if str(url).strip()
            ],
            extraction_confidence=brand_profile.extraction_confidence,
            asset_candidates=stored_asset_candidates,
        )

    async def _persist_results(self, result: BrandCoreOutput) -> None:
        existing = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == self.project_id)
        )
        brand_profile = existing.scalar_one_or_none()

        payload = {
            "company_name": result.company_name,
            "tagline": result.tagline,
            "products_services": result.products_services,
            "money_pages": result.money_pages,
            "unique_value_props": result.unique_value_props,
            "differentiators": result.differentiators,
            "target_roles": result.extracted_target_audience.get("target_roles", []),
            "target_industries": result.extracted_target_audience.get("target_industries", []),
            "company_sizes": result.extracted_target_audience.get("company_sizes", []),
            "primary_pains": result.extracted_target_audience.get("primary_pains", []),
            "desired_outcomes": result.extracted_target_audience.get("desired_outcomes", []),
            "objections": result.extracted_target_audience.get("objections", []),
            "tone_attributes": result.tone_attributes,
            "allowed_claims": result.allowed_claims,
            "restricted_claims": result.restricted_claims,
            "in_scope_topics": result.in_scope_topics,
            "out_of_scope_topics": result.out_of_scope_topics,
            "source_pages": result.source_pages,
            "extraction_confidence": result.extraction_confidence,
            "suggested_icp_niches": [],
        }

        if brand_profile:
            brand_profile.patch(
                self.session,
                BrandProfilePatchDTO.from_partial(payload),
            )
        else:
            BrandProfile.create(
                self.session,
                BrandProfileCreateDTO(
                    project_id=self.project_id,
                    company_name=result.company_name,
                    tagline=result.tagline,
                    products_services=result.products_services,
                    money_pages=result.money_pages,
                    unique_value_props=result.unique_value_props,
                    differentiators=result.differentiators,
                    target_roles=result.extracted_target_audience.get("target_roles", []),
                    target_industries=result.extracted_target_audience.get(
                        "target_industries", []
                    ),
                    company_sizes=result.extracted_target_audience.get("company_sizes", []),
                    primary_pains=result.extracted_target_audience.get("primary_pains", []),
                    desired_outcomes=result.extracted_target_audience.get(
                        "desired_outcomes", []
                    ),
                    objections=result.extracted_target_audience.get("objections", []),
                    suggested_icp_niches=[],
                    tone_attributes=result.tone_attributes,
                    allowed_claims=result.allowed_claims,
                    restricted_claims=result.restricted_claims,
                    in_scope_topics=result.in_scope_topics,
                    out_of_scope_topics=result.out_of_scope_topics,
                    source_pages=result.source_pages,
                    extraction_confidence=result.extraction_confidence,
                ),
            )

        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, self.step_number)

        self.set_result_summary(
            {
                "company_name": result.company_name,
                "products_count": len(result.products_services),
                "money_pages_count": len(result.money_pages),
                "target_roles_count": len(result.extracted_target_audience.get("target_roles", [])),
                "extraction_confidence": result.extraction_confidence,
                "source_pages_count": len(result.source_pages),
                "stored_asset_candidates": len(result.asset_candidates),
            }
        )

        await self.session.commit()
