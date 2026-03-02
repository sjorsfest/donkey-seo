"""Step 2: Scrape website and extract core brand profile."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    homepage_visual_signals: dict[str, Any]
    site_visual_signals: dict[str, Any]
    extraction_attempts: int = 1
    extraction_warnings: list[str] = field(default_factory=list)


class Step02BrandCoreService(BaseStepService[BrandCoreInput, BrandCoreOutput]):
    """Step 2: Scrape and extract core brand attributes."""

    step_number = 2
    step_name = "brand_core"
    is_optional = False
    _MAX_STORED_ASSET_CANDIDATES = 80
    _MAX_EXTRACTION_ATTEMPTS = 3

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
        extraction_attempts = 0
        extraction_warnings: list[str] = []
        brand_profile: Any | None = None
        extracted_target_audience: dict[str, list[str]] = {}

        for attempt in range(1, self._MAX_EXTRACTION_ATTEMPTS + 1):
            extraction_attempts = attempt
            try:
                candidate_profile = await agent.run(agent_input)
            except Exception as exc:
                extraction_warnings.append(
                    f"attempt_{attempt}:agent_error:{exc.__class__.__name__}"
                )
                if attempt < self._MAX_EXTRACTION_ATTEMPTS:
                    logger.warning(
                        "Brand extraction attempt failed; retrying",
                        extra={
                            "project_id": input_data.project_id,
                            "attempt": attempt,
                            "max_attempts": self._MAX_EXTRACTION_ATTEMPTS,
                        },
                    )
                    continue
                logger.exception(
                    "Brand extraction failed after max retries; continuing with fallback output",
                    extra={
                        "project_id": input_data.project_id,
                        "attempts": self._MAX_EXTRACTION_ATTEMPTS,
                    },
                )
                break

            candidate_target_audience = build_extracted_target_audience(
                candidate_profile.target_audience
            )
            quality_issues = self._brand_quality_issues(
                unique_value_props=list(candidate_profile.unique_value_props or []),
                differentiators=list(candidate_profile.differentiators or []),
                products_services=list(candidate_profile.products_services or []),
                extracted_target_audience=candidate_target_audience,
            )
            if not quality_issues:
                brand_profile = candidate_profile
                extracted_target_audience = candidate_target_audience
                break

            issue_text = ",".join(quality_issues)
            extraction_warnings.append(f"attempt_{attempt}:low_signal:{issue_text}")
            brand_profile = candidate_profile
            extracted_target_audience = candidate_target_audience

            if attempt < self._MAX_EXTRACTION_ATTEMPTS:
                logger.warning(
                    "Brand extraction quality below threshold; retrying",
                    extra={
                        "project_id": input_data.project_id,
                        "attempt": attempt,
                        "max_attempts": self._MAX_EXTRACTION_ATTEMPTS,
                        "issues": quality_issues,
                    },
                )
                continue

            logger.warning(
                "Brand extraction quality remained low after max retries; continuing with best-effort output",
                extra={
                    "project_id": input_data.project_id,
                    "attempts": self._MAX_EXTRACTION_ATTEMPTS,
                    "issues": quality_issues,
                },
            )

        if brand_profile is None:
            extraction_warnings.append("fallback_brand_profile_applied")
            raw_asset_candidates = [
                candidate
                for candidate in scraped_data.get("asset_candidates", [])
                if isinstance(candidate, dict)
            ]
            stored_asset_candidates = raw_asset_candidates[: self._MAX_STORED_ASSET_CANDIDATES]
            homepage_visual_signals = scraped_data.get("homepage_visual_signals", {})
            site_visual_signals = scraped_data.get("site_visual_signals", {})
            source_pages = [
                str(url)
                for url in scraped_data.get("source_urls", [])
                if str(url).strip()
            ]
            await self.update_steps_config(
                {
                    "setup_state": {
                        "brand_source_pages": source_pages,
                        "asset_candidates": stored_asset_candidates,
                        "homepage_visual_signals": (
                            homepage_visual_signals
                            if isinstance(homepage_visual_signals, dict)
                            else {}
                        ),
                        "site_visual_signals": (
                            site_visual_signals if isinstance(site_visual_signals, dict) else {}
                        ),
                    }
                }
            )
            await self._update_progress(100, "Brand core extracted (fallback mode)")
            return BrandCoreOutput(
                company_name=str(getattr(project, "name", "") or domain),
                tagline=None,
                products_services=[],
                money_pages=[],
                unique_value_props=[],
                differentiators=[],
                extracted_target_audience={
                    "target_roles": [],
                    "target_industries": [],
                    "company_sizes": [],
                    "primary_pains": [],
                    "desired_outcomes": [],
                    "objections": [],
                },
                tone_attributes=[],
                allowed_claims=[],
                restricted_claims=[],
                in_scope_topics=[],
                out_of_scope_topics=[],
                source_pages=source_pages,
                extraction_confidence=0.0,
                asset_candidates=stored_asset_candidates,
                homepage_visual_signals=(
                    homepage_visual_signals if isinstance(homepage_visual_signals, dict) else {}
                ),
                site_visual_signals=(
                    site_visual_signals if isinstance(site_visual_signals, dict) else {}
                ),
                extraction_attempts=extraction_attempts,
                extraction_warnings=extraction_warnings,
            )

        raw_asset_candidates = [
            candidate
            for candidate in scraped_data.get("asset_candidates", [])
            if isinstance(candidate, dict)
        ]
        stored_asset_candidates = raw_asset_candidates[: self._MAX_STORED_ASSET_CANDIDATES]
        homepage_visual_signals = scraped_data.get("homepage_visual_signals", {})
        site_visual_signals = scraped_data.get("site_visual_signals", {})
        await self.update_steps_config(
            {
                "setup_state": {
                    "brand_source_pages": [
                        str(url)
                        for url in scraped_data.get("source_urls", [])
                        if str(url).strip()
                    ],
                    "asset_candidates": stored_asset_candidates,
                    "homepage_visual_signals": (
                        homepage_visual_signals
                        if isinstance(homepage_visual_signals, dict)
                        else {}
                    ),
                    "site_visual_signals": (
                        site_visual_signals if isinstance(site_visual_signals, dict) else {}
                    ),
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
            homepage_visual_signals=(
                homepage_visual_signals if isinstance(homepage_visual_signals, dict) else {}
            ),
            site_visual_signals=(
                site_visual_signals if isinstance(site_visual_signals, dict) else {}
            ),
            extraction_attempts=extraction_attempts,
            extraction_warnings=extraction_warnings,
        )

    @staticmethod
    def _has_non_empty_strings(values: list[str]) -> bool:
        return any(str(value).strip() for value in values)

    def _has_product_signals(self, products_services: list[Any]) -> bool:
        for product in products_services:
            name = str(getattr(product, "name", "") or "").strip()
            description = str(getattr(product, "description", "") or "").strip()
            if name or description:
                return True
        return False

    def _has_icp_signals(self, extracted_target_audience: dict[str, list[str]]) -> bool:
        return any(
            self._has_non_empty_strings(extracted_target_audience.get(key, []))
            for key in (
                "target_roles",
                "target_industries",
                "company_sizes",
                "primary_pains",
                "desired_outcomes",
                "objections",
            )
        )

    def _brand_quality_issues(
        self,
        *,
        unique_value_props: list[str],
        differentiators: list[str],
        products_services: list[Any],
        extracted_target_audience: dict[str, list[str]],
    ) -> list[str]:
        issues: list[str] = []

        has_positioning = (
            self._has_non_empty_strings(unique_value_props)
            or self._has_non_empty_strings(differentiators)
        )
        if not has_positioning:
            issues.append("missing_positioning")

        has_products = self._has_product_signals(products_services)
        has_icp_signals = self._has_icp_signals(extracted_target_audience)
        if not has_products and not has_icp_signals:
            issues.append("missing_product_and_icp_signals")

        return issues

    async def _validate_output(self, result: BrandCoreOutput, input_data: BrandCoreInput) -> None:
        return None

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
                "homepage_visual_signal_keys": len(result.homepage_visual_signals),
                "extraction_attempts": result.extraction_attempts,
                "extraction_warnings_count": len(result.extraction_warnings),
                "extraction_warnings": result.extraction_warnings,
            }
        )

        await self.session.commit()
