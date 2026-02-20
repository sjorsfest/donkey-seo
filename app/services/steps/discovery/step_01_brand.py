"""Step 1: Brand Profile + ICP Extraction.

Scrapes key pages and extracts brand positioning, products, and audience using LLM.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.agents.brand_extractor import BrandExtractorAgent, BrandExtractorInput
from app.agents.brand_visual_guide import (
    BrandVisualGuideAgent,
    BrandVisualGuideInput,
)
from app.agents.icp_recommender import ICPRecommenderAgent, ICPRecommenderInput
from app.integrations.asset_store import BrandAssetStore
from app.integrations.scraper import scrape_website
from app.models.brand import BrandProfile
from app.models.generated_dtos import BrandProfileCreateDTO, BrandProfilePatchDTO
from app.models.project import Project
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class BrandInput:
    """Input for Step 1."""

    project_id: str


@dataclass
class BrandOutput:
    """Output from Step 1."""

    company_name: str
    tagline: str | None
    products_services: list[dict[str, Any]]
    money_pages: list[dict[str, Any]]
    unique_value_props: list[str]
    differentiators: list[str]
    target_audience: dict[str, Any]
    suggested_icp_niches: list[dict[str, Any]]
    tone_attributes: list[str]
    allowed_claims: list[str]
    restricted_claims: list[str]
    in_scope_topics: list[str]
    out_of_scope_topics: list[str]
    source_pages: list[str]
    extraction_confidence: float
    brand_assets: list[dict[str, Any]]
    visual_style_guide: dict[str, Any]
    visual_prompt_contract: dict[str, Any]
    visual_extraction_confidence: float
    visual_last_synced_at: datetime


class Step01BrandService(BaseStepService[BrandInput, BrandOutput]):
    """Step 1: Brand Profile + ICP Extraction.

    Uses WebsiteScraper to crawl key pages, then BrandExtractorAgent
    to extract structured brand information.
    """

    step_number = 1
    step_name = "brand_profile"
    is_optional = False
    _TARGET_AUDIENCE_KEYS = (
        "target_roles",
        "target_industries",
        "company_sizes",
        "primary_pains",
        "desired_outcomes",
        "objections",
    )
    _PROMPT_CONTRACT_REQUIRED_VARIABLES = [
        "article_topic",
        "audience",
        "intent",
        "visual_goal",
        "brand_voice",
        "asset_refs",
    ]

    async def _validate_preconditions(self, input_data: BrandInput) -> None:
        """Validate Step 0 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 0:
            raise ValueError("Step 0 (Setup) must be completed first")

    async def _execute(self, input_data: BrandInput) -> BrandOutput:
        """Execute brand profile extraction."""
        # Load project
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()
        domain = project.domain
        logger.info(
            "Starting brand extraction",
            extra={"project_id": input_data.project_id, "domain": domain},
        )

        await self._update_progress(10, f"Scraping website: {domain}...")

        # Scrape website
        scraped_data = await scrape_website(domain, max_pages=10)
        logger.info(
            "Website scraped",
            extra={
                "domain": domain,
                "pages_scraped": len(scraped_data.get("source_urls", [])),
                "content_length": len(scraped_data.get("combined_content", "")),
            },
        )

        if scraped_data.get("error"):
            logger.warning(
                "Scraping failed",
                extra={"domain": domain, "error": scraped_data["error"]},
            )
            raise ValueError(f"Failed to scrape website: {scraped_data['error']}")

        await self._update_progress(40, "Analyzing brand content with AI...")

        # Prepare input for agent
        agent_input = BrandExtractorInput(
            domain=domain,
            scraped_content=scraped_data.get("combined_content", ""),
            additional_context=None,
        )

        # Run brand extraction agent
        agent = BrandExtractorAgent()
        brand_profile = await agent.run(agent_input)
        logger.info(
            "Brand profile extracted",
            extra={
                "company_name": brand_profile.company_name,
                "confidence": brand_profile.extraction_confidence,
                "products_count": len(brand_profile.products_services),
            },
        )

        extracted_target_audience = {
            "target_roles": list(brand_profile.target_audience.target_roles),
            "target_industries": list(brand_profile.target_audience.target_industries),
            "company_sizes": list(brand_profile.target_audience.company_sizes),
            "primary_pains": list(brand_profile.target_audience.primary_pains),
            "desired_outcomes": list(brand_profile.target_audience.desired_outcomes),
            "objections": list(brand_profile.target_audience.common_objections),
        }

        await self._update_progress(65, "Expanding ICP opportunities with AI...")
        icp_niches: list[dict[str, Any]] = []
        try:
            recommender = ICPRecommenderAgent()
            recommendation = await recommender.run(
                ICPRecommenderInput(
                    company_name=brand_profile.company_name,
                    tagline=brand_profile.tagline,
                    products_services=[
                        {
                            "name": product.name,
                            "description": product.description,
                            "category": product.category,
                            "target_audience": product.target_audience,
                            "core_benefits": product.core_benefits,
                        }
                        for product in brand_profile.products_services
                    ],
                    unique_value_props=brand_profile.unique_value_props,
                    differentiators=brand_profile.differentiators,
                    current_target_audience=extracted_target_audience,
                )
            )
            icp_niches = [niche.model_dump() for niche in recommendation.suggested_niches]
            logger.info(
                "Generated ICP niche recommendations",
                extra={
                    "company_name": brand_profile.company_name,
                    "niches_count": len(icp_niches),
                    "confidence": recommendation.recommendation_confidence,
                },
            )
        except Exception:
            logger.exception(
                "ICP niche recommendation failed; falling back to extracted website ICP only",
                extra={"company_name": brand_profile.company_name},
            )

        merged_target_audience = self._merge_target_audience(
            extracted_target_audience=extracted_target_audience,
            suggested_icp_niches=icp_niches,
        )

        await self._update_progress(80, "Discovering and storing brand image assets...")

        existing_brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        existing_brand = existing_brand_result.scalar_one_or_none()

        brand_assets = list(existing_brand.brand_assets or []) if existing_brand else []
        raw_asset_candidates = [
            candidate
            for candidate in scraped_data.get("asset_candidates", [])
            if isinstance(candidate, dict)
        ]
        asset_candidates = [
            candidate
            for candidate in raw_asset_candidates
            if not self._is_low_quality_icon_candidate(candidate)
        ]
        skipped_assets = len(raw_asset_candidates) - len(asset_candidates)
        if skipped_assets:
            logger.info(
                "Skipping low-quality icon asset candidates",
                extra={"project_id": input_data.project_id, "skipped_assets": skipped_assets},
            )

        if asset_candidates:
            try:
                store = BrandAssetStore()
                brand_assets = await store.ingest_asset_candidates(
                    project_id=input_data.project_id,
                    asset_candidates=asset_candidates,
                    existing_assets=brand_assets,
                    origin="step_01_auto",
                )
            except Exception:
                logger.exception(
                    "Brand asset ingestion failed; continuing with textual profile only",
                    extra={"project_id": input_data.project_id, "domain": domain},
                )

        await self._update_progress(88, "Generating visual prompt style guide...")

        visual_style_guide = self._default_visual_style_guide(
            tone_attributes=brand_profile.tone_attributes,
            differentiators=brand_profile.differentiators,
        )
        visual_prompt_contract = self._default_visual_prompt_contract()
        visual_extraction_confidence = self._fallback_visual_confidence(
            extraction_confidence=brand_profile.extraction_confidence,
            has_assets=bool(brand_assets),
        )

        try:
            visual_agent = BrandVisualGuideAgent()
            visual_output = await visual_agent.run(
                BrandVisualGuideInput(
                    company_name=brand_profile.company_name,
                    tagline=brand_profile.tagline,
                    tone_attributes=brand_profile.tone_attributes,
                    unique_value_props=brand_profile.unique_value_props,
                    differentiators=brand_profile.differentiators,
                    target_roles=merged_target_audience.get("target_roles", []),
                    target_industries=merged_target_audience.get("target_industries", []),
                    brand_assets=brand_assets,
                )
            )
            visual_style_guide = visual_output.visual_style_guide.model_dump()
            visual_prompt_contract = self._normalize_prompt_contract(
                visual_output.visual_prompt_contract.model_dump()
            )
            visual_extraction_confidence = float(visual_output.extraction_confidence)
        except Exception:
            logger.exception(
                "Visual style guide generation failed; storing deterministic fallback",
                extra={"project_id": input_data.project_id, "domain": domain},
            )

        visual_last_synced_at = datetime.now(timezone.utc)

        await self._update_progress(90, "Finalizing brand profile...")

        # Convert agent output to step output
        return BrandOutput(
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
            target_audience=merged_target_audience,
            suggested_icp_niches=icp_niches,
            tone_attributes=brand_profile.tone_attributes,
            allowed_claims=brand_profile.allowed_claims,
            restricted_claims=brand_profile.restricted_claims,
            in_scope_topics=brand_profile.in_scope_topics,
            out_of_scope_topics=brand_profile.out_of_scope_topics,
            source_pages=scraped_data.get("source_urls", []),
            extraction_confidence=brand_profile.extraction_confidence,
            brand_assets=brand_assets,
            visual_style_guide=visual_style_guide,
            visual_prompt_contract=visual_prompt_contract,
            visual_extraction_confidence=visual_extraction_confidence,
            visual_last_synced_at=visual_last_synced_at,
        )

    async def _persist_results(self, result: BrandOutput) -> None:
        """Save brand profile to database."""
        # Check if brand profile already exists
        existing = await self.session.execute(
            select(BrandProfile).where(
                BrandProfile.project_id == self.project_id
            )
        )
        brand_profile = existing.scalar_one_or_none()

        profile_payload = {
            "company_name": result.company_name,
            "tagline": result.tagline,
            "products_services": result.products_services,
            "money_pages": result.money_pages,
            "unique_value_props": result.unique_value_props,
            "differentiators": result.differentiators,
            "target_roles": result.target_audience.get("target_roles", []),
            "target_industries": result.target_audience.get("target_industries", []),
            "company_sizes": result.target_audience.get("company_sizes", []),
            "primary_pains": result.target_audience.get("primary_pains", []),
            "desired_outcomes": result.target_audience.get("desired_outcomes", []),
            "objections": result.target_audience.get("objections", []),
            "suggested_icp_niches": result.suggested_icp_niches,
            "tone_attributes": result.tone_attributes,
            "allowed_claims": result.allowed_claims,
            "restricted_claims": result.restricted_claims,
            "in_scope_topics": result.in_scope_topics,
            "out_of_scope_topics": result.out_of_scope_topics,
            "source_pages": result.source_pages,
            "extraction_confidence": result.extraction_confidence,
            "brand_assets": result.brand_assets,
            "visual_style_guide": result.visual_style_guide,
            "visual_prompt_contract": result.visual_prompt_contract,
            "visual_extraction_confidence": result.visual_extraction_confidence,
            "visual_last_synced_at": result.visual_last_synced_at,
        }

        if brand_profile:
            brand_profile.patch(
                self.session,
                BrandProfilePatchDTO.from_partial(profile_payload),
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
                    target_roles=result.target_audience.get("target_roles", []),
                    target_industries=result.target_audience.get("target_industries", []),
                    company_sizes=result.target_audience.get("company_sizes", []),
                    primary_pains=result.target_audience.get("primary_pains", []),
                    desired_outcomes=result.target_audience.get("desired_outcomes", []),
                    objections=result.target_audience.get("objections", []),
                    suggested_icp_niches=result.suggested_icp_niches,
                    tone_attributes=result.tone_attributes,
                    allowed_claims=result.allowed_claims,
                    restricted_claims=result.restricted_claims,
                    in_scope_topics=result.in_scope_topics,
                    out_of_scope_topics=result.out_of_scope_topics,
                    source_pages=result.source_pages,
                    extraction_confidence=result.extraction_confidence,
                    brand_assets=result.brand_assets,
                    visual_style_guide=result.visual_style_guide,
                    visual_prompt_contract=result.visual_prompt_contract,
                    visual_extraction_confidence=result.visual_extraction_confidence,
                    visual_last_synced_at=result.visual_last_synced_at,
                ),
            )

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, 1)

        # Set result summary
        self.set_result_summary({
            "company_name": result.company_name,
            "products_count": len(result.products_services),
            "money_pages_count": len(result.money_pages),
            "icp_niches_count": len(result.suggested_icp_niches),
            "target_roles_count": len(result.target_audience.get("target_roles", [])),
            "extraction_confidence": result.extraction_confidence,
            "source_pages_count": len(result.source_pages),
            "brand_assets_count": len(result.brand_assets),
            "visual_extraction_confidence": result.visual_extraction_confidence,
        })

        await self.session.commit()

    @classmethod
    def _merge_target_audience(
        cls,
        *,
        extracted_target_audience: dict[str, list[str]],
        suggested_icp_niches: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Merge website ICP extraction with LLM-suggested ICP niches."""
        merged: dict[str, list[str]] = {
            key: cls._normalize_string_list(extracted_target_audience.get(key, []))
            for key in cls._TARGET_AUDIENCE_KEYS
        }
        suggested_lists = cls._extract_niche_audience_lists(suggested_icp_niches)
        max_items = {
            "target_roles": 20,
            "target_industries": 20,
            "company_sizes": 12,
            "primary_pains": 20,
            "desired_outcomes": 20,
            "objections": 20,
        }

        for key in cls._TARGET_AUDIENCE_KEYS:
            combined = [*merged.get(key, []), *suggested_lists.get(key, [])]
            merged[key] = cls._normalize_string_list(combined)[: max_items[key]]

        return merged

    @classmethod
    def _extract_niche_audience_lists(
        cls,
        suggested_icp_niches: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Flatten list-style fields from niche recommendations."""
        extracted: dict[str, list[str]] = {
            key: []
            for key in cls._TARGET_AUDIENCE_KEYS
        }
        objections_aliases = ("objections", "likely_objections", "common_objections")

        for niche in suggested_icp_niches:
            for key in (
                "target_roles",
                "target_industries",
                "company_sizes",
                "primary_pains",
                "desired_outcomes",
            ):
                values = niche.get(key, [])
                if isinstance(values, list):
                    extracted[key].extend(str(value) for value in values)

            for alias in objections_aliases:
                if alias not in niche:
                    continue
                objections = niche.get(alias, [])
                if isinstance(objections, list):
                    extracted["objections"].extend(str(value) for value in objections)
                    break

        return extracted

    @staticmethod
    def _normalize_string_list(items: list[str]) -> list[str]:
        """Trim, deduplicate, and preserve order for text list fields."""
        normalized: list[str] = []
        seen: set[str] = set()

        for item in items:
            value = " ".join(str(item).strip().split())
            if not value:
                continue

            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)

        return normalized

    @staticmethod
    def _is_low_quality_icon_candidate(candidate: dict[str, Any]) -> bool:
        """Skip icon assets that are typically too small for visual guide generation."""
        role = str(candidate.get("role") or "").strip().lower()
        origin = str(candidate.get("origin") or "").strip().lower()
        source_url = str(candidate.get("url") or "").strip().lower()

        if role == "icon" or origin == "link_icon":
            return True

        low_quality_tokens = ("favicon", "apple-touch-icon", "apple-touch")
        return any(token in source_url for token in low_quality_tokens)

    @classmethod
    def _normalize_prompt_contract(cls, prompt_contract: dict[str, Any]) -> dict[str, Any]:
        """Ensure prompt contract has deterministic required fields."""
        required = cls._normalize_string_list(
            [
                *cls._PROMPT_CONTRACT_REQUIRED_VARIABLES,
                *[
                    str(item)
                    for item in (prompt_contract.get("required_variables") or [])
                ],
            ]
        )
        prompt_contract["required_variables"] = required

        template = str(prompt_contract.get("template") or "").strip()
        if not template:
            template = cls._default_visual_prompt_contract()["template"]

        for variable in cls._PROMPT_CONTRACT_REQUIRED_VARIABLES:
            placeholder = "{" + variable + "}"
            if placeholder not in template:
                template = f"{template} {placeholder}".strip()
        prompt_contract["template"] = " ".join(template.split())

        for key in ("forbidden_terms", "fallback_rules", "render_targets"):
            prompt_contract[key] = cls._normalize_string_list(
                [str(value) for value in (prompt_contract.get(key) or [])]
            )

        return prompt_contract

    @staticmethod
    def _fallback_visual_confidence(
        *,
        extraction_confidence: float,
        has_assets: bool,
    ) -> float:
        baseline = min(max(extraction_confidence * 0.5, 0.0), 0.6)
        if has_assets:
            baseline = max(baseline, 0.2)
        return round(baseline, 3)

    @staticmethod
    def _default_visual_style_guide(
        *,
        tone_attributes: list[str],
        differentiators: list[str],
    ) -> dict[str, Any]:
        tone_text = ", ".join(tone_attributes[:4]) if tone_attributes else "professional"
        differentiators_text = (
            ", ".join(differentiators[:3])
            if differentiators
            else "clear positioning and practical value"
        )
        return {
            "brand_palette": {},
            "contrast_rules": [
                "Maintain high contrast between text and background elements.",
            ],
            "composition_rules": [
                "Prefer clean layouts with one dominant focal point.",
                "Align imagery with a tone that is " + tone_text + ".",
            ],
            "subject_rules": [
                "Center visuals around realistic business scenarios.",
                "Reinforce differentiators: " + differentiators_text + ".",
            ],
            "camera_lighting_rules": [
                "Use natural lighting and avoid harsh color casts.",
            ],
            "logo_usage_rules": [
                "Place logos in clear space and avoid distortion.",
            ],
            "negative_rules": [
                "Avoid exaggerated claims, misleading visuals, and generic stock look.",
            ],
            "accessibility_rules": [
                "Ensure readable overlays and avoid low-contrast color combinations.",
            ],
        }

    @classmethod
    def _default_visual_prompt_contract(cls) -> dict[str, Any]:
        return {
            "template": (
                "Create an on-brand image for {article_topic} aimed at {audience}. "
                "Intent: {intent}. Visual goal: {visual_goal}. "
                "Brand voice: {brand_voice}. Asset references: {asset_refs}."
            ),
            "required_variables": list(cls._PROMPT_CONTRACT_REQUIRED_VARIABLES),
            "forbidden_terms": [
                "photoreal celebrity likeness",
                "unverified performance claims",
            ],
            "fallback_rules": [
                "If no logo asset is available, use neutral composition with brand colors only.",
                "If brand palette is missing, keep palette restrained and high-contrast.",
            ],
            "render_targets": [
                "blog_hero",
                "social_preview",
                "feature_callout",
            ],
        }
