"""Step 1: Brand Profile + ICP Extraction.

Scrapes key pages and extracts brand positioning, products, and audience using LLM.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.brand_extractor import BrandExtractorAgent, BrandExtractorInput
from app.integrations.scraper import scrape_website
from app.models.brand import BrandProfile
from app.models.pipeline import StepExecution
from app.models.project import Project
from app.services.steps.base_step import BaseStepService, StepResult


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
    tone_attributes: list[str]
    allowed_claims: list[str]
    restricted_claims: list[str]
    in_scope_topics: list[str]
    out_of_scope_topics: list[str]
    source_pages: list[str]
    extraction_confidence: float


class Step01BrandService(BaseStepService[BrandInput, BrandOutput]):
    """Step 1: Brand Profile + ICP Extraction.

    Uses WebsiteScraper to crawl key pages, then BrandExtractorAgent
    to extract structured brand information.
    """

    step_number = 1
    step_name = "brand_profile"
    is_optional = False

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

        await self._update_progress(10, f"Scraping website: {domain}...")

        # Scrape website
        scraped_data = await scrape_website(domain, max_pages=10)

        if scraped_data.get("error"):
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
            target_audience={
                "target_roles": brand_profile.target_audience.target_roles,
                "target_industries": brand_profile.target_audience.target_industries,
                "company_sizes": brand_profile.target_audience.company_sizes,
                "primary_pains": brand_profile.target_audience.primary_pains,
                "desired_outcomes": brand_profile.target_audience.desired_outcomes,
                "objections": brand_profile.target_audience.common_objections,
            },
            tone_attributes=brand_profile.tone_attributes,
            allowed_claims=brand_profile.allowed_claims,
            restricted_claims=brand_profile.restricted_claims,
            in_scope_topics=brand_profile.in_scope_topics,
            out_of_scope_topics=brand_profile.out_of_scope_topics,
            source_pages=scraped_data.get("source_urls", []),
            extraction_confidence=brand_profile.extraction_confidence,
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

        if brand_profile:
            # Update existing
            brand_profile.company_name = result.company_name
            brand_profile.tagline = result.tagline
            brand_profile.products_services = result.products_services
            brand_profile.money_pages = result.money_pages
            brand_profile.unique_value_props = result.unique_value_props
            brand_profile.differentiators = result.differentiators
            brand_profile.target_roles = result.target_audience.get("target_roles", [])
            brand_profile.target_industries = result.target_audience.get("target_industries", [])
            brand_profile.company_sizes = result.target_audience.get("company_sizes", [])
            brand_profile.primary_pains = result.target_audience.get("primary_pains", [])
            brand_profile.desired_outcomes = result.target_audience.get("desired_outcomes", [])
            brand_profile.objections = result.target_audience.get("objections", [])
            brand_profile.tone_attributes = result.tone_attributes
            brand_profile.allowed_claims = result.allowed_claims
            brand_profile.restricted_claims = result.restricted_claims
            brand_profile.in_scope_topics = result.in_scope_topics
            brand_profile.out_of_scope_topics = result.out_of_scope_topics
            brand_profile.source_pages = result.source_pages
            brand_profile.extraction_confidence = result.extraction_confidence
        else:
            # Create new
            brand_profile = BrandProfile(
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
                tone_attributes=result.tone_attributes,
                allowed_claims=result.allowed_claims,
                restricted_claims=result.restricted_claims,
                in_scope_topics=result.in_scope_topics,
                out_of_scope_topics=result.out_of_scope_topics,
                source_pages=result.source_pages,
                extraction_confidence=result.extraction_confidence,
            )
            self.session.add(brand_profile)

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 1

        # Set result summary
        self.set_result_summary({
            "company_name": result.company_name,
            "products_count": len(result.products_services),
            "money_pages_count": len(result.money_pages),
            "extraction_confidence": result.extraction_confidence,
            "source_pages_count": len(result.source_pages),
        })

        await self.session.commit()
