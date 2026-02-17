"""Step 2: Seed Keyword Generation.

Generates 20-50 seed keywords organized into buckets based on brand profile.
"""

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.agents.topic_generator import TopicGeneratorAgent, TopicGeneratorInput
from app.models.brand import BrandProfile
from app.models.generated_dtos import SeedTopicCreateDTO
from app.models.keyword import SeedTopic
from app.models.project import Project
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class SeedsInput:
    """Input for Step 2."""

    project_id: str


@dataclass
class SeedsOutput:
    """Output from Step 2."""

    buckets_created: int
    seeds_created: int
    buckets: list[dict[str, Any]]
    seeds: list[dict[str, Any]]
    known_gaps: list[str]


class Step02SeedsService(BaseStepService[SeedsInput, SeedsOutput]):
    """Step 2: Seed Keyword Generation.

    Uses TopicGeneratorAgent to create seed keyword buckets and seed keywords
    based on the brand profile from Step 1.
    """

    step_number = 2
    step_name = "seed_keywords"
    is_optional = False

    async def _validate_preconditions(self, input_data: SeedsInput) -> None:
        """Validate Step 1 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 1:
            raise ValueError("Step 1 (Brand Profile) must be completed first")

        # Check brand profile exists
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        if not brand_result.scalar_one_or_none():
            raise ValueError("Brand profile not found. Run Step 1 first.")

    async def _execute(self, input_data: SeedsInput) -> SeedsOutput:
        """Execute seed keyword generation."""
        # Load brand profile
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = result.scalar_one()
        strategy = await self.get_run_strategy()
        logger.info("Generating seed keywords", extra={"project_id": input_data.project_id, "company": brand.company_name})

        await self._update_progress(10, "Preparing brand context for seed keyword generation...")

        in_scope_topics = self._dedupe_topics(
            list(brand.in_scope_topics or []) + strategy.include_topics
        )
        out_of_scope_topics = self._dedupe_topics(
            list(brand.out_of_scope_topics or []) + strategy.exclude_topics
        )
        offer_categories = self._extract_offer_categories(brand.products_services or [])
        buyer_jobs = self._extract_buyer_jobs(brand)

        # Prepare agent input
        agent_input = TopicGeneratorInput(
            company_name=brand.company_name or "Company",
            products_services=brand.products_services or [],
            offer_categories=offer_categories,
            target_audience={
                "target_roles": strategy.icp_roles or brand.target_roles or [],
                "target_industries": strategy.icp_industries or brand.target_industries or [],
                "primary_pains": strategy.icp_pains or brand.primary_pains or [],
                "desired_outcomes": brand.desired_outcomes or [],
            },
            buyer_jobs=buyer_jobs,
            conversion_intents=strategy.conversion_intents,
            unique_value_props=brand.unique_value_props or [],
            in_scope_topics=in_scope_topics,
            out_of_scope_topics=out_of_scope_topics,
        )

        await self._update_progress(30, "Generating seed keywords...")

        # Run topic generator agent
        agent = TopicGeneratorAgent()
        output = await agent.run(agent_input)
        logger.info("Seed keywords generated", extra={"buckets_count": len(output.buckets), "seeds_count": len(output.seed_keywords), "known_gaps": len(output.known_gaps)})

        await self._update_progress(80, "Processing generated seed keywords...")

        # Convert to output format
        buckets = [
            {
                "name": b.name,
                "description": b.description,
                "icp_relevance": b.icp_relevance,
                "product_tie_in": b.product_tie_in,
            }
            for b in output.buckets
        ]

        seeds = [
            {
                "keyword": s.keyword,
                "bucket_name": s.bucket_name,
                "relevance_score": s.relevance_score,
            }
            for s in output.seed_keywords
        ]

        await self._update_progress(100, "Seed keyword generation complete")

        return SeedsOutput(
            buckets_created=len(buckets),
            seeds_created=len(seeds),
            buckets=buckets,
            seeds=seeds,
            known_gaps=output.known_gaps,
        )

    def _dedupe_topics(self, topics: list[str]) -> list[str]:
        """Deduplicate topic names case-insensitively."""
        seen: set[str] = set()
        deduped: list[str] = []
        for topic in topics:
            cleaned = topic.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _extract_offer_categories(self, products_services: list[dict[str, Any]]) -> list[str]:
        """Extract a compact set of offer categories from product/service records."""
        categories: list[str] = []
        for product in products_services:
            category = (product.get("category") or "").strip()
            if category:
                categories.append(category)
        return self._dedupe_topics(categories)

    def _extract_buyer_jobs(self, brand: BrandProfile) -> list[str]:
        """Build buyer jobs from pains/outcomes when explicit jobs are unavailable."""
        jobs: list[str] = []
        for pain in brand.primary_pains or []:
            jobs.append(f"solve {pain}")
        jobs.extend(brand.desired_outcomes or [])
        return self._dedupe_topics(jobs)

    async def _validate_output(self, result: SeedsOutput, input_data: SeedsInput) -> None:
        """Ensure output can be consumed by Step 3."""
        if result.seeds_created <= 0:
            raise ValueError(
                "Step 2 generated 0 seed keywords. Step 3 requires at least one seed keyword."
            )

    async def _persist_results(self, result: SeedsOutput) -> None:
        """Save seed keywords to database."""
        # Delete existing seed topics for this project
        existing = await self.session.execute(
            select(SeedTopic).where(SeedTopic.project_id == self.project_id)
        )
        for topic in existing.scalars():
            await topic.delete(self.session)

        # Create bucket name to info map
        bucket_map = {b["name"]: b for b in result.buckets}

        # Create seed keyword records
        for seed_data in result.seeds:
            bucket_info = bucket_map.get(seed_data["bucket_name"], {})

            SeedTopic.create(
                self.session,
                SeedTopicCreateDTO(
                    project_id=self.project_id,
                    name=seed_data["keyword"],
                    description=bucket_info.get("description"),
                    pillar_type=seed_data["bucket_name"],
                    icp_relevance=bucket_info.get("icp_relevance"),
                    product_tie_in=bucket_info.get("product_tie_in"),
                    relevance_score=seed_data["relevance_score"],
                ),
            )

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 2

        # Set result summary
        self.set_result_summary({
            "buckets_created": result.buckets_created,
            "seeds_created": result.seeds_created,
            "known_gaps_count": len(result.known_gaps),
            "bucket_names": [b["name"] for b in result.buckets],
        })

        await self.session.commit()
