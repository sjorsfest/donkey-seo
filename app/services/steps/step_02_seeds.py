"""Step 2: Seed Topic Generation.

Generates 10-50 seed topics organized into 3-8 pillars based on brand profile.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.topic_generator import TopicGeneratorAgent, TopicGeneratorInput
from app.models.brand import BrandProfile
from app.models.keyword import SeedTopic
from app.models.pipeline import StepExecution
from app.models.project import Project
from app.services.steps.base_step import BaseStepService, StepResult


@dataclass
class SeedsInput:
    """Input for Step 2."""

    project_id: str


@dataclass
class SeedsOutput:
    """Output from Step 2."""

    pillars_created: int
    topics_created: int
    pillars: list[dict[str, Any]]
    topics: list[dict[str, Any]]
    known_gaps: list[str]


class Step02SeedsService(BaseStepService[SeedsInput, SeedsOutput]):
    """Step 2: Seed Topic Generation.

    Uses TopicGeneratorAgent to create content pillars and seed topics
    based on the brand profile from Step 1.
    """

    step_number = 2
    step_name = "seed_topics"
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
        """Execute seed topic generation."""
        # Load brand profile
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = result.scalar_one()

        await self._update_progress(10, "Preparing brand context for topic generation...")

        # Prepare agent input
        agent_input = TopicGeneratorInput(
            company_name=brand.company_name or "Company",
            products_services=brand.products_services or [],
            target_audience={
                "target_roles": brand.target_roles or [],
                "target_industries": brand.target_industries or [],
                "primary_pains": brand.primary_pains or [],
                "desired_outcomes": brand.desired_outcomes or [],
            },
            unique_value_props=brand.unique_value_props or [],
            in_scope_topics=brand.in_scope_topics or [],
            out_of_scope_topics=brand.out_of_scope_topics or [],
        )

        await self._update_progress(30, "Generating content pillars and seed topics...")

        # Run topic generator agent
        agent = TopicGeneratorAgent()
        output = await agent.run(agent_input)

        await self._update_progress(80, "Processing generated topics...")

        # Convert to output format
        pillars = [
            {
                "name": p.name,
                "description": p.description,
                "icp_relevance": p.icp_relevance,
                "product_tie_in": p.product_tie_in,
            }
            for p in output.pillars
        ]

        topics = [
            {
                "topic_phrase": t.topic_phrase,
                "pillar_name": t.pillar_name,
                "intended_content_types": t.intended_content_types,
                "coverage_intent": t.coverage_intent,
                "funnel_stage": t.funnel_stage,
                "relevance_score": t.relevance_score,
            }
            for t in output.seed_topics
        ]

        await self._update_progress(100, "Seed topic generation complete")

        return SeedsOutput(
            pillars_created=len(pillars),
            topics_created=len(topics),
            pillars=pillars,
            topics=topics,
            known_gaps=output.known_gaps,
        )

    async def _persist_results(self, result: SeedsOutput) -> None:
        """Save seed topics to database."""
        import uuid

        # Delete existing seed topics for this project
        existing = await self.session.execute(
            select(SeedTopic).where(SeedTopic.project_id == self.project_id)
        )
        for topic in existing.scalars():
            await self.session.delete(topic)

        # Create pillar name to description map
        pillar_map = {p["name"]: p for p in result.pillars}

        # Create seed topics
        for topic_data in result.topics:
            pillar_info = pillar_map.get(topic_data["pillar_name"], {})

            seed_topic = SeedTopic(
                project_id=uuid.UUID(self.project_id),
                name=topic_data["topic_phrase"],
                description=pillar_info.get("description"),
                pillar_type="main_pillar",
                icp_relevance=pillar_info.get("icp_relevance"),
                product_tie_in=pillar_info.get("product_tie_in"),
                intended_content_types=topic_data["intended_content_types"],
                coverage_intent=topic_data["coverage_intent"],
                relevance_score=topic_data["relevance_score"],
            )
            self.session.add(seed_topic)

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 2

        # Set result summary
        self.set_result_summary({
            "pillars_created": result.pillars_created,
            "topics_created": result.topics_created,
            "known_gaps_count": len(result.known_gaps),
            "pillar_names": [p["name"] for p in result.pillars],
        })

        await self.session.commit()
