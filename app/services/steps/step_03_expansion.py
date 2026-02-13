"""Step 3: Keyword Expansion.

Expands seed topics into keyword candidates using DataForSEO API.
Includes budget control to prevent API cost explosion.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.integrations.dataforseo import DataForSEOClient, get_location_code
from app.models.brand import BrandProfile
from app.models.generated_dtos import KeywordCreateDTO
from app.models.keyword import Keyword, SeedTopic
from app.models.project import Project
from app.persistence.typed import create, delete
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class ExpansionInput:
    """Input for Step 3."""

    project_id: str


@dataclass
class ExpansionOutput:
    """Output from Step 3."""

    keywords_generated: int
    api_calls_made: int
    budget_remaining: int
    seeds_processed: int
    seeds_skipped: int
    keywords: list[dict[str, Any]] = field(default_factory=list)


class Step03ExpansionService(BaseStepService[ExpansionInput, ExpansionOutput]):
    """Step 3: Keyword Expansion.

    Uses DataForSEO API to expand seed topics into keyword candidates.
    Implements budget control to prevent API cost explosion.

    Budget Control Settings (from Project.settings):
    - expansion_budget: Max keywords to generate (default 500)
    - seeds_per_pillar: Limit seeds expanded per pillar (default 5)
    - api_calls_per_step: Max API calls per step run (default 50)
    """

    step_number = 3
    step_name = "keyword_expansion"
    is_optional = False

    # Default budget settings
    DEFAULT_EXPANSION_BUDGET = 500
    DEFAULT_SEEDS_PER_PILLAR = 5
    DEFAULT_API_CALLS_PER_STEP = 50

    async def _validate_preconditions(self, input_data: ExpansionInput) -> None:
        """Validate Step 2 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 2:
            raise ValueError("Step 2 (Seed Topics) must be completed first")

        # Check seed topics exist
        seeds_result = await self.session.execute(
            select(SeedTopic).where(SeedTopic.project_id == input_data.project_id)
        )
        if not seeds_result.scalars().first():
            raise ValueError("No seed topics found. Run Step 2 first.")

    async def _execute(self, input_data: ExpansionInput) -> ExpansionOutput:
        """Execute keyword expansion with budget control."""
        # Load project and settings
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()

        # Load brand profile for exclusion filters
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()

        # Get budget settings from project
        settings = project.api_budget_caps or {}
        expansion_budget = settings.get("expansion_budget", self.DEFAULT_EXPANSION_BUDGET)
        seeds_per_pillar = settings.get("seeds_per_pillar", self.DEFAULT_SEEDS_PER_PILLAR)
        api_calls_limit = settings.get("api_calls_per_step", self.DEFAULT_API_CALLS_PER_STEP)
        logger.info("Starting keyword expansion", extra={"project_id": input_data.project_id, "expansion_budget": expansion_budget, "seeds_per_pillar": seeds_per_pillar, "api_calls_limit": api_calls_limit})

        # Get locale for API calls
        locale = project.primary_locale or "en-US"
        location_code = get_location_code(locale)
        language_code = project.primary_language or "en"

        await self._update_progress(5, "Loading seed topics...")

        # Load seed topics, grouped by pillar
        seeds_result = await self.session.execute(
            select(SeedTopic).where(SeedTopic.project_id == input_data.project_id)
        )
        all_seeds = list(seeds_result.scalars())

        # Group seeds by pillar and select top N per pillar
        pillar_seeds: dict[str, list[SeedTopic]] = {}
        for seed in all_seeds:
            pillar = seed.pillar_type or "default"
            if pillar not in pillar_seeds:
                pillar_seeds[pillar] = []
            pillar_seeds[pillar].append(seed)

        # Select top seeds per pillar (by relevance score)
        selected_seeds = []
        for pillar, seeds in pillar_seeds.items():
            sorted_seeds = sorted(
                seeds,
                key=lambda s: s.relevance_score or 0,
                reverse=True
            )
            selected_seeds.extend(sorted_seeds[:seeds_per_pillar])

        seeds_skipped = len(all_seeds) - len(selected_seeds)
        logger.info("Seeds selected for expansion", extra={"selected_count": len(selected_seeds), "skipped_count": seeds_skipped})

        await self._update_progress(10, f"Selected {len(selected_seeds)} seeds for expansion...")

        # Prepare exclusion filters
        out_of_scope = set()
        if brand:
            out_of_scope = set(t.lower() for t in (brand.out_of_scope_topics or []))

        # Collect all seed phrases for batched API calls
        seed_phrases = [seed.name for seed in selected_seeds]

        all_keywords: list[dict[str, Any]] = []
        seen_keywords: set[str] = set()
        api_calls_made = 0

        async with DataForSEOClient() as client:
            # Batch keyword suggestions (1 API call for all seeds)
            await self._update_progress(20, f"Fetching keyword suggestions for {len(seed_phrases)} seeds...")
            try:
                suggestions = await client.get_keyword_suggestions(
                    seeds=seed_phrases,
                    location_code=location_code,
                    language_code=language_code,
                    limit=expansion_budget,
                )
                api_calls_made += 1

                for kw in suggestions:
                    kw_text = kw.get("keyword", "")
                    kw_normalized = kw_text.lower().strip()

                    if self._should_include_keyword(kw_normalized, seen_keywords, out_of_scope):
                        seen_keywords.add(kw_normalized)
                        all_keywords.append({
                            "keyword_text": kw_text,
                            "keyword_normalized": kw_normalized,
                            "source_seed_topic": None,
                            "seed_topic_id": None,
                            "source_method": "suggestion",
                            "search_volume": kw.get("search_volume"),
                            "cpc": kw.get("cpc"),
                            "competition": kw.get("competition"),
                            "exclusion_flags": [],
                        })
            except Exception:
                logger.warning("Keyword suggestion API error", extra={"seeds": len(seed_phrases), "method": "suggestion"})

            # Batch question keywords (1 API call for all seeds)
            if len(all_keywords) < expansion_budget:
                await self._update_progress(60, f"Fetching question keywords for {len(seed_phrases)} seeds...")
                try:
                    questions = await client.get_keyword_questions(
                        keywords=seed_phrases,
                        location_code=location_code,
                        language_code=language_code,
                        limit=expansion_budget // 5,
                    )
                    api_calls_made += 1

                    for kw in questions:
                        kw_text = kw.get("keyword", "")
                        kw_normalized = kw_text.lower().strip()

                        if self._should_include_keyword(kw_normalized, seen_keywords, out_of_scope):
                            seen_keywords.add(kw_normalized)
                            all_keywords.append({
                                "keyword_text": kw_text,
                                "keyword_normalized": kw_normalized,
                                "source_seed_topic": None,
                                "seed_topic_id": None,
                                "source_method": "questions",
                                "search_volume": kw.get("search_volume"),
                                "cpc": kw.get("cpc"),
                                "competition": None,
                                "exclusion_flags": [],
                            })
                except Exception:
                    logger.warning("Question keywords API error", extra={"seeds": len(seed_phrases), "method": "questions"})

        await self._update_progress(95, f"Processed {len(all_keywords)} keywords...")

        budget_remaining = max(0, expansion_budget - len(all_keywords))
        logger.info("Keyword expansion complete", extra={"keywords_generated": len(all_keywords), "api_calls_made": api_calls_made, "budget_remaining": budget_remaining})

        await self._update_progress(100, "Keyword expansion complete")

        return ExpansionOutput(
            keywords_generated=len(all_keywords),
            api_calls_made=api_calls_made,
            budget_remaining=budget_remaining,
            seeds_processed=len(selected_seeds),
            seeds_skipped=seeds_skipped,
            keywords=all_keywords,
        )

    async def _validate_output(self, result: ExpansionOutput, input_data: ExpansionInput) -> None:
        """Ensure output can be consumed by Step 4."""
        if result.seeds_processed <= 0:
            raise ValueError(
                "Step 3 processed 0 seeds. Step 4 requires expanded keywords sourced from seeds."
            )

        if result.keywords_generated <= 0:
            raise ValueError(
                "Step 3 generated 0 keywords. Step 4 requires at least one active keyword."
            )

    def _should_include_keyword(
        self,
        keyword_normalized: str,
        seen: set[str],
        out_of_scope: set[str],
    ) -> bool:
        """Check if keyword should be included."""
        # Skip if already seen
        if keyword_normalized in seen:
            return False

        # Skip if empty
        if not keyword_normalized or len(keyword_normalized) < 2:
            return False

        # Skip if out of scope
        for exclusion in out_of_scope:
            if exclusion in keyword_normalized:
                return False

        # Skip obvious spam/adult content patterns
        spam_patterns = [
            "xxx", "porn", "nude", "naked", "sex",
            "casino", "gambling", "slot machine",
        ]
        for pattern in spam_patterns:
            if pattern in keyword_normalized:
                return False

        return True

    async def _persist_results(self, result: ExpansionOutput) -> None:
        """Save expanded keywords to database."""
        # Delete existing expansion keywords for this project
        existing = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == self.project_id,
                Keyword.source == "expansion",
            )
        )
        for keyword in existing.scalars():
            await delete(self.session, Keyword, keyword)

        # Create new keywords
        for kw_data in result.keywords:
            create(
                self.session,
                Keyword,
                KeywordCreateDTO(
                    project_id=self.project_id,
                    keyword=kw_data["keyword_text"],
                    keyword_normalized=kw_data["keyword_normalized"],
                    source="expansion",
                    seed_topic_id=kw_data.get("seed_topic_id"),
                    search_volume=kw_data.get("search_volume"),
                    cpc=kw_data.get("cpc"),
                    competition=kw_data.get("competition"),
                    status="active",
                ),
            )

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 3

        # Set result summary
        self.set_result_summary({
            "keywords_generated": result.keywords_generated,
            "api_calls_made": result.api_calls_made,
            "budget_remaining": result.budget_remaining,
            "seeds_processed": result.seeds_processed,
            "seeds_skipped": result.seeds_skipped,
        })

        await self.session.commit()
