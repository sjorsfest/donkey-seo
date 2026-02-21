"""Step 2: Seed Keyword Generation.

Generates 20-50 seed keywords organized into buckets based on brand profile.
"""

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.agents.topic_generator import TopicGeneratorAgent, TopicGeneratorInput
from app.models.brand import BrandProfile
from app.models.generated_dtos import SeedTopicCreateDTO
from app.models.keyword import SeedTopic
from app.models.project import Project
from app.services.market_diagnosis import (
    collect_known_entities,
    diagnose_market_mode,
)
from app.services.discovery_capabilities import CAPABILITY_SEED_GENERATION
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
    market_mode: str


class Step02SeedsService(BaseStepService[SeedsInput, SeedsOutput]):
    """Step 2: Seed Keyword Generation.

    Uses TopicGeneratorAgent to create seed keyword buckets and seed keywords
    based on the brand profile from Step 1.
    """

    step_number = 1
    step_name = "seed_keywords"
    capability_key = CAPABILITY_SEED_GENERATION
    is_optional = False
    MAX_SEED_WORDS = 4
    MAX_WORKFLOW_ENTITIES = 4
    MAX_WORKFLOW_SYNTHETIC_SEEDS = 24
    WORKFLOW_ENTITY_STOPWORDS = {
        "best",
        "cheap",
        "affordable",
        "free",
        "support",
        "customer",
        "service",
        "services",
        "software",
        "tool",
        "tools",
        "platform",
        "system",
        "systems",
    }

    async def _validate_preconditions(self, input_data: SeedsInput) -> None:
        """Validate Step 1 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        # Check brand profile exists
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        if not brand_result.scalar_one_or_none():
            raise ValueError("Brand profile not found. Run setup pipeline first.")

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
        diagnosis = diagnose_market_mode(
            source="step2_initial",
            override=getattr(strategy, "market_mode_override", "auto"),
            seed_terms=in_scope_topics + offer_categories + buyer_jobs,
        )
        await self.set_market_diagnosis(diagnosis.to_dict())
        market_mode = diagnosis.mode
        learning_context = await self.build_learning_context(
            self.capability_key,
            "TopicGeneratorAgent",
        )

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
            learning_context=learning_context,
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
        seeds = self._sanitize_seed_candidates(
            seeds=seeds,
            out_of_scope_topics=out_of_scope_topics,
        )

        # Add workflow expansion only for explicitly fragmented markets.
        # Mixed markets are already broad and tend to drift when synthetic combinations dominate.
        if market_mode == "fragmented_workflow":
            buckets, seeds = self._union_workflow_seed_expansion(
                brand=brand,
                strategy=strategy,
                buckets=buckets,
                seeds=seeds,
            )
            seeds = self._sanitize_seed_candidates(
                seeds=seeds,
                out_of_scope_topics=out_of_scope_topics,
            )

        await self._update_progress(100, "Seed keyword generation complete")

        return SeedsOutput(
            buckets_created=len(buckets),
            seeds_created=len(seeds),
            buckets=buckets,
            seeds=seeds,
            known_gaps=output.known_gaps,
            market_mode=market_mode,
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
        project.current_step = max(project.current_step, self.step_number)

        # Set result summary
        self.set_result_summary({
            "buckets_created": result.buckets_created,
            "seeds_created": result.seeds_created,
            "known_gaps_count": len(result.known_gaps),
            "bucket_names": [b["name"] for b in result.buckets],
            "market_mode": result.market_mode,
        })

        await self.session.commit()

    def _union_workflow_seed_expansion(
        self,
        *,
        brand: BrandProfile,
        strategy: Any,
        buckets: list[dict[str, Any]],
        seeds: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Union workflow/integration/replacement seed patterns without removing existing seeds."""
        workflow_bucket_name = "Workflow Integrations"
        if workflow_bucket_name not in {bucket.get("name") for bucket in buckets}:
            buckets.append({
                "name": workflow_bucket_name,
                "description": "Integration, automation, workflow, and replacement intent seeds",
                "icp_relevance": "Captures fragmented workflow demand and conversion-ready queries",
                "product_tie_in": "Bridges product capabilities to real implementation workflows",
            })

        known_entities = self._workflow_entities(
            brand=brand,
            include_topics=list(getattr(strategy, "include_topics", [])),
        )
        incumbents = self._extract_incumbents(brand)
        generated_keywords: list[str] = []

        # Verb/workflow seeds.
        for verb in [
            "connect",
            "integrate",
            "sync",
            "send",
            "forward",
            "route",
            "webhook",
            "automate",
            "notify",
            "import",
            "export",
        ]:
            if known_entities:
                for entity in known_entities[: self.MAX_WORKFLOW_ENTITIES]:
                    generated_keywords.append(f"{verb} {entity}")
            else:
                generated_keywords.append(verb)

        # Entity-pair seeds.
        for entity_a, entity_b in itertools.combinations(known_entities[: self.MAX_WORKFLOW_ENTITIES], 2):
            generated_keywords.extend([
                f"{entity_a} to {entity_b}",
                f"{entity_a} integration {entity_b}",
            ])

        # Replacement and comparison seeds.
        for incumbent in incumbents[:6]:
            generated_keywords.extend([
                f"{incumbent} alternative",
                f"replace {incumbent}",
                f"{incumbent} without code",
                f"{incumbent} without agents",
            ])

        # Adjacent automation seeds.
        generated_keywords.extend([
            "notification automation",
            "workflow automation",
            "api integration",
            "webhook automation",
            "integration bot",
        ])

        existing_keywords = {str(seed.get("keyword", "")).strip().lower() for seed in seeds}
        added_count = 0
        for keyword in generated_keywords:
            cleaned = re.sub(r"\s+", " ", keyword.strip())
            normalized = cleaned.lower()
            if (
                not cleaned
                or normalized in existing_keywords
                or not self._is_seed_keyword_shape(cleaned)
            ):
                continue
            existing_keywords.add(normalized)
            seeds.append({
                "keyword": cleaned,
                "bucket_name": workflow_bucket_name,
                "relevance_score": 0.62,
            })
            added_count += 1
            if added_count >= self.MAX_WORKFLOW_SYNTHETIC_SEEDS:
                break

        return buckets, seeds

    def _extract_incumbents(self, brand: BrandProfile) -> list[str]:
        """Extract incumbent competitor terms for replacement seeds."""
        incumbents: list[str] = []
        for competitor in brand.competitor_positioning or []:
            name = (competitor.get("name") or competitor.get("brand") or "").strip().lower()
            if name:
                incumbents.append(name)
        return self._dedupe_topics(incumbents)

    def _sanitize_seed_candidates(
        self,
        *,
        seeds: list[dict[str, Any]],
        out_of_scope_topics: list[str],
    ) -> list[dict[str, Any]]:
        """Enforce deterministic seed quality and remove obvious drift seeds."""
        sanitized: list[dict[str, Any]] = []
        seen: set[str] = set()
        out_of_scope = {
            re.sub(r"\s+", " ", topic.strip().lower())
            for topic in out_of_scope_topics
            if topic and topic.strip()
        }

        for seed in seeds:
            keyword = re.sub(r"\s+", " ", str(seed.get("keyword") or "").strip())
            normalized = keyword.lower()
            if (
                not keyword
                or normalized in seen
                or not self._is_seed_keyword_shape(keyword)
            ):
                continue
            if any(exclusion in normalized for exclusion in out_of_scope):
                continue
            seen.add(normalized)
            sanitized.append({
                "keyword": keyword,
                "bucket_name": str(seed.get("bucket_name") or "General"),
                "relevance_score": float(seed.get("relevance_score") or 0.5),
            })
        return sanitized

    def _is_seed_keyword_shape(self, keyword: str) -> bool:
        """Return True when a seed keyword matches shape constraints."""
        words = [w for w in keyword.split() if w]
        if not words or len(words) > self.MAX_SEED_WORDS:
            return False
        if keyword.count("?") > 0:
            return False
        if re.search(r"\b(lorem ipsum|test keyword|dummy)\b", keyword.lower()):
            return False
        return True

    def _workflow_entities(
        self,
        *,
        brand: BrandProfile,
        include_topics: list[str],
    ) -> list[str]:
        """Collect compact entities for workflow seed generation."""
        entities = collect_known_entities(
            brand=brand,
            seed_terms=include_topics,
        )
        compact: list[str] = []
        for entity in sorted(entities):
            cleaned = re.sub(r"\s+", " ", entity.strip().lower())
            words = [word for word in cleaned.split() if word]
            if not words or len(words) > 3:
                continue
            if all(word in self.WORKFLOW_ENTITY_STOPWORDS for word in words):
                continue
            compact.append(cleaned)
        return compact
