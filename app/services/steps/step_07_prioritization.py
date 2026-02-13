"""Step 7: Topic Prioritization.

Ranks topic backlog by business value and achievability.
Uses per-project weights with auditable explanations.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prioritization_agent import (
    PrioritizationAgent,
    PrioritizationAgentInput,
)
from app.models.brand import BrandProfile
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.steps.base_step import BaseStepService, StepResult

logger = logging.getLogger(__name__)


@dataclass
class PrioritizationInput:
    """Input for Step 7."""

    project_id: str


@dataclass
class PrioritizationOutput:
    """Output from Step 7."""

    topics_ranked: int
    weights_used: dict[str, float]
    topics: list[dict[str, Any]] = field(default_factory=list)
    strategy_notes: str = ""


# Default priority weights (can be overridden per-project)
DEFAULT_PRIORITY_WEIGHTS = {
    "demand": 0.30,  # Search volume opportunity
    "achievability": 0.25,  # Inverse of difficulty
    "business_alignment": 0.30,  # Relevance to products/money pages
    "authority": 0.15,  # Supports topical authority building
}

# Low coherence threshold - triggers Step 8 SERP validation
LOW_COHERENCE_THRESHOLD = 0.7


class Step07PrioritizationService(BaseStepService[PrioritizationInput, PrioritizationOutput]):
    """Step 7: Topic Prioritization.

    Scoring approach:
    1. Calculate demand opportunity: volume / competition
    2. Calculate achievability: inverse of avg keyword difficulty
    3. Calculate business alignment: topic relevance to products/money pages
    4. Calculate authority value: supports pillar topics
    5. Apply weighted formula using PROJECT-SPECIFIC weights
    6. Generate auditable explanations

    Key features:
    - Per-project weight customization (different business types need different weights)
    - Auditable scoring with top factors and limiting factors
    - Data quality tracking (incomplete metrics flag)
    """

    step_number = 7
    step_name = "prioritization"
    is_optional = False

    # Score normalization ranges
    DIFFICULTY_MAX = 100.0
    VOLUME_REFERENCE = 10000  # For normalization
    LLM_BATCH_SIZE = 15  # Topics per LLM call

    async def _validate_preconditions(self, input_data: PrioritizationInput) -> None:
        """Validate Step 6 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 6:
            raise ValueError("Step 6 (Clustering) must be completed first")

        # Check topics exist
        topics_result = await self.session.execute(
            select(Topic).where(Topic.project_id == input_data.project_id).limit(1)
        )
        if not topics_result.scalars().first():
            raise ValueError("No topics found. Run Step 6 first.")

    async def _execute(self, input_data: PrioritizationInput) -> PrioritizationOutput:
        """Execute topic prioritization."""
        # Load project and settings
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()

        # Load brand profile for context
        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()

        await self._update_progress(5, "Loading topics...")

        # Load all topics with their keywords
        topics_result = await self.session.execute(
            select(Topic).where(Topic.project_id == input_data.project_id)
        )
        all_topics = list(topics_result.scalars())

        # Get per-project priority weights early for logging
        weights = self._get_priority_weights(project)
        logger.info("Prioritization starting", extra={"project_id": input_data.project_id, "topic_count": len(all_topics), "weights": weights})

        if not all_topics:
            return PrioritizationOutput(
                topics_ranked=0,
                weights_used=DEFAULT_PRIORITY_WEIGHTS,
            )

        await self._update_progress(10, f"Scoring {len(all_topics)} topics...")

        # Prepare brand context
        brand_context = self._build_brand_context(brand)
        money_pages = self._extract_money_pages(brand)
        primary_goal = project.primary_goal or ""

        # Calculate priority scores for all topics
        scored_topics = []
        for i, topic in enumerate(all_topics):
            progress = 10 + int((i / len(all_topics)) * 40)
            if i % 10 == 0:
                await self._update_progress(
                    progress,
                    f"Scoring topic {i + 1}/{len(all_topics)}..."
                )

            # Load keywords for this topic
            keywords_result = await self.session.execute(
                select(Keyword).where(Keyword.topic_id == topic.id)
            )
            keywords = list(keywords_result.scalars())

            # Calculate factor scores
            factors = self._calculate_factors(topic, keywords, brand, money_pages)

            # Calculate weighted priority score
            priority_score = self._calculate_priority_score(factors, weights)

            # Generate auditable explanation
            explanation = self._generate_explanation(factors, keywords)

            scored_topics.append({
                "topic": topic,
                "keywords": keywords,
                "priority_score": priority_score,
                "priority_factors": factors,
                "explanation": explanation,
            })

        await self._update_progress(50, "Sorting by priority...")

        # Sort by priority score
        scored_topics.sort(key=lambda x: x["priority_score"], reverse=True)

        await self._update_progress(55, "Validating with LLM...")

        # Validate and enhance with LLM (in batches)
        agent = PrioritizationAgent()
        all_prioritizations = []

        for batch_start in range(0, len(scored_topics), self.LLM_BATCH_SIZE):
            batch_end = min(batch_start + self.LLM_BATCH_SIZE, len(scored_topics))
            batch = scored_topics[batch_start:batch_end]

            progress = 55 + int((batch_start / len(scored_topics)) * 35)
            await self._update_progress(
                progress,
                f"Validating batch {batch_start // self.LLM_BATCH_SIZE + 1}..."
            )

            try:
                # Prepare topics for agent
                agent_topics = []
                for st in batch:
                    topic = st["topic"]
                    keywords = st["keywords"]
                    primary_kw = next(
                        (kw for kw in keywords if kw.id == topic.primary_keyword_id),
                        keywords[0] if keywords else None
                    )
                    agent_topics.append({
                        "name": topic.name,
                        "primary_keyword": primary_kw.keyword if primary_kw else "",
                        "dominant_intent": topic.dominant_intent,
                        "funnel_stage": topic.funnel_stage,
                        "total_volume": topic.total_volume or 0,
                        "avg_difficulty": topic.avg_difficulty or 0,
                        "keyword_count": len(keywords),
                        "priority_score": st["priority_score"],
                        "priority_factors": st["priority_factors"],
                    })

                agent_input = PrioritizationAgentInput(
                    topics=agent_topics,
                    brand_context=brand_context,
                    money_pages=money_pages,
                    primary_goal=primary_goal,
                )
                output = await agent.run(agent_input)

                # Match prioritizations to topics
                for j, prioritization in enumerate(output.prioritizations):
                    topic_idx = batch_start + j
                    if topic_idx < len(scored_topics):
                        all_prioritizations.append({
                            "topic_idx": topic_idx,
                            "expected_role": prioritization.expected_role,
                            "recommended_url_type": prioritization.recommended_url_type,
                            "target_money_pages": prioritization.target_money_pages,
                            "validation_notes": prioritization.validation_notes,
                        })

                # Store strategy notes from first batch
                if batch_start == 0:
                    strategy_notes = output.overall_strategy_notes

            except Exception:
                logger.warning("Prioritization LLM batch failed, using fallback", extra={"batch_start": batch_start})
                # Fallback: assign defaults based on metrics
                for j, st in enumerate(batch):
                    topic = st["topic"]
                    topic_idx = batch_start + j
                    all_prioritizations.append({
                        "topic_idx": topic_idx,
                        "expected_role": self._infer_role(st),
                        "recommended_url_type": self._infer_url_type(topic),
                        "target_money_pages": money_pages[:2] if money_pages else [],
                        "validation_notes": "Auto-assigned (LLM unavailable)",
                    })

        await self._update_progress(90, "Finalizing rankings...")

        # Build final output with all data
        output_topics = []
        for rank, st in enumerate(scored_topics, 1):
            topic = st["topic"]
            prioritization = next(
                (p for p in all_prioritizations if p["topic_idx"] == rank - 1),
                None
            )

            # Determine if SERP validation needed
            needs_serp_validation = (
                (topic.cluster_coherence or 1.0) < LOW_COHERENCE_THRESHOLD
            )

            output_topics.append({
                "topic_id": str(topic.id),
                "name": topic.name,
                "priority_rank": rank,
                "priority_score": st["priority_score"],
                "priority_factors": st["priority_factors"],
                "score_explanation": st["explanation"],
                "expected_role": prioritization["expected_role"] if prioritization else "authority_builder",
                "recommended_url_type": prioritization["recommended_url_type"] if prioritization else "blog",
                "target_money_pages": prioritization["target_money_pages"] if prioritization else [],
                "needs_serp_validation": needs_serp_validation,
                "validation_notes": prioritization["validation_notes"] if prioritization else "",
            })

        logger.info("Prioritization complete", extra={"topics_ranked": len(output_topics)})

        await self._update_progress(100, "Prioritization complete")

        return PrioritizationOutput(
            topics_ranked=len(output_topics),
            weights_used=weights,
            topics=output_topics,
            strategy_notes=strategy_notes if 'strategy_notes' in dir() else "",
        )

    async def _validate_output(
        self,
        result: PrioritizationOutput,
        input_data: PrioritizationInput,
    ) -> None:
        """Ensure output can be consumed by Step 12."""
        if result.topics_ranked <= 0:
            raise ValueError(
                "Step 7 ranked 0 topics. Step 12 requires at least one prioritized topic."
            )

    def _get_priority_weights(self, project: Project) -> dict[str, float]:
        """Get priority weights from project settings or use defaults."""
        # Check if project has custom weights in api_budget_caps (reusing field)
        # TODO: Add dedicated priority_weights field to Project model
        if project.api_budget_caps and "priority_weights" in project.api_budget_caps:
            custom = project.api_budget_caps["priority_weights"]
            # Merge with defaults (in case some weights are missing)
            return {**DEFAULT_PRIORITY_WEIGHTS, **custom}
        return DEFAULT_PRIORITY_WEIGHTS.copy()

    def _build_brand_context(self, brand: BrandProfile | None) -> str:
        """Build brand context string for LLM."""
        if not brand:
            return ""

        parts = []
        if brand.company_name:
            parts.append(f"Company: {brand.company_name}")
        if brand.products_services:
            products = [p.get("name", "") for p in brand.products_services[:5]]
            parts.append(f"Products: {', '.join(products)}")
        if brand.unique_value_props:
            parts.append(f"Value Props: {', '.join(brand.unique_value_props[:3])}")

        return "\n".join(parts)

    def _extract_money_pages(self, brand: BrandProfile | None) -> list[str]:
        """Extract money page URLs from brand profile."""
        if not brand or not brand.money_pages:
            return []
        return [mp.get("url", "") for mp in brand.money_pages if mp.get("url")]

    def _calculate_factors(
        self,
        topic: Topic,
        keywords: list[Keyword],
        brand: BrandProfile | None,
        money_pages: list[str],
    ) -> dict[str, float]:
        """Calculate individual priority factors (0-1 normalized)."""
        # Demand score: based on total volume
        total_volume = topic.total_volume or sum(kw.search_volume or 0 for kw in keywords)
        demand_score = min(total_volume / self.VOLUME_REFERENCE, 1.0)

        # Achievability score: inverse of difficulty
        avg_difficulty = topic.avg_difficulty
        if avg_difficulty is None:
            difficulties = [kw.difficulty for kw in keywords if kw.difficulty is not None]
            avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 50.0
        achievability_score = 1.0 - (avg_difficulty / self.DIFFICULTY_MAX)
        achievability_score = max(0.0, min(1.0, achievability_score))

        # Business alignment: based on intent and funnel stage
        business_alignment = self._calculate_business_alignment(
            topic, keywords, brand, money_pages
        )

        # Authority score: supports pillar topics
        authority_score = self._calculate_authority_score(topic)

        return {
            "demand": demand_score,
            "achievability": achievability_score,
            "business_alignment": business_alignment,
            "authority": authority_score,
        }

    def _calculate_business_alignment(
        self,
        topic: Topic,
        keywords: list[Keyword],
        brand: BrandProfile | None,
        money_pages: list[str],
    ) -> float:
        """Calculate how well topic aligns with business goals."""
        score = 0.5  # Base score

        # Boost for commercial/transactional intent
        intent = topic.dominant_intent or ""
        if intent == "transactional":
            score += 0.3
        elif intent == "commercial":
            score += 0.2
        elif intent == "navigational":
            score += 0.1

        # Boost for BOFU/MOFU funnel stage
        funnel = topic.funnel_stage or ""
        if funnel == "bofu":
            score += 0.2
        elif funnel == "mofu":
            score += 0.1

        # Boost if topic relates to products (check overlap with product names)
        if brand and brand.products_services:
            product_names = [p.get("name", "").lower() for p in brand.products_services]
            topic_name_lower = topic.name.lower()
            if any(pn in topic_name_lower for pn in product_names if pn):
                score += 0.15

        return min(1.0, score)

    def _calculate_authority_score(self, topic: Topic) -> float:
        """Calculate how well topic supports topical authority."""
        score = 0.5  # Base score

        # Boost for informational content (builds authority)
        if topic.dominant_intent == "informational":
            score += 0.2

        # Boost for TOFU content (awareness, education)
        if topic.funnel_stage == "tofu":
            score += 0.15

        # Boost if it's a pillar topic (has seed topic reference)
        if topic.pillar_seed_topic_id:
            score += 0.15

        # Boost for high keyword count (comprehensive topic)
        if topic.keyword_count and topic.keyword_count >= 10:
            score += 0.1

        return min(1.0, score)

    def _calculate_priority_score(
        self,
        factors: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Calculate weighted priority score (0-100)."""
        total = 0.0
        for factor_name, factor_value in factors.items():
            weight = weights.get(factor_name, 0.0)
            total += factor_value * weight

        # Normalize to 0-100 scale
        return round(total * 100, 2)

    def _generate_explanation(
        self,
        factors: dict[str, float],
        keywords: list[Keyword],
    ) -> dict[str, Any]:
        """Generate auditable explanation for the priority score."""
        # Sort factors by value
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)

        # Top factors (highest contributors)
        top_factors = []
        for name, value in sorted_factors[:2]:
            if name == "demand":
                total_vol = sum(kw.search_volume or 0 for kw in keywords)
                top_factors.append(f"Demand: {total_vol:,}/mo volume")
            elif name == "achievability":
                avg_diff = sum(kw.difficulty or 50 for kw in keywords) / max(len(keywords), 1)
                top_factors.append(f"Achievability: {avg_diff:.0f} avg difficulty")
            elif name == "business_alignment":
                top_factors.append(f"Business fit: {value:.0%} aligned")
            elif name == "authority":
                top_factors.append(f"Authority: {value:.0%} value")

        # Limiting factors (lowest contributors)
        limiting_factors = []
        for name, value in sorted_factors[-2:]:
            if value < 0.5:
                if name == "demand":
                    limiting_factors.append(f"Low volume")
                elif name == "achievability":
                    limiting_factors.append(f"High difficulty")
                elif name == "business_alignment":
                    limiting_factors.append(f"Weak business fit")
                elif name == "authority":
                    limiting_factors.append(f"Low authority value")

        # Data quality: check for missing metrics
        missing_metrics = 0
        for kw in keywords:
            if kw.search_volume is None:
                missing_metrics += 1
            if kw.difficulty is None:
                missing_metrics += 1

        total_possible = len(keywords) * 2  # volume + difficulty per keyword
        data_quality = 1.0 - (missing_metrics / max(total_possible, 1))

        return {
            "top_factors": top_factors,
            "limiting_factors": limiting_factors,
            "data_quality": round(data_quality, 2),
        }

    def _infer_role(self, scored_topic: dict) -> str:
        """Infer expected role from metrics (fallback)."""
        topic = scored_topic["topic"]
        factors = scored_topic["priority_factors"]

        # Quick win: high achievability, decent demand
        if factors.get("achievability", 0) > 0.6 and factors.get("demand", 0) > 0.3:
            return "quick_win"

        # Revenue driver: high business alignment, commercial/transactional
        if factors.get("business_alignment", 0) > 0.7:
            return "revenue_driver"

        # Default: authority builder
        return "authority_builder"

    def _infer_url_type(self, topic: Topic) -> str:
        """Infer URL type from topic characteristics (fallback)."""
        page_type = topic.dominant_page_type or ""
        intent = topic.dominant_intent or ""

        if page_type in ("comparison", "alternatives"):
            return "comparison"
        elif page_type in ("landing",):
            return "landing"
        elif page_type in ("tool", "template"):
            return "resource"
        elif intent in ("transactional",):
            return "landing"
        else:
            return "blog"

    async def _persist_results(self, result: PrioritizationOutput) -> None:
        """Save priority rankings to database."""
        for topic_data in result.topics:
            topic_result = await self.session.execute(
                select(Topic).where(Topic.id == topic_data["topic_id"])
            )
            topic = topic_result.scalar_one_or_none()

            if topic:
                topic.priority_rank = topic_data["priority_rank"]
                topic.priority_score = topic_data["priority_score"]
                topic.priority_factors = {
                    **topic_data["priority_factors"],
                    "explanation": topic_data["score_explanation"],
                }
                topic.recommended_url_type = topic_data["recommended_url_type"]
                topic.recommended_publish_order = topic_data["priority_rank"]
                topic.target_money_pages = topic_data["target_money_pages"]
                topic.expected_role = topic_data["expected_role"]

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 7

        # Set result summary
        self.set_result_summary({
            "topics_ranked": result.topics_ranked,
            "weights_used": result.weights_used,
            "quick_wins": sum(1 for t in result.topics if t.get("expected_role") == "quick_win"),
            "authority_builders": sum(1 for t in result.topics if t.get("expected_role") == "authority_builder"),
            "revenue_drivers": sum(1 for t in result.topics if t.get("expected_role") == "revenue_driver"),
            "needs_serp_validation": sum(1 for t in result.topics if t.get("needs_serp_validation")),
        })

        await self.session.commit()
