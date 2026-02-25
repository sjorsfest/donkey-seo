"""Step 7: Topic Prioritization.

Ranks topic backlog by business value and achievability, then applies
project-specific fit gating before briefing.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.agents.prioritization_agent import (
    PrioritizationAgent,
    PrioritizationAgentInput,
)
from app.models.brand import BrandProfile
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.discovery.topic_overlap import (
    build_comparison_key,
    build_family_key,
    compute_topic_overlap,
    is_exact_pair_duplicate,
    is_sibling_pair,
    normalize_text_tokens,
)
from app.services.discovery_capabilities import CAPABILITY_PRIORITIZATION
from app.services.run_strategy import RunStrategy
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class PrioritizationInput:
    """Input for Step 7."""

    project_id: str


@dataclass
class PrioritizationOutput:
    """Output from Step 7."""

    topics_ranked: int
    weights_used: dict[str, Any]
    topics: list[dict[str, Any]] = field(default_factory=list)
    strategy_notes: str = ""
    quality_warnings: list[str] = field(default_factory=list)


WORKFLOW_OPPORTUNITY_WEIGHTS = {
    "difficulty_ease": 0.25,
    "mean_intent_score": 0.20,
    "serp_opportunity_signal": 0.20,
    "adjusted_volume_norm": 0.15,
    "strategic_fit": 0.10,
    "comparison_relevance": 0.10,
    "coherence": 0.00,
}
ESTABLISHED_OPPORTUNITY_WEIGHTS = {
    "difficulty_ease": 0.25,
    "mean_intent_score": 0.20,
    "serp_signal": 0.20,
    "raw_volume_norm": 0.15,
    "strategic_fit": 0.15,
    "comparison_relevance": 0.05,
}

LOW_COHERENCE_THRESHOLD = 0.7
LLM_BLEND_WEIGHT = 0.70
DFI_WORKFLOW_THRESHOLD = 0.10
EXTREME_DIFFICULTY_THRESHOLD = 90.0
BRAND_SCORE_WEIGHT = 0.70
OPPORTUNITY_SCORE_WEIGHT = 0.30
LLM_RERANK_LIMIT = 40
LLM_RERANK_MAX_DELTA = 8.0
LLM_RETRY_ATTEMPTS = 2
LLM_FIT_ADJUSTMENT_MIN = -0.15
LLM_FIT_ADJUSTMENT_MAX = 0.18
PRIMARY_PROMOTION_TOLERANCE = 0.04
SECONDARY_PROMOTION_TOLERANCE = 0.06
MIN_GAP_BETWEEN_THRESHOLDS = 0.03
DIVERSIFICATION_SOFT_OVERLAP_THRESHOLD = 0.72
DIVERSIFICATION_HARD_OVERLAP_THRESHOLD = 0.85
PRIMARY_INTENT_PAGE_CAP = 2
PRIMARY_FAMILY_CAP = 2

DYNAMIC_FIT_ALPHA = 0.55
DYNAMIC_FIT_BETA = 0.45
DYNAMIC_OPPORTUNITY_ALPHA = 0.85
DYNAMIC_OPPORTUNITY_BETA = 0.15
FINAL_DYNAMIC_FIT_WEIGHT = 0.70
FINAL_DYNAMIC_OPPORTUNITY_WEIGHT = 0.30

PROFILE_CALIBRATION: dict[str, dict[str, float]] = {
    "aggressive": {
        "q_primary": 0.80,
        "q_secondary": 0.62,
        "floor_primary": 0.34,
        "floor_secondary": 0.27,
    },
    "moderate": {
        "q_primary": 0.72,
        "q_secondary": 0.54,
        "floor_primary": 0.31,
        "floor_secondary": 0.24,
    },
    "lenient": {
        "q_primary": 0.64,
        "q_secondary": 0.46,
        "floor_primary": 0.27,
        "floor_secondary": 0.21,
    },
}


class Step07PrioritizationService(BaseStepService[PrioritizationInput, PrioritizationOutput]):
    """Step 7: Topic Prioritization with fit gating."""

    step_number = 6
    step_name = "prioritization"
    capability_key = CAPABILITY_PRIORITIZATION
    is_optional = False

    DIFFICULTY_MAX = 100.0
    LLM_BATCH_SIZE = 15
    COMPARISON_MODIFIERS = (" vs ", " versus ", " alternative", " alternatives", " comparison", " compare ")

    async def _validate_preconditions(self, input_data: PrioritizationInput) -> None:
        """Validate Step 6 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        topics_result = await self.session.execute(
            select(Topic).where(Topic.project_id == input_data.project_id).limit(1)
        )
        if not topics_result.scalars().first():
            raise ValueError("No topics found. Run Step 5 first.")

    async def _execute(self, input_data: PrioritizationInput) -> PrioritizationOutput:
        """Execute topic prioritization with hybrid scoring and fit gating."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()
        strategy = await self.get_run_strategy()

        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()

        await self._update_progress(5, "Loading topics...")

        topics_result = await self.session.execute(
            select(Topic)
            .where(Topic.project_id == input_data.project_id)
            .options(selectinload(Topic.keywords))
        )
        all_topics = list(topics_result.scalars())
        base_market_mode = await self.get_market_mode(default="mixed")

        weights_used: dict[str, Any] = {
            "fragmented_workflow": WORKFLOW_OPPORTUNITY_WEIGHTS,
            "established_category": ESTABLISHED_OPPORTUNITY_WEIGHTS,
            "brand_weight": BRAND_SCORE_WEIGHT,
            "opportunity_weight": OPPORTUNITY_SCORE_WEIGHT,
            "mixed_dfi_threshold": DFI_WORKFLOW_THRESHOLD,
        }
        logger.info(
            "Prioritization starting",
            extra={
                "project_id": input_data.project_id,
                "topic_count": len(all_topics),
                "weights": weights_used,
                "fit_profile": strategy.fit_threshold_profile,
                "market_mode": base_market_mode,
            },
        )

        if not all_topics:
            return PrioritizationOutput(
                topics_ranked=0,
                weights_used=weights_used,
            )

        await self._update_progress(10, f"Scoring {len(all_topics)} topics...")

        brand_context = self._build_brand_context(brand, strategy)
        learning_context = await self.build_learning_context(
            self.capability_key,
            "PrioritizationAgent",
        )
        if learning_context:
            brand_context = (
                f"{brand_context}\n\n{learning_context}"
                if brand_context
                else learning_context
            )
        money_pages = self._extract_money_pages(brand)
        primary_goal = project.primary_goal or ""
        raw_volume_reference = self._calculate_volume_reference(all_topics)
        adjusted_volume_reference = self._calculate_adjusted_volume_reference(all_topics)

        scored_topics: list[dict[str, Any]] = []
        for i, topic in enumerate(all_topics):
            if i % 10 == 0:
                progress = 10 + int((i / max(len(all_topics), 1)) * 40)
                await self._update_progress(progress, f"Scoring topic {i + 1}/{len(all_topics)}...")

            keywords = list(topic.keywords)
            factors = self._calculate_factors(
                topic=topic,
                keywords=keywords,
                brand=brand,
                money_pages=money_pages,
                raw_volume_reference=raw_volume_reference,
                adjusted_volume_reference=adjusted_volume_reference,
            )
            fit_assessment = self._calculate_fit_assessment(
                topic=topic,
                keywords=keywords,
                brand=brand,
                strategy=strategy,
                business_alignment_score=factors["business_alignment"],
            )
            factors["strategic_fit"] = fit_assessment["fit_score"]
            factors["fit_score"] = fit_assessment["fit_score"]

            effective_market_mode = self._resolve_topic_market_mode(
                base_market_mode=base_market_mode,
                dfi=factors["demand_fragmentation_index"],
            )
            factors["effective_market_mode"] = effective_market_mode
            factors["dfi_threshold"] = DFI_WORKFLOW_THRESHOLD

            opportunity_score = self._calculate_opportunity_score(
                topic=topic,
                factors=factors,
                effective_market_mode=effective_market_mode,
                fit_assessment=fit_assessment,
            )
            deterministic_priority_score = self._calculate_mode_aware_priority_score(
                factors=factors,
                effective_market_mode=effective_market_mode,
                brand_fit_score=fit_assessment["fit_score"],
                opportunity_score=opportunity_score,
            )
            factors["brand_fit_score"] = fit_assessment["fit_score"]
            factors["opportunity_score"] = opportunity_score
            factors["deterministic_priority_score"] = deterministic_priority_score
            factors["final_priority_score"] = deterministic_priority_score
            factors["llm_rerank_delta"] = 0.0

            explanation = self._generate_explanation(factors, keywords, fit_assessment)

            scored_topics.append({
                "topic_id": str(topic.id),
                "topic": topic,
                "keywords": keywords,
                "priority_score": deterministic_priority_score,
                "deterministic_priority_score": deterministic_priority_score,
                "scoring_factors": factors,
                "fit_assessment": fit_assessment,
                "deterministic_business_alignment": factors["business_alignment"],
                "deterministic_authority": factors["authority"],
                "explanation": explanation,
                "effective_market_mode": effective_market_mode,
            })

        self._apply_dynamic_scores(scored_topics)
        primary_threshold, secondary_threshold = self._calibrate_dynamic_thresholds(
            scored_topics,
            strategy,
        )
        deterministic_candidates = self._deterministic_prefilter(
            scored_topics=scored_topics,
            secondary_threshold=secondary_threshold,
        )
        deterministic_candidates.sort(key=lambda item: item["deterministic_priority_score"], reverse=True)

        for st in scored_topics:
            fit = st["fit_assessment"]
            fit["fit_tier"] = "excluded"
            fit["fit_threshold_used"] = secondary_threshold
            if fit.get("hard_exclusion_reason"):
                fit["final_cut_reason_code"] = "hard_exclusion"
            else:
                fit["final_cut_reason_code"] = "below_deterministic_prefilter"
            st["priority_score"] = st["deterministic_priority_score"]
            st["llm_rerank_delta"] = 0.0
            st["llm_fit_adjustment"] = 0.0
            st["llm_tier_recommendation"] = None
            st["final_cut_pool_rank"] = None

        final_cut_limit = self._final_cut_pool_limit(strategy)
        llm_candidates = deterministic_candidates[:final_cut_limit]
        for idx, st in enumerate(llm_candidates):
            st["final_cut_pool_rank"] = idx + 1
        for st in deterministic_candidates[final_cut_limit:]:
            st["fit_assessment"]["final_cut_reason_code"] = "outside_final_cut_pool"

        await self._update_progress(55, "Evaluating with LLM final cut...")

        agent = PrioritizationAgent()
        prioritizations_by_topic_id: dict[str, dict[str, Any]] = {}
        strategy_notes = ""

        for batch_start in range(0, len(llm_candidates), self.LLM_BATCH_SIZE):
            batch_end = min(batch_start + self.LLM_BATCH_SIZE, len(llm_candidates))
            batch = llm_candidates[batch_start:batch_end]

            progress = 55 + int((batch_start / max(len(llm_candidates), 1)) * 25)
            await self._update_progress(
                progress,
                f"LLM final cut batch {batch_start // self.LLM_BATCH_SIZE + 1}...",
            )

            output = await self._run_prioritization_agent_with_retry(
                agent=agent,
                batch=batch,
                brand_context=brand_context,
                money_pages=money_pages,
                primary_goal=primary_goal,
            )
            if output is None:
                logger.warning(
                    "Prioritization LLM batch failed after retries, using deterministic fallback",
                    extra={"batch_start": batch_start},
                )
                for st in batch:
                    topic = st["topic"]
                    prioritizations_by_topic_id[st["topic_id"]] = self._default_prioritization_for_topic(
                        scored_topic=st,
                        topic=topic,
                        money_pages=money_pages,
                    )
                    prioritizations_by_topic_id[st["topic_id"]]["final_cut_reason_code"] = "deterministic_fallback"
                continue

            if batch_start == 0:
                strategy_notes = output.overall_strategy_notes

            for prioritization in output.prioritizations:
                raw_topic_id = str((prioritization.topic_id or "")).strip()
                topic_id = raw_topic_id if raw_topic_id else None
                if topic_id is None or topic_id not in {item["topic_id"] for item in batch}:
                    topic_idx = prioritization.topic_index
                    if topic_idx < 0 or topic_idx >= len(batch):
                        continue
                    topic_id = batch[topic_idx]["topic_id"]
                prioritizations_by_topic_id[topic_id] = {
                    "llm_business_alignment": prioritization.llm_business_alignment,
                    "llm_business_alignment_rationale": prioritization.llm_business_alignment_rationale,
                    "llm_authority_value": prioritization.llm_authority_value,
                    "llm_authority_value_rationale": prioritization.llm_authority_value_rationale,
                    "expected_role": prioritization.expected_role,
                    "recommended_url_type": prioritization.recommended_url_type,
                    "recommended_publish_order": prioritization.recommended_publish_order,
                    "target_money_pages": prioritization.target_money_pages,
                    "validation_notes": prioritization.validation_notes,
                    "llm_tier_recommendation": prioritization.llm_tier_recommendation,
                    "llm_fit_adjustment": prioritization.llm_fit_adjustment,
                    "llm_final_cut_rationale": prioritization.llm_final_cut_rationale,
                    "recommended_primary_keyword": prioritization.recommended_primary_keyword,
                    "recommended_primary_keyword_rationale": (
                        prioritization.recommended_primary_keyword_rationale
                    ),
                }

        await self._update_progress(80, "Finalizing hybrid scores...")

        for st in llm_candidates:
            topic_id = st["topic_id"]
            topic = st["topic"]
            prioritization = prioritizations_by_topic_id.get(topic_id)
            if not prioritization:
                prioritization = self._default_prioritization_for_topic(
                    scored_topic=st,
                    topic=topic,
                    money_pages=money_pages,
                )
                prioritization["final_cut_reason_code"] = "deterministic_fallback"
                prioritizations_by_topic_id[topic_id] = prioritization

            llm_delta = self._llm_rerank_delta(
                deterministic_business_alignment=st["deterministic_business_alignment"],
                deterministic_authority=st["deterministic_authority"],
                llm_business_alignment=prioritization.get("llm_business_alignment"),
                llm_authority=prioritization.get("llm_authority_value"),
            )
            llm_fit_adjustment = self._clamp(
                self._to_float(prioritization.get("llm_fit_adjustment")) or 0.0,
                LLM_FIT_ADJUSTMENT_MIN,
                LLM_FIT_ADJUSTMENT_MAX,
            )
            adjusted_fit = self._clamp(
                float(st.get("dynamic_fit_score") or 0.0) + llm_fit_adjustment,
                0.0,
                1.0,
            )
            llm_tier = str(prioritization.get("llm_tier_recommendation") or "secondary").strip().lower()
            fit = st["fit_assessment"]

            fit_tier, final_cut_reason_code = self._resolve_final_cut_tier(
                llm_tier=llm_tier,
                adjusted_fit=adjusted_fit,
                primary_threshold=primary_threshold,
                secondary_threshold=secondary_threshold,
            )

            fit["fit_tier"] = fit_tier
            fit["fit_threshold_used"] = primary_threshold if fit_tier == "primary" else secondary_threshold
            fit["final_cut_reason_code"] = final_cut_reason_code

            final_priority_score = round(
                st["deterministic_priority_score"] + llm_delta + (llm_fit_adjustment * 12.0),
                2,
            )
            st["priority_score"] = final_priority_score
            st["llm_rerank_delta"] = llm_delta
            st["llm_fit_adjustment"] = llm_fit_adjustment
            st["llm_tier_recommendation"] = llm_tier
            st["adjusted_fit_score"] = adjusted_fit

        self._apply_zero_result_fallback(
            scored_topics=scored_topics,
            deterministic_candidates=deterministic_candidates,
            secondary_threshold=secondary_threshold,
            strategy=strategy,
        )

        diversification_summary = self._apply_diversification(
            scored_topics=scored_topics,
            primary_threshold=primary_threshold,
            secondary_threshold=secondary_threshold,
        )

        scored_topics.sort(
            key=lambda item: (
                self._fit_tier_sort_value(item["fit_assessment"].get("fit_tier")),
                -float(item.get("priority_score") or 0.0),
            )
        )

        await self._update_progress(90, "Finalizing rankings...")

        output_topics: list[dict[str, Any]] = []
        next_rank = 1

        for st in scored_topics:
            topic = st["topic"]
            topic_id = st["topic_id"]
            prioritization = prioritizations_by_topic_id.get(topic_id, {})
            keywords = st.get("keywords", [])
            fit_assessment = st["fit_assessment"]
            fit_tier = fit_assessment["fit_tier"]
            is_eligible = fit_tier in {"primary", "secondary"}
            resolved_primary_keyword = self._resolve_post_prioritization_primary_keyword(
                topic=topic,
                keywords=keywords,
                prioritization=prioritization,
            )
            resolved_primary_keyword_id = str(resolved_primary_keyword.id) if resolved_primary_keyword else None
            resolved_primary_keyword_text = resolved_primary_keyword.keyword if resolved_primary_keyword else None
            rank = next_rank if is_eligible else None
            if is_eligible:
                next_rank += 1

            needs_serp_validation = (topic.cluster_coherence or 1.0) < LOW_COHERENCE_THRESHOLD
            diagnostics = {
                "fit_reasons": fit_assessment.get("reasons", []),
                "fit_components": fit_assessment.get("components", {}),
                "fit_percentile_rank": st.get("fit_percentile_rank"),
                "opportunity_percentile_rank": st.get("opportunity_percentile_rank"),
                "calibrated_primary_threshold": primary_threshold,
                "calibrated_secondary_threshold": secondary_threshold,
                "final_cut_pool_rank": st.get("final_cut_pool_rank"),
                "llm_final_cut_rationale": prioritization.get("llm_final_cut_rationale", ""),
                "llm_rationales": {
                    "business_alignment": prioritization.get("llm_business_alignment_rationale", ""),
                    "authority": prioritization.get("llm_authority_value_rationale", ""),
                },
                "recommended_primary_keyword": prioritization.get("recommended_primary_keyword", ""),
                "recommended_primary_keyword_rationale": prioritization.get(
                    "recommended_primary_keyword_rationale",
                    "",
                ),
                "score_explanation": st["explanation"],
                "diversification": st.get("diversification", {}),
            }

            output_topics.append({
                "topic_id": topic_id,
                "name": topic.name,
                "primary_keyword_id": resolved_primary_keyword_id,
                "primary_keyword": resolved_primary_keyword_text,
                "priority_rank": rank,
                "priority_score": st["priority_score"],
                "deterministic_priority_score": st["deterministic_priority_score"],
                "final_priority_score": st["priority_score"],
                "market_mode": st["effective_market_mode"],
                "demand_fragmentation_index": st["scoring_factors"].get("demand_fragmentation_index"),
                "adjusted_volume_sum": st["scoring_factors"].get("adjusted_volume_sum"),
                "fit_tier": fit_tier,
                "fit_score": fit_assessment["fit_score"],
                "brand_fit_score": st["scoring_factors"].get("brand_fit_score"),
                "opportunity_score": st["scoring_factors"].get("opportunity_score"),
                "dynamic_fit_score": st.get("dynamic_fit_score"),
                "dynamic_opportunity_score": st.get("dynamic_opportunity_score"),
                "llm_rerank_delta": st.get("llm_rerank_delta"),
                "llm_fit_adjustment": st.get("llm_fit_adjustment"),
                "llm_tier_recommendation": st.get("llm_tier_recommendation"),
                "fit_threshold_primary": primary_threshold,
                "fit_threshold_secondary": secondary_threshold,
                "hard_exclusion_reason": fit_assessment.get("hard_exclusion_reason"),
                "final_cut_reason_code": fit_assessment.get("final_cut_reason_code"),
                "expected_role": prioritization.get("expected_role", "authority_builder") if is_eligible else "excluded",
                "recommended_url_type": prioritization.get("recommended_url_type", "blog") if is_eligible else None,
                "recommended_publish_order": (
                    prioritization.get("recommended_publish_order", rank) if is_eligible else None
                ),
                "target_money_pages": prioritization.get("target_money_pages", []) if is_eligible else [],
                "needs_serp_validation": needs_serp_validation,
                "validation_notes": prioritization.get("validation_notes", ""),
                "prioritization_diagnostics": diagnostics,
            })

        ranked_count = sum(1 for t in output_topics if t["priority_rank"] is not None)
        logger.info(
            "Prioritization complete",
            extra={
                "topics_ranked": ranked_count,
                "primary_topics": sum(1 for t in output_topics if t.get("fit_tier") == "primary"),
                "secondary_topics": sum(1 for t in output_topics if t.get("fit_tier") == "secondary"),
                "excluded_topics": sum(1 for t in output_topics if t.get("fit_tier") == "excluded"),
                "final_cut_candidates": len(llm_candidates),
                "eligible_topics_missing_primary_keyword": sum(
                    1
                    for t in output_topics
                    if t.get("fit_tier") in {"primary", "secondary"} and not t.get("primary_keyword_id")
                ),
                **diversification_summary,
            },
        )

        await self._update_progress(100, "Prioritization complete")

        return PrioritizationOutput(
            topics_ranked=ranked_count,
            weights_used=weights_used,
            topics=output_topics,
            strategy_notes=strategy_notes,
            quality_warnings=[],
        )

    async def _validate_output(self, result: PrioritizationOutput, input_data: PrioritizationInput) -> None:
        """Ensure output can be consumed by Step 12."""
        if result.topics_ranked <= 0:
            steps_config = await self.get_steps_config()
            loop_state = steps_config.get("loop_state")
            in_discovery_loop = (
                isinstance(loop_state, dict)
                and loop_state.get("mode") == "stepwise_discovery"
            )
            if in_discovery_loop:
                result.quality_warnings.append(
                    "No eligible topics passed fit gating this iteration; discovery loop will continue "
                    "to search for more viable topics."
                )
                return
            raise ValueError(
                "Step 7 ranked 0 topics after fit gating. Step 12 requires at least one eligible topic. "
                "Consider re-running from Step 2 with broader scope_mode, additional include_topics, "
                "or a more lenient fit_threshold_profile."
            )

        # Check if eligible count is below the strategy target and warn
        strategy = await self.get_run_strategy()
        target = strategy.eligible_target()
        if result.topics_ranked < target:
            result.quality_warnings.append(
                f"Only {result.topics_ranked} topics passed fit gating (target: {target}). "
                f"Consider re-running from Step 2 with: broader scope_mode (e.g. broad_education), "
                f"additional include_topics, or a more lenient fit_threshold_profile."
            )
            logger.warning(
                "Insufficient eligible topics after fit gating",
                extra={
                    "eligible": result.topics_ranked,
                    "target": target,
                },
            )

    def _get_priority_weights(self, project: Project) -> dict[str, float]:
        """Compatibility helper retained for legacy callers."""
        if project.api_budget_caps and "priority_weights" in project.api_budget_caps:
            custom = project.api_budget_caps["priority_weights"]
            return {**ESTABLISHED_OPPORTUNITY_WEIGHTS, **custom}
        return ESTABLISHED_OPPORTUNITY_WEIGHTS.copy()

    def _build_brand_context(self, brand: BrandProfile | None, strategy: RunStrategy) -> str:
        """Build brand + strategy context string for LLM."""
        parts: list[str] = []

        if brand:
            if brand.company_name:
                parts.append(f"Company: {brand.company_name}")
            if brand.products_services:
                products = [p.get("name", "") for p in brand.products_services[:5] if p.get("name")]
                if products:
                    parts.append(f"Products: {', '.join(products)}")
            if brand.unique_value_props:
                parts.append(f"Value Props: {', '.join(brand.unique_value_props[:3])}")

        if strategy.conversion_intents:
            parts.append(f"Conversion Intents: {', '.join(strategy.conversion_intents[:8])}")
        if strategy.icp_roles:
            parts.append(f"ICP Roles: {', '.join(strategy.icp_roles[:8])}")
        if strategy.icp_industries:
            parts.append(f"ICP Industries: {', '.join(strategy.icp_industries[:8])}")
        if strategy.icp_pains:
            parts.append(f"ICP Pains: {', '.join(strategy.icp_pains[:8])}")
        if strategy.include_topics:
            parts.append(f"In Scope: {', '.join(strategy.include_topics[:8])}")
        if strategy.exclude_topics:
            parts.append(f"Out of Scope: {', '.join(strategy.exclude_topics[:8])}")

        return "\n".join(parts)

    def _extract_money_pages(self, brand: BrandProfile | None) -> list[str]:
        """Extract money page URLs from brand profile."""
        if not brand or not brand.money_pages:
            return []
        return [mp.get("url", "") for mp in brand.money_pages if mp.get("url")]

    def _calculate_volume_reference(self, all_topics: list[Topic]) -> float:
        """Calculate adaptive volume reference (90th percentile)."""
        volumes = [
            topic.total_volume
            for topic in all_topics
            if topic.total_volume is not None and topic.total_volume > 0
        ]
        if not volumes:
            return 0

        volumes_sorted = sorted(volumes)
        percentile_90_idx = int(len(volumes_sorted) * 0.90)
        reference = volumes_sorted[min(percentile_90_idx, len(volumes_sorted) - 1)]
        return max(reference, 100)

    def _calculate_adjusted_volume_reference(self, all_topics: list[Topic]) -> float:
        """Calculate adaptive reference for adjusted cluster volume."""
        adjusted_volumes = [
            topic.adjusted_volume_sum
            for topic in all_topics
            if topic.adjusted_volume_sum is not None and topic.adjusted_volume_sum > 0
        ]
        if not adjusted_volumes:
            return self._calculate_volume_reference(all_topics)
        adjusted_sorted = sorted(adjusted_volumes)
        percentile_90_idx = int(len(adjusted_sorted) * 0.90)
        reference = adjusted_sorted[min(percentile_90_idx, len(adjusted_sorted) - 1)]
        return max(reference, 100)

    def _calculate_factors(
        self,
        topic: Topic,
        keywords: list[Keyword],
        brand: BrandProfile | None,
        money_pages: list[str],
        raw_volume_reference: float,
        adjusted_volume_reference: float,
    ) -> dict[str, Any]:
        """Calculate market-aware cluster factors."""
        raw_volume_sum = topic.total_volume if topic.total_volume is not None else sum(
            kw.search_volume or 0 for kw in keywords
        )
        adjusted_volume_sum = (
            topic.adjusted_volume_sum
            if topic.adjusted_volume_sum is not None
            else sum(
                kw.adjusted_volume
                if kw.adjusted_volume is not None
                else (kw.search_volume or 0)
                for kw in keywords
            )
        )

        if raw_volume_reference <= 0:
            raw_volume_norm = 0.5
        else:
            raw_volume_norm = min(raw_volume_sum / raw_volume_reference, 1.0)
        if adjusted_volume_reference <= 0:
            adjusted_volume_norm = 0.5
        else:
            adjusted_volume_norm = min(adjusted_volume_sum / adjusted_volume_reference, 1.0)

        avg_difficulty = self._topic_avg_difficulty(topic=topic, keywords=keywords)
        difficulty_ease = max(0.0, min(1.0, 1.0 - (avg_difficulty / self.DIFFICULTY_MAX)))
        mean_intent_score = self._mean_intent_score(keywords)
        business_alignment = self._calculate_business_alignment(topic, keywords, brand, money_pages)
        authority_score = self._calculate_authority_score(topic)
        serp_opportunity_signal = self._estimate_serp_opportunity_signal(topic, keywords)
        serp_signal = self._estimate_serp_signal(topic, keywords)
        dfi = self._calculate_demand_fragmentation_index(
            keyword_count=max(topic.keyword_count, len(keywords)),
            raw_volume_sum=raw_volume_sum,
        )

        return {
            "demand": raw_volume_norm,
            "achievability": difficulty_ease,
            "raw_volume_norm": raw_volume_norm,
            "adjusted_volume_norm": adjusted_volume_norm,
            "raw_volume_sum": float(raw_volume_sum),
            "adjusted_volume_sum": float(adjusted_volume_sum),
            "difficulty_ease": difficulty_ease,
            "mean_intent_score": mean_intent_score,
            "serp_opportunity_signal": serp_opportunity_signal,
            "serp_signal": serp_signal,
            "demand_fragmentation_index": dfi,
            "business_alignment": business_alignment,
            "authority": authority_score,
        }

    def _topic_avg_difficulty(self, topic: Topic, keywords: list[Keyword]) -> float:
        avg_difficulty = topic.avg_difficulty
        if avg_difficulty is None:
            difficulties = [kw.difficulty for kw in keywords if kw.difficulty is not None]
            avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 50.0
        return float(avg_difficulty)

    def _mean_intent_score(self, keywords: list[Keyword]) -> float:
        if not keywords:
            return 0.5
        scores = [kw.intent_score if kw.intent_score is not None else 0.5 for kw in keywords]
        return max(0.0, min(1.0, sum(scores) / len(scores)))

    def _calculate_demand_fragmentation_index(self, *, keyword_count: int, raw_volume_sum: float) -> float:
        """DFI = num_keywords_in_cluster / max(aggregate_volume, 1)."""
        denominator = max(raw_volume_sum, 1.0)
        return keyword_count / denominator

    def _resolve_topic_market_mode(self, *, base_market_mode: str, dfi: float) -> str:
        """Resolve per-topic mode in mixed runs using DFI."""
        if base_market_mode == "mixed":
            return "fragmented_workflow" if dfi >= DFI_WORKFLOW_THRESHOLD else "established_category"
        return base_market_mode

    def _calculate_mode_aware_priority_score(
        self,
        *,
        factors: dict[str, Any],
        effective_market_mode: str,
        brand_fit_score: float,
        opportunity_score: float,
    ) -> float:
        """Calculate final deterministic score where brand-fit dominates."""
        # Keep market-mode impact bounded: only opportunity internals vary by mode.
        score = (brand_fit_score * BRAND_SCORE_WEIGHT) + (opportunity_score * OPPORTUNITY_SCORE_WEIGHT)
        return round(score * 100, 2)

    def _calculate_opportunity_score(
        self,
        *,
        topic: Topic,
        factors: dict[str, Any],
        effective_market_mode: str,
        fit_assessment: dict[str, Any],
    ) -> float:
        """Estimate opportunity while preventing demand from overpowering fit."""
        comparison_relevance = fit_assessment["components"].get("comparison_relevance", 0.0)
        weights = (
            WORKFLOW_OPPORTUNITY_WEIGHTS
            if effective_market_mode == "fragmented_workflow"
            else ESTABLISHED_OPPORTUNITY_WEIGHTS
        )
        inputs: dict[str, float] = {
            "difficulty_ease": float(factors.get("difficulty_ease", 0.0)),
            "mean_intent_score": float(factors.get("mean_intent_score", 0.0)),
            "serp_opportunity_signal": float(factors.get("serp_opportunity_signal", 0.0)),
            "serp_signal": float(factors.get("serp_signal", 0.0)),
            "adjusted_volume_norm": float(factors.get("adjusted_volume_norm", 0.0)),
            "raw_volume_norm": float(factors.get("raw_volume_norm", 0.0)),
            "strategic_fit": float(factors.get("strategic_fit", 0.0)),
            "comparison_relevance": float(comparison_relevance),
            "coherence": float(topic.cluster_coherence if topic.cluster_coherence is not None else 0.6),
        }
        total = 0.0
        for factor_name, weight in weights.items():
            total += inputs.get(factor_name, 0.0) * weight
        return max(0.0, min(1.0, total))

    def _estimate_serp_opportunity_signal(self, topic: Topic, keywords: list[Keyword]) -> float:
        """Heuristic SERP opportunity proxy before live SERP validation."""
        coherence = topic.cluster_coherence if topic.cluster_coherence is not None else 0.6
        ugc_risk_ratio = self._risk_ratio(keywords, "ugc_dominated")
        comparison_ratio = self._ratio_from_keywords(keywords, "is_comparison")
        signal = (
            0.45
            + (ugc_risk_ratio * 0.25)
            + ((1.0 - coherence) * 0.20)
            + (comparison_ratio * 0.10)
        )
        return max(0.0, min(1.0, signal))

    def _estimate_serp_signal(self, topic: Topic, keywords: list[Keyword]) -> float:
        """Conservative SERP-quality proxy for established markets."""
        coherence = topic.cluster_coherence if topic.cluster_coherence is not None else 0.6
        mismatch_ratio = self._mismatch_ratio(keywords)
        signal = (coherence * 0.65) + ((1.0 - mismatch_ratio) * 0.35)
        return max(0.0, min(1.0, signal))

    def _risk_ratio(self, keywords: list[Keyword], flag: str) -> float:
        if not keywords:
            return 0.0
        hits = 0
        for keyword in keywords:
            risk_flags = keyword.risk_flags or []
            if flag in risk_flags:
                hits += 1
        return hits / len(keywords)

    def _ratio_from_keywords(self, keywords: list[Keyword], signal_flag: str) -> float:
        if not keywords:
            return 0.0
        hits = 0
        for keyword in keywords:
            signals = keyword.discovery_signals if isinstance(keyword.discovery_signals, dict) else {}
            if signals.get(signal_flag):
                hits += 1
        return hits / len(keywords)

    def _mismatch_ratio(self, keywords: list[Keyword]) -> float:
        if not keywords:
            return 0.0
        mismatches = 0
        total = 0
        for keyword in keywords:
            flags = keyword.serp_mismatch_flags or []
            if not isinstance(flags, list):
                continue
            total += 1
            if "intent_mismatch" in flags or "page_type_mismatch" in flags:
                mismatches += 1
        if total == 0:
            return 0.0
        return mismatches / total

    def _calculate_business_alignment(
        self,
        topic: Topic,
        keywords: list[Keyword],
        brand: BrandProfile | None,
        money_pages: list[str],
    ) -> float:
        """Deterministic business-alignment fallback."""
        score = 0.5

        intent = topic.dominant_intent or ""
        if intent == "transactional":
            score += 0.3
        elif intent == "commercial":
            score += 0.2
        elif intent == "navigational":
            score += 0.1

        funnel = topic.funnel_stage or ""
        if funnel == "bofu":
            score += 0.2
        elif funnel == "mofu":
            score += 0.1

        if money_pages:
            score += 0.05

        if brand and brand.products_services:
            product_names = [p.get("name", "").lower() for p in brand.products_services]
            topic_text = " ".join([topic.name.lower()] + [kw.keyword.lower() for kw in keywords[:8]])
            if any(pn and pn in topic_text for pn in product_names):
                score += 0.15

        return min(1.0, score)

    def _calculate_authority_score(self, topic: Topic) -> float:
        """Calculate authority value fallback."""
        score = 0.5
        if topic.dominant_intent == "informational":
            score += 0.2
        if topic.funnel_stage == "tofu":
            score += 0.15
        if topic.pillar_seed_topic_id:
            score += 0.15
        if topic.keyword_count and topic.keyword_count >= 10:
            score += 0.1
        return min(1.0, score)

    def _calculate_fit_assessment(
        self,
        topic: Topic,
        keywords: list[Keyword],
        brand: BrandProfile | None,
        strategy: RunStrategy,
        business_alignment_score: float,
    ) -> dict[str, Any]:
        """Calculate brand-first fit score and reasons for gating."""
        topic_text = self._build_topic_text(topic, keywords)
        topic_tokens = self._tokenize_text(topic_text)

        offer_terms = self._collect_offer_terms(brand, strategy)
        icp_terms = self._collect_icp_terms(strategy)

        if offer_terms:
            overlap_relevance = self._overlap_score(topic_tokens, offer_terms)
            solution_relevance = (overlap_relevance * 0.7) + (business_alignment_score * 0.3)
        else:
            overlap_relevance = business_alignment_score
            solution_relevance = business_alignment_score
        icp_relevance = self._overlap_score(topic_tokens, icp_terms) if icp_terms else 0.50
        comparison_relevance = self._comparison_relevance_score(
            topic=topic,
            keywords=keywords,
            strategy=strategy,
            topic_text=topic_text,
        )
        conversion_path = self._conversion_path_score(topic, strategy)
        serp_suitability = self._serp_intent_suitability(topic, keywords)

        brand_fit_score = (
            (solution_relevance * 0.38)
            + (icp_relevance * 0.18)
            + (conversion_path * 0.22)
            + (serp_suitability * 0.12)
            + (comparison_relevance * 0.10)
        )

        hard_exclusion_reason = self._hard_exclusion_reason(
            topic_text=topic_text,
            keywords=keywords,
            brand=brand,
            strategy=strategy,
        )
        avg_difficulty = self._topic_avg_difficulty(topic=topic, keywords=keywords)
        if (
            hard_exclusion_reason is None
            and avg_difficulty >= EXTREME_DIFFICULTY_THRESHOLD
        ):
            hard_exclusion_reason = "extreme_difficulty"

        if strategy.scope_mode == "strict" and solution_relevance < 0.35:
            brand_fit_score *= 0.8
        elif strategy.scope_mode == "broad_education" and topic.dominant_intent == "informational":
            brand_fit_score = min(1.0, brand_fit_score + 0.03)

        if hard_exclusion_reason:
            brand_fit_score = min(brand_fit_score, 0.10)

        reasons = [
            f"Solution relevance: {solution_relevance:.0%}",
            f"ICP relevance: {icp_relevance:.0%}",
            f"Conversion path: {conversion_path:.0%}",
            f"SERP/intent suitability: {serp_suitability:.0%}",
            f"Comparison relevance: {comparison_relevance:.0%}",
            f"Avg difficulty: {avg_difficulty:.1f}",
        ]
        if hard_exclusion_reason:
            reasons.append(f"Hard exclusion: {hard_exclusion_reason}")

        return {
            "fit_score": round(max(0.0, min(1.0, brand_fit_score)), 4),
            "components": {
                "solution_relevance": round(solution_relevance, 4),
                "icp_relevance": round(icp_relevance, 4),
                "conversion_path_plausibility": round(conversion_path, 4),
                "serp_intent_suitability": round(serp_suitability, 4),
                "comparison_relevance": round(comparison_relevance, 4),
                "offer_overlap": round(overlap_relevance, 4),
                "brand_fit_score": round(max(0.0, min(1.0, brand_fit_score)), 4),
            },
            "reasons": reasons,
            "hard_exclusion_reason": hard_exclusion_reason,
            "fit_tier": "excluded",
            "fit_threshold_used": None,
        }

    def _hard_exclusion_reason(
        self,
        topic_text: str,
        keywords: list[Keyword],
        brand: BrandProfile | None,
        strategy: RunStrategy,
    ) -> str | None:
        """Return policy-level hard exclusion reason, if any."""
        for excluded_topic in strategy.exclude_topics:
            if excluded_topic.lower() in topic_text:
                return "matches_excluded_topic"

        if strategy.branded_keyword_mode == "allow_all":
            return None

        own_terms, competitor_terms = self._extract_brand_terms(brand)
        full_text = " ".join(kw.keyword.lower() for kw in keywords)
        has_own_brand = self._contains_any_term(full_text, own_terms)
        has_competitor_brand = self._contains_any_term(full_text, competitor_terms)

        if has_competitor_brand and not has_own_brand:
            if strategy.branded_keyword_mode == "exclude_all":
                return "competitor_branded"
            if strategy.branded_keyword_mode == "comparisons_only" and not self._contains_comparison_modifier(full_text):
                return "competitor_branded_non_comparison"

        return None

    def _build_topic_text(self, topic: Topic, keywords: list[Keyword]) -> str:
        """Build rich topic text for relevance checks."""
        keyword_text = " ".join(kw.keyword for kw in keywords[:30])
        description = topic.description or ""
        cluster_notes = topic.cluster_notes or ""
        return f"{topic.name} {description} {cluster_notes} {keyword_text}".lower()

    def _comparison_relevance_score(
        self,
        *,
        topic: Topic,
        keywords: list[Keyword],
        strategy: RunStrategy,
        topic_text: str,
    ) -> float:
        """Score comparison relevance with brand policy awareness."""
        if strategy.branded_keyword_mode == "exclude_all":
            return 0.0
        keyword_flags_ratio = self._ratio_from_keywords(keywords, "is_comparison")
        page_type_bonus = 0.15 if (topic.dominant_page_type or "") in {"comparison", "alternatives"} else 0.0
        text_bonus = 0.10 if self._contains_comparison_modifier(topic_text) else 0.0
        score = keyword_flags_ratio + page_type_bonus + text_bonus
        return max(0.0, min(1.0, score))

    def _collect_offer_terms(self, brand: BrandProfile | None, strategy: RunStrategy) -> set[str]:
        """Collect terms describing the offering and strategic in-scope focus."""
        terms: set[str] = set()
        for topic in strategy.include_topics:
            terms.update(self._tokenize_text(topic))
        if not brand:
            return terms

        for product in brand.products_services or []:
            terms.update(self._tokenize_text(str(product.get("name") or "")))
            terms.update(self._tokenize_text(str(product.get("category") or "")))
            for benefit in product.get("core_benefits") or []:
                terms.update(self._tokenize_text(str(benefit)))
        for uvp in brand.unique_value_props or []:
            terms.update(self._tokenize_text(uvp))

        return terms

    def _collect_icp_terms(self, strategy: RunStrategy) -> set[str]:
        """Collect ICP terms used for relevance checks."""
        terms: set[str] = set()
        for source in (strategy.icp_roles, strategy.icp_industries, strategy.icp_pains):
            for value in source:
                terms.update(self._tokenize_text(value))
        return terms

    def _conversion_path_score(self, topic: Topic, strategy: RunStrategy) -> float:
        """Estimate plausibility of converting readers for configured intents."""
        intent = topic.dominant_intent or "informational"
        base_by_intent = {
            "transactional": 0.95,
            "commercial": 0.80,
            "navigational": 0.70,
            "informational": 0.45,
        }
        score = base_by_intent.get(intent, 0.5)
        funnel = topic.funnel_stage or "tofu"
        if funnel == "bofu":
            score += 0.10
        elif funnel == "mofu":
            score += 0.05

        intents_text = " ".join(i.lower() for i in strategy.conversion_intents)
        if re.search(r"(newsletter|subscribe|awareness|education)", intents_text):
            if intent == "informational":
                score += 0.15
        if re.search(r"(trial|signup|demo|contact|purchase|book)", intents_text):
            if intent in {"commercial", "transactional"}:
                score += 0.10

        return max(0.0, min(1.0, score))

    def _serp_intent_suitability(self, topic: Topic, keywords: list[Keyword]) -> float:
        """Estimate suitability using cluster coherence and keyword risk flags."""
        coherence = topic.cluster_coherence if topic.cluster_coherence is not None else 0.6
        risk_flags = 0
        for kw in keywords:
            risk_flags += len(kw.risk_flags or [])
        avg_risk = risk_flags / max(len(keywords), 1)
        penalty = min(0.3, avg_risk * 0.08)
        return max(0.2, min(1.0, coherence - penalty))

    def _apply_dynamic_scores(self, scored_topics: list[dict[str, Any]]) -> None:
        """Attach percentile-aware dynamic fit/opportunity scores."""
        fit_values = [float(st["fit_assessment"]["fit_score"]) for st in scored_topics]
        opportunity_values = [
            float(st["scoring_factors"].get("opportunity_score") or 0.0)
            for st in scored_topics
        ]
        for st in scored_topics:
            fit_score = float(st["fit_assessment"]["fit_score"])
            opportunity_score = float(st["scoring_factors"].get("opportunity_score") or 0.0)
            fit_percentile_rank = self._percentile_rank(fit_values, fit_score)
            opportunity_percentile_rank = self._percentile_rank(opportunity_values, opportunity_score)
            dynamic_fit_score = (
                (fit_score * DYNAMIC_FIT_ALPHA)
                + (fit_percentile_rank * DYNAMIC_FIT_BETA)
            )
            dynamic_opportunity_score = (
                (opportunity_score * DYNAMIC_OPPORTUNITY_ALPHA)
                + (opportunity_percentile_rank * DYNAMIC_OPPORTUNITY_BETA)
            )
            deterministic_priority_score = round(
                100
                * (
                    (dynamic_fit_score * FINAL_DYNAMIC_FIT_WEIGHT)
                    + (dynamic_opportunity_score * FINAL_DYNAMIC_OPPORTUNITY_WEIGHT)
                ),
                2,
            )
            st["fit_percentile_rank"] = round(fit_percentile_rank, 4)
            st["opportunity_percentile_rank"] = round(opportunity_percentile_rank, 4)
            st["dynamic_fit_score"] = round(dynamic_fit_score, 4)
            st["dynamic_opportunity_score"] = round(dynamic_opportunity_score, 4)
            st["deterministic_priority_score"] = deterministic_priority_score
            st["priority_score"] = deterministic_priority_score
            st["scoring_factors"]["fit_percentile_rank"] = round(fit_percentile_rank, 4)
            st["scoring_factors"]["opportunity_percentile_rank"] = round(opportunity_percentile_rank, 4)
            st["scoring_factors"]["dynamic_fit_score"] = round(dynamic_fit_score, 4)
            st["scoring_factors"]["dynamic_opportunity_score"] = round(dynamic_opportunity_score, 4)
            st["scoring_factors"]["deterministic_priority_score"] = deterministic_priority_score

    def _calibrate_dynamic_thresholds(
        self,
        scored_topics: list[dict[str, Any]],
        strategy: RunStrategy,
    ) -> tuple[float, float]:
        """Compute quantile+floor thresholds bounded by profile defaults."""
        profile = PROFILE_CALIBRATION.get(
            strategy.fit_threshold_profile,
            PROFILE_CALIBRATION["moderate"],
        )
        dynamic_fit_values = [
            float(st.get("dynamic_fit_score") or 0.0)
            for st in scored_topics
            if not st["fit_assessment"].get("hard_exclusion_reason")
        ]
        if not dynamic_fit_values:
            return strategy.base_threshold(), strategy.relaxed_threshold()
        q_primary = self._quantile(dynamic_fit_values, profile["q_primary"])
        q_secondary = self._quantile(dynamic_fit_values, profile["q_secondary"])
        primary_threshold = self._clamp(
            q_primary,
            profile["floor_primary"],
            strategy.base_threshold(),
        )
        secondary_threshold = self._clamp(
            q_secondary,
            profile["floor_secondary"],
            strategy.relaxed_threshold(),
        )
        secondary_threshold = min(
            secondary_threshold,
            max(0.0, primary_threshold - MIN_GAP_BETWEEN_THRESHOLDS),
        )
        return round(primary_threshold, 4), round(secondary_threshold, 4)

    def _deterministic_prefilter(
        self,
        *,
        scored_topics: list[dict[str, Any]],
        secondary_threshold: float,
    ) -> list[dict[str, Any]]:
        """Return non-hard-excluded near-fit candidates for LLM final cut."""
        candidates: list[dict[str, Any]] = []
        for st in scored_topics:
            fit = st["fit_assessment"]
            if fit.get("hard_exclusion_reason"):
                fit["fit_tier"] = "excluded"
                fit["final_cut_reason_code"] = "hard_exclusion"
                continue
            if float(st.get("dynamic_fit_score") or 0.0) >= secondary_threshold:
                candidates.append(st)
        return candidates

    def _quantile(self, values: list[float], q: float) -> float:
        """Return q-quantile from values in [0,1]."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * q))))
        return float(sorted_values[index])

    def _percentile_rank(self, values: list[float], value: float) -> float:
        """Return percentile rank for value in values."""
        if not values:
            return 0.0
        less_equal = sum(1 for item in values if item <= value)
        return self._clamp(less_equal / max(len(values), 1), 0.0, 1.0)

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _to_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_brand_terms(self, brand: BrandProfile | None) -> tuple[set[str], set[str]]:
        """Extract own-brand and competitor terms."""
        own_brand_terms: set[str] = set()
        competitor_terms: set[str] = set()
        if not brand:
            return own_brand_terms, competitor_terms

        if brand.company_name:
            own_brand_terms.add(brand.company_name.lower())
        for product in brand.products_services or []:
            name = (product.get("name") or "").strip().lower()
            if name:
                own_brand_terms.add(name)
        for competitor in brand.competitor_positioning or []:
            name = (competitor.get("name") or competitor.get("brand") or "").strip().lower()
            if name:
                competitor_terms.add(name)
        return own_brand_terms, competitor_terms

    def _contains_any_term(self, text: str, terms: set[str]) -> bool:
        """Return True when text contains any term."""
        for term in terms:
            if len(term) < 3:
                continue
            if term in text:
                return True
        return False

    def _contains_comparison_modifier(self, text: str) -> bool:
        """Return True when text looks like comparison/alternatives intent."""
        return any(marker in f" {text} " for marker in self.COMPARISON_MODIFIERS)

    def _tokenize_text(self, text: str) -> set[str]:
        """Tokenize free text into normalized terms."""
        return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2}

    def _overlap_score(self, topic_terms: set[str], reference_terms: set[str]) -> float:
        """Calculate F1-style overlap score for two term sets."""
        if not topic_terms or not reference_terms:
            return 0.0
        overlap = len(topic_terms & reference_terms)
        precision = overlap / max(len(topic_terms), 1)
        recall = overlap / max(len(reference_terms), 1)
        if precision + recall == 0:
            return 0.0
        return min(1.0, (2 * precision * recall) / (precision + recall))

    def _calculate_priority_score(self, factors: dict[str, float], weights: dict[str, float]) -> float:
        """Calculate weighted priority score (0-100)."""
        total = 0.0
        for factor_name, weight in weights.items():
            total += factors.get(factor_name, 0.0) * weight
        return round(total * 100, 2)

    def _final_cut_pool_limit(self, strategy: RunStrategy) -> int:
        """Return capped top-N size for LLM final cut."""
        return min(LLM_RERANK_LIMIT, max(20, strategy.eligible_target() * 4))

    def _resolve_final_cut_tier(
        self,
        *,
        llm_tier: str,
        adjusted_fit: float,
        primary_threshold: float,
        secondary_threshold: float,
    ) -> tuple[str, str]:
        """Resolve final fit tier + reason from LLM recommendation and thresholds."""
        normalized_tier = str(llm_tier or "secondary").strip().lower()
        if normalized_tier == "exclude":
            return "excluded", "llm_reject"
        if normalized_tier == "primary" and adjusted_fit >= (primary_threshold - PRIMARY_PROMOTION_TOLERANCE):
            return "primary", "llm_primary"
        if normalized_tier in {"primary", "secondary"} and adjusted_fit >= (
            secondary_threshold - SECONDARY_PROMOTION_TOLERANCE
        ):
            return "secondary", "llm_secondary"
        return "excluded", "below_threshold_after_llm"

    def _apply_zero_result_fallback(
        self,
        *,
        scored_topics: list[dict[str, Any]],
        deterministic_candidates: list[dict[str, Any]],
        secondary_threshold: float,
        strategy: RunStrategy,
    ) -> None:
        """Auto-accept a small deterministic subset when LLM final cut returns none."""
        eligible_after_final_cut = sum(
            1 for st in scored_topics if st["fit_assessment"].get("fit_tier") in {"primary", "secondary"}
        )
        if eligible_after_final_cut > 0 or not deterministic_candidates:
            return
        fallback_cap = min(3, strategy.eligible_target())
        for st in deterministic_candidates[:fallback_cap]:
            fit = st["fit_assessment"]
            if fit.get("hard_exclusion_reason"):
                continue
            fit["fit_tier"] = "secondary"
            fit["fit_threshold_used"] = secondary_threshold
            fit["final_cut_reason_code"] = "llm_zero_result_fallback"

    def _apply_diversification(
        self,
        *,
        scored_topics: list[dict[str, Any]],
        primary_threshold: float,
        secondary_threshold: float,
    ) -> dict[str, int]:
        """Suppress duplicate topics while keeping valid sibling comparison pairs."""
        summary = {
            "exact_pair_duplicates_removed": 0,
            "near_duplicate_excluded": 0,
            "overlap_demoted": 0,
            "diversity_cap_demotions": 0,
            "sibling_pairs_kept": 0,
        }

        for st in scored_topics:
            signature = self._build_diversification_signature(st)
            st["diversification"] = {
                **signature,
                "max_overlap_score": 0.0,
                "most_similar_topic_id": None,
                "relationship_type": "none",
                "action": "not_eligible",
            }

        eligible = [
            st for st in scored_topics
            if st["fit_assessment"].get("fit_tier") in {"primary", "secondary"}
        ]
        eligible.sort(key=lambda item: float(item.get("priority_score") or 0.0), reverse=True)

        kept: list[dict[str, Any]] = []
        for st in eligible:
            fit = st["fit_assessment"]
            diversification = st["diversification"]
            excluded = False
            demoted = False
            sibling_relation_seen = False

            for kept_topic in kept:
                kept_div = kept_topic["diversification"]
                relationship_type = "overlap"
                if is_exact_pair_duplicate(
                    diversification.get("comparison_key"),
                    kept_div.get("comparison_key"),
                ):
                    diversification["max_overlap_score"] = 1.0
                    diversification["most_similar_topic_id"] = kept_topic["topic_id"]
                    diversification["relationship_type"] = "exact_pair_duplicate"
                    diversification["action"] = "exclude"
                    fit["fit_tier"] = "excluded"
                    fit["fit_threshold_used"] = secondary_threshold
                    fit["final_cut_reason_code"] = "exact_pair_duplicate"
                    summary["exact_pair_duplicates_removed"] += 1
                    excluded = True
                    break

                sibling_pair = is_sibling_pair(
                    diversification.get("comparison_key"),
                    kept_div.get("comparison_key"),
                )
                if sibling_pair:
                    relationship_type = "sibling_pair"

                overlap_score = compute_topic_overlap(
                    keyword_tokens_a=diversification.get("keyword_tokens", set()),
                    keyword_tokens_b=kept_div.get("keyword_tokens", set()),
                    text_tokens_a=diversification.get("topic_tokens", set()),
                    text_tokens_b=kept_div.get("topic_tokens", set()),
                    serp_domains_a=diversification.get("serp_domains", set()),
                    serp_domains_b=kept_div.get("serp_domains", set()),
                    intent_a=(st["topic"].dominant_intent or ""),
                    intent_b=(kept_topic["topic"].dominant_intent or ""),
                    page_type_a=(st["topic"].dominant_page_type or ""),
                    page_type_b=(kept_topic["topic"].dominant_page_type or ""),
                )

                if overlap_score > float(diversification.get("max_overlap_score") or 0.0):
                    diversification["max_overlap_score"] = round(overlap_score, 4)
                    diversification["most_similar_topic_id"] = kept_topic["topic_id"]
                    diversification["relationship_type"] = relationship_type

                if sibling_pair:
                    sibling_relation_seen = True
                    continue

                if overlap_score >= DIVERSIFICATION_HARD_OVERLAP_THRESHOLD:
                    diversification["action"] = "exclude"
                    fit["fit_tier"] = "excluded"
                    fit["fit_threshold_used"] = secondary_threshold
                    fit["final_cut_reason_code"] = "near_duplicate_hard"
                    summary["near_duplicate_excluded"] += 1
                    excluded = True
                    break

                if (
                    overlap_score >= DIVERSIFICATION_SOFT_OVERLAP_THRESHOLD
                    and fit.get("fit_tier") == "primary"
                ):
                    fit["fit_tier"] = "secondary"
                    fit["fit_threshold_used"] = secondary_threshold
                    fit["final_cut_reason_code"] = "overlap_demoted"
                    diversification["action"] = "demote"
                    demoted = True

            if excluded:
                continue

            if sibling_relation_seen:
                summary["sibling_pairs_kept"] += 1
            if demoted:
                summary["overlap_demoted"] += 1
            if diversification["action"] == "not_eligible":
                diversification["action"] = "keep"
            kept.append(st)

        intent_page_counts: dict[tuple[str, str], int] = {}
        family_counts: dict[str, int] = {}
        for st in kept:
            fit = st["fit_assessment"]
            if fit.get("fit_tier") != "primary":
                continue

            topic = st["topic"]
            diversification = st["diversification"]
            intent_page_key = (
                str(topic.dominant_intent or "unknown"),
                str(topic.dominant_page_type or "unknown"),
            )
            family_key = str(diversification.get("family_key") or "family:uncategorized")
            intent_page_count = intent_page_counts.get(intent_page_key, 0)
            family_count = family_counts.get(family_key, 0)

            if intent_page_count >= PRIMARY_INTENT_PAGE_CAP or family_count >= PRIMARY_FAMILY_CAP:
                fit["fit_tier"] = "secondary"
                fit["fit_threshold_used"] = secondary_threshold
                fit["final_cut_reason_code"] = "diversity_cap_demotion"
                diversification["action"] = "demote"
                summary["diversity_cap_demotions"] += 1
                continue

            intent_page_counts[intent_page_key] = intent_page_count + 1
            family_counts[family_key] = family_count + 1

        for st in scored_topics:
            diversification = st.get("diversification", {})
            diversification.pop("topic_tokens", None)
            diversification.pop("keyword_tokens", None)
            diversification.pop("serp_domains", None)

        return summary

    def _build_diversification_signature(self, scored_topic: dict[str, Any]) -> dict[str, Any]:
        """Build topic signature used for overlap suppression."""
        topic = scored_topic["topic"]
        keywords: list[Keyword] = list(scored_topic.get("keywords") or [])
        ranked_keywords = self._rank_keywords_for_topic_prioritization(
            topic=topic,
            keywords=keywords,
        )
        ordered_keywords = ranked_keywords if ranked_keywords else keywords
        keyword_texts = [kw.keyword for kw in ordered_keywords if getattr(kw, "keyword", None)]
        primary_keyword = keyword_texts[0] if keyword_texts else ""
        topic_text = " ".join(
            [
                topic.name or "",
                topic.description or "",
                topic.cluster_notes or "",
            ]
        ).strip()
        keyword_text = " ".join(keyword_texts[:50])

        return {
            "family_key": build_family_key(topic.name or "", primary_keyword, keyword_texts),
            "comparison_key": build_comparison_key(topic.name or "", primary_keyword),
            "topic_tokens": normalize_text_tokens(topic_text),
            "keyword_tokens": normalize_text_tokens(keyword_text),
            "serp_domains": self._extract_topic_serp_domains(topic=topic, keywords=keywords),
        }

    def _resolve_primary_keyword_text(self, *, topic: Topic, keywords: list[Keyword]) -> str:
        """Resolve primary keyword text from topic keyword list."""
        if topic.primary_keyword_id:
            for keyword in keywords:
                if str(keyword.id) == str(topic.primary_keyword_id):
                    return keyword.keyword
        if keywords:
            return keywords[0].keyword
        return ""

    def _rank_keywords_for_topic_prioritization(
        self,
        *,
        topic: Topic,
        keywords: list[Keyword],
    ) -> list[Keyword]:
        """Rank topic keywords by topic semantic relevance (not just volume)."""
        if not keywords:
            return []

        topic_text = " ".join(
            [
                topic.name or "",
                topic.description or "",
                topic.cluster_notes or "",
            ]
        ).strip()
        topic_tokens = self._tokenize_text(topic_text)
        keyword_tokens_by_id: dict[str, set[str]] = {}
        token_frequency: Counter[str] = Counter()
        volume_values = [self._keyword_volume_value(kw) for kw in keywords]
        max_volume = max(volume_values) if volume_values else 1.0
        if max_volume <= 0:
            max_volume = 1.0

        for keyword in keywords:
            kw_tokens = self._tokenize_text(str(getattr(keyword, "keyword", "")))
            keyword_tokens_by_id[str(keyword.id)] = kw_tokens
            for token in kw_tokens:
                token_frequency[token] += 1

        max_token_frequency = max(token_frequency.values(), default=1)
        scored: list[tuple[float, float, float, Keyword]] = []

        for keyword in keywords:
            kw_tokens = keyword_tokens_by_id.get(str(keyword.id), set())
            overlap = self._overlap_score(kw_tokens, topic_tokens) if topic_tokens else 0.0
            representativeness = (
                (
                    sum(token_frequency.get(token, 0) for token in kw_tokens)
                    / max(len(kw_tokens), 1)
                )
                / max(max_token_frequency, 1)
                if kw_tokens else 0.0
            )
            raw_intent_score = getattr(keyword, "intent_score", None)
            intent_score = float(raw_intent_score if raw_intent_score is not None else 0.5)
            raw_difficulty = getattr(keyword, "difficulty", None)
            difficulty = float(raw_difficulty if raw_difficulty is not None else 55.0)
            difficulty_ease = max(0.0, min(1.0, 1.0 - (difficulty / 100.0)))
            volume = self._keyword_volume_value(keyword)
            volume_norm = max(0.0, min(1.0, volume / max_volume))
            raw_signals = getattr(keyword, "discovery_signals", None)
            signals = raw_signals if isinstance(raw_signals, dict) else {}
            comparison_bonus = 0.06 if bool(signals.get("is_comparison")) else 0.0
            integration_bonus = 0.04 if bool(signals.get("has_integration_term")) else 0.0

            score = (
                (overlap * 0.42)
                + (representativeness * 0.23)
                + (intent_score * 0.13)
                + (difficulty_ease * 0.10)
                + (volume_norm * 0.08)
                + comparison_bonus
                + integration_bonus
            )
            scored.append((float(score), float(overlap), float(volume), keyword))

        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored]

    def _keyword_volume_value(self, keyword: Any) -> float:
        """Resolve keyword volume value as float for ranking math."""
        adjusted_volume = getattr(keyword, "adjusted_volume", None)
        if adjusted_volume is not None:
            try:
                return float(adjusted_volume)
            except (TypeError, ValueError):
                pass
        search_volume = getattr(keyword, "search_volume", 0)
        try:
            return float(search_volume or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _resolve_post_prioritization_primary_keyword(
        self,
        *,
        topic: Topic,
        keywords: list[Keyword],
        prioritization: dict[str, Any],
    ) -> Keyword | None:
        """Resolve primary keyword after topic-level prioritization decisions."""
        if not keywords:
            return None

        suggested_keyword_text = str(
            prioritization.get("recommended_primary_keyword") or ""
        ).strip()
        if suggested_keyword_text:
            matched = self._match_keyword_by_text(
                keywords=keywords,
                candidate_text=suggested_keyword_text,
            )
            if matched is not None:
                return matched

        ranked = self._rank_keywords_for_topic_prioritization(
            topic=topic,
            keywords=keywords,
        )
        if ranked:
            return ranked[0]

        if topic.primary_keyword_id:
            for keyword in keywords:
                if str(keyword.id) == str(topic.primary_keyword_id):
                    return keyword
        return keywords[0]

    def _match_keyword_by_text(
        self,
        *,
        keywords: list[Keyword],
        candidate_text: str,
    ) -> Keyword | None:
        """Match LLM-selected keyword text to an actual keyword row."""
        normalized_candidate = " ".join(candidate_text.lower().split())
        if not normalized_candidate:
            return None

        for keyword in keywords:
            normalized_keyword = " ".join(keyword.keyword.lower().split())
            if normalized_keyword == normalized_candidate:
                return keyword

        for keyword in keywords:
            normalized_keyword = " ".join(keyword.keyword.lower().split())
            if (
                normalized_candidate in normalized_keyword
                or normalized_keyword in normalized_candidate
            ):
                return keyword
        return None

    def _extract_topic_serp_domains(self, *, topic: Topic, keywords: list[Keyword]) -> set[str]:
        """Extract SERP domains from primary/evidence keywords when available."""
        selected: list[Keyword] = []
        primary_id = str(topic.primary_keyword_id) if topic.primary_keyword_id else None
        evidence_id = str(topic.serp_evidence_keyword_id) if topic.serp_evidence_keyword_id else None
        for keyword in keywords:
            keyword_id = str(keyword.id)
            if keyword_id == primary_id or keyword_id == evidence_id:
                selected.append(keyword)
        if not selected and primary_id:
            for keyword in keywords:
                if str(keyword.id) == primary_id:
                    selected.append(keyword)
                    break

        domains: set[str] = set()
        for keyword in selected:
            for row in keyword.serp_top_results or []:
                if not isinstance(row, dict):
                    continue
                raw_domain = str(row.get("domain") or row.get("host") or "").strip().lower()
                if not raw_domain:
                    raw_url = str(row.get("url") or row.get("link") or "").strip()
                    if raw_url:
                        raw_domain = urlparse(raw_url).netloc.strip().lower()
                if raw_domain.startswith("www."):
                    raw_domain = raw_domain[4:]
                if raw_domain:
                    domains.add(raw_domain)
        return domains

    def _blend_qualitative_scores(self, deterministic_score: float, llm_score: float | None) -> float:
        """Blend deterministic and LLM qualitative scores."""
        if llm_score is None:
            return deterministic_score
        return (llm_score * LLM_BLEND_WEIGHT) + (deterministic_score * (1 - LLM_BLEND_WEIGHT))

    async def _run_prioritization_agent_with_retry(
        self,
        *,
        agent: PrioritizationAgent,
        batch: list[dict[str, Any]],
        brand_context: str,
        money_pages: list[str],
        primary_goal: str,
    ) -> Any | None:
        """Run prioritization agent with compact-mode retry."""
        for attempt in range(LLM_RETRY_ATTEMPTS):
            compact_mode = attempt > 0
            try:
                output = await agent.run(
                    PrioritizationAgentInput(
                        topics=self._build_agent_topics(batch=batch, compact_mode=compact_mode),
                        brand_context=brand_context if not compact_mode else brand_context[:600],
                        money_pages=money_pages,
                        primary_goal=primary_goal,
                        compact_mode=compact_mode,
                    )
                )
                return output
            except Exception:
                logger.warning(
                    "Prioritization LLM attempt failed",
                    extra={"attempt": attempt + 1, "compact_mode": compact_mode},
                )
        return None

    def _build_agent_topics(self, *, batch: list[dict[str, Any]], compact_mode: bool) -> list[dict[str, Any]]:
        """Build batch payload for prioritization agent."""
        agent_topics: list[dict[str, Any]] = []
        for i, st in enumerate(batch):
            topic = st["topic"]
            keywords = st["keywords"]
            ranked_keywords = self._rank_keywords_for_topic_prioritization(
                topic=topic,
                keywords=keywords,
            )
            primary_kw = ranked_keywords[0] if ranked_keywords else None
            keyword_candidates = [kw.keyword for kw in ranked_keywords[:8]]
            topic_payload: dict[str, Any] = {
                "topic_index": i,
                "topic_id": st["topic_id"],
                "name": topic.name,
                "primary_keyword": primary_kw.keyword if primary_kw else "",
                "keyword_candidates": keyword_candidates,
                "dominant_intent": topic.dominant_intent,
                "funnel_stage": topic.funnel_stage,
                "total_volume": topic.total_volume or 0,
                "avg_difficulty": topic.avg_difficulty or 0,
                "keyword_count": len(keywords),
                "priority_score": st["deterministic_priority_score"],
            }
            if compact_mode:
                topic_payload["scoring_factors"] = {
                    "fit_score": st["fit_assessment"]["fit_score"],
                    "market_mode": st["effective_market_mode"],
                }
            else:
                topic_payload["scoring_factors"] = st["scoring_factors"]
            agent_topics.append(topic_payload)
        return agent_topics

    def _default_prioritization_for_topic(
        self,
        *,
        scored_topic: dict[str, Any],
        topic: Topic,
        money_pages: list[str],
    ) -> dict[str, Any]:
        """Fallback prioritization when LLM output is missing."""
        ranked_keywords = self._rank_keywords_for_topic_prioritization(
            topic=topic,
            keywords=list(scored_topic.get("keywords") or []),
        )
        fallback_primary_keyword = ranked_keywords[0].keyword if ranked_keywords else ""
        return {
            "llm_business_alignment": None,
            "llm_business_alignment_rationale": "",
            "llm_authority_value": None,
            "llm_authority_value_rationale": "",
            "llm_tier_recommendation": "secondary",
            "llm_fit_adjustment": 0.0,
            "llm_final_cut_rationale": "Deterministic fallback (LLM unavailable)",
            "recommended_primary_keyword": fallback_primary_keyword,
            "recommended_primary_keyword_rationale": (
                "Top cluster-representative keyword from deterministic relevance scoring."
            ),
            "expected_role": self._infer_role(scored_topic),
            "recommended_url_type": self._infer_url_type(topic),
            "recommended_publish_order": None,
            "target_money_pages": money_pages[:2] if money_pages else [],
            "validation_notes": "Auto-assigned (LLM unavailable)",
        }

    def _llm_rerank_delta(
        self,
        *,
        deterministic_business_alignment: float,
        deterministic_authority: float,
        llm_business_alignment: float | None,
        llm_authority: float | None,
    ) -> float:
        """Convert blended qualitative changes into bounded score delta."""
        blended_business = self._blend_qualitative_scores(deterministic_business_alignment, llm_business_alignment)
        blended_authority = self._blend_qualitative_scores(deterministic_authority, llm_authority)
        deterministic_combo = (deterministic_business_alignment * 0.7) + (deterministic_authority * 0.3)
        blended_combo = (blended_business * 0.7) + (blended_authority * 0.3)
        raw_delta = (blended_combo - deterministic_combo) * 100 * 0.35
        return round(max(-LLM_RERANK_MAX_DELTA, min(LLM_RERANK_MAX_DELTA, raw_delta)), 2)

    def _fit_tier_sort_value(self, fit_tier: str | None) -> int:
        """Return sort ordinal for fit tiers."""
        normalized = str(fit_tier or "").strip().lower()
        if normalized == "primary":
            return 0
        if normalized == "secondary":
            return 1
        return 2

    def _generate_explanation(
        self,
        factors: dict[str, float],
        keywords: list[Keyword],
        fit_assessment: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate auditable explanation for priority + fit."""
        scoring_factors = [
            "brand_fit_score",
            "opportunity_score",
            "mean_intent_score",
            "difficulty_ease",
            "raw_volume_norm",
            "adjusted_volume_norm",
            "strategic_fit",
            "serp_opportunity_signal",
            "serp_signal",
        ]
        sorted_factors = sorted(
            [(k, v) for k, v in factors.items() if k in scoring_factors],
            key=lambda x: x[1],
            reverse=True,
        )

        top_factors: list[str] = []
        for name, value in sorted_factors[:2]:
            if name == "raw_volume_norm":
                total_vol = int(factors.get("raw_volume_sum", 0))
                top_factors.append(f"Raw demand: {total_vol:,}/mo")
            elif name == "adjusted_volume_norm":
                adjusted_total = int(factors.get("adjusted_volume_sum", 0))
                top_factors.append(f"Adjusted demand: {adjusted_total:,} cluster score volume")
            elif name == "difficulty_ease":
                avg_diff = sum(kw.difficulty or 50 for kw in keywords) / max(len(keywords), 1)
                top_factors.append(f"Difficulty ease: {value:.0%} (avg KD {avg_diff:.0f})")
            elif name == "mean_intent_score":
                top_factors.append(f"Intent strength: {value:.0%}")
            elif name == "brand_fit_score":
                top_factors.append(f"Brand fit: {value:.0%}")
            elif name == "opportunity_score":
                top_factors.append(f"Opportunity: {value:.0%}")
            elif name == "strategic_fit":
                top_factors.append(f"Strategic fit: {value:.0%}")
            elif name in {"serp_opportunity_signal", "serp_signal"}:
                top_factors.append(f"SERP signal: {value:.0%}")

        limiting_factors: list[str] = []
        for name, value in sorted_factors[-2:]:
            if value < 0.5:
                if name in {"raw_volume_norm", "adjusted_volume_norm"}:
                    limiting_factors.append("Low volume")
                elif name == "difficulty_ease":
                    limiting_factors.append("High difficulty")
                elif name == "strategic_fit":
                    limiting_factors.append("Weak business fit")
                elif name == "brand_fit_score":
                    limiting_factors.append("Low brand relevance")
                elif name == "opportunity_score":
                    limiting_factors.append("Weak opportunity")
                elif name in {"serp_opportunity_signal", "serp_signal"}:
                    limiting_factors.append("Weak SERP signal")
                elif name == "mean_intent_score":
                    limiting_factors.append("Weak intent clarity")

        missing_metrics = 0
        for kw in keywords:
            if kw.search_volume is None:
                missing_metrics += 1
            if kw.difficulty is None:
                missing_metrics += 1

        total_possible = len(keywords) * 2
        data_quality = 1.0 - (missing_metrics / max(total_possible, 1))

        return {
            "top_factors": top_factors,
            "limiting_factors": limiting_factors,
            "data_quality": round(data_quality, 2),
            "fit_reasons": fit_assessment["reasons"],
            "fit_score": fit_assessment["fit_score"],
            "brand_fit_score": factors.get("brand_fit_score"),
            "opportunity_score": factors.get("opportunity_score"),
            "deterministic_priority_score": factors.get("deterministic_priority_score"),
            "final_priority_score": factors.get("final_priority_score"),
            "llm_rerank_delta": factors.get("llm_rerank_delta"),
            "market_mode": factors.get("effective_market_mode"),
            "demand_fragmentation_index": factors.get("demand_fragmentation_index"),
        }

    def _infer_role(self, scored_topic: dict[str, Any]) -> str:
        """Infer expected role from deterministic metrics."""
        factors = scored_topic["scoring_factors"]
        if factors.get("achievability", 0) > 0.6 and factors.get("demand", 0) > 0.3:
            return "quick_win"
        if factors.get("business_alignment", 0) > 0.7:
            return "revenue_driver"
        return "authority_builder"

    def _infer_url_type(self, topic: Topic) -> str:
        """Infer URL type from topic characteristics."""
        page_type = topic.dominant_page_type or ""
        intent = topic.dominant_intent or ""

        if page_type in ("comparison", "alternatives"):
            return "comparison"
        if page_type in ("landing",):
            return "landing"
        if page_type in ("tool", "template"):
            return "resource"
        if intent in ("transactional",):
            return "landing"
        return "blog"

    async def _persist_results(self, result: PrioritizationOutput) -> None:
        """Save rankings and fit metadata to database."""
        for topic_data in result.topics:
            topic_result = await self.session.execute(
                select(Topic).where(Topic.id == topic_data["topic_id"])
            )
            topic = topic_result.scalar_one_or_none()
            if not topic:
                continue

            topic.priority_rank = topic_data["priority_rank"]
            topic.priority_score = topic_data["priority_score"]
            if (
                topic_data.get("fit_tier") in {"primary", "secondary"}
                and topic_data.get("primary_keyword_id")
            ):
                topic.primary_keyword_id = topic_data["primary_keyword_id"]
            topic.market_mode = topic_data.get("market_mode")
            topic.demand_fragmentation_index = topic_data.get("demand_fragmentation_index")
            topic.adjusted_volume_sum = topic_data.get("adjusted_volume_sum")
            topic.fit_tier = topic_data.get("fit_tier")
            topic.fit_score = topic_data.get("fit_score")
            topic.brand_fit_score = topic_data.get("brand_fit_score")
            topic.opportunity_score = topic_data.get("opportunity_score")
            topic.dynamic_fit_score = topic_data.get("dynamic_fit_score")
            topic.dynamic_opportunity_score = topic_data.get("dynamic_opportunity_score")
            topic.deterministic_priority_score = topic_data.get("deterministic_priority_score")
            topic.final_priority_score = topic_data.get("final_priority_score")
            topic.llm_rerank_delta = topic_data.get("llm_rerank_delta")
            topic.llm_fit_adjustment = topic_data.get("llm_fit_adjustment")
            topic.llm_tier_recommendation = topic_data.get("llm_tier_recommendation")
            topic.fit_threshold_primary = topic_data.get("fit_threshold_primary")
            topic.fit_threshold_secondary = topic_data.get("fit_threshold_secondary")
            topic.hard_exclusion_reason = topic_data.get("hard_exclusion_reason")
            topic.final_cut_reason_code = topic_data.get("final_cut_reason_code")
            topic.prioritization_diagnostics = topic_data.get("prioritization_diagnostics")
            topic.recommended_url_type = topic_data.get("recommended_url_type")
            topic.recommended_publish_order = topic_data.get("recommended_publish_order")
            topic.target_money_pages = topic_data.get("target_money_pages", [])
            topic.expected_role = topic_data.get("expected_role")

        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, self.step_number)

        summary: dict[str, Any] = {
            "topics_ranked": result.topics_ranked,
            "weights_used": result.weights_used,
            "primary_topics": sum(1 for t in result.topics if t.get("fit_tier") == "primary"),
            "secondary_topics": sum(1 for t in result.topics if t.get("fit_tier") == "secondary"),
            "excluded_topics": sum(1 for t in result.topics if t.get("fit_tier") == "excluded"),
            "needs_serp_validation": sum(1 for t in result.topics if t.get("needs_serp_validation")),
            "workflow_topics": sum(1 for t in result.topics if t.get("market_mode") == "fragmented_workflow"),
            "established_topics": sum(1 for t in result.topics if t.get("market_mode") == "established_category"),
            "eligible_topics_missing_primary_keyword": sum(
                1
                for t in result.topics
                if t.get("fit_tier") in {"primary", "secondary"} and not t.get("primary_keyword_id")
            ),
            "exact_pair_duplicates_removed": sum(
                1 for t in result.topics if t.get("final_cut_reason_code") == "exact_pair_duplicate"
            ),
            "near_duplicate_excluded": sum(
                1 for t in result.topics if t.get("final_cut_reason_code") == "near_duplicate_hard"
            ),
            "overlap_demoted": sum(
                1 for t in result.topics if t.get("final_cut_reason_code") == "overlap_demoted"
            ),
            "diversity_cap_demotions": sum(
                1 for t in result.topics if t.get("final_cut_reason_code") == "diversity_cap_demotion"
            ),
            "sibling_pairs_kept": sum(
                1
                for t in result.topics
                if (
                    isinstance(t.get("prioritization_diagnostics"), dict)
                    and isinstance(t["prioritization_diagnostics"].get("diversification"), dict)
                    and t["prioritization_diagnostics"]["diversification"].get("relationship_type") == "sibling_pair"
                    and t["prioritization_diagnostics"]["diversification"].get("action") == "keep"
                )
            ),
        }
        if result.quality_warnings:
            summary["quality_warnings"] = result.quality_warnings
        self.set_result_summary(summary)

        await self.session.commit()
