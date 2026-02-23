"""Step 7: Topic Prioritization.

Ranks topic backlog by business value and achievability, then applies
project-specific fit gating before briefing.
"""

import logging
import re
from dataclasses import dataclass, field
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


WORKFLOW_PRIORITY_WEIGHTS = {
    "mean_intent_score": 0.45,
    "difficulty_ease": 0.20,
    "adjusted_volume_norm": 0.15,
    "strategic_fit": 0.10,
    "serp_opportunity_signal": 0.10,
}
ESTABLISHED_PRIORITY_WEIGHTS = {
    "raw_volume_norm": 0.40,
    "difficulty_ease": 0.25,
    "mean_intent_score": 0.20,
    "serp_signal": 0.15,
}

LOW_COHERENCE_THRESHOLD = 0.7
LLM_BLEND_WEIGHT = 0.70
DFI_WORKFLOW_THRESHOLD = 0.10
EXTREME_DIFFICULTY_THRESHOLD = 90.0


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
            "fragmented_workflow": WORKFLOW_PRIORITY_WEIGHTS,
            "established_category": ESTABLISHED_PRIORITY_WEIGHTS,
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

            priority_score = self._calculate_mode_aware_priority_score(
                factors=factors,
                effective_market_mode=effective_market_mode,
            )
            explanation = self._generate_explanation(factors, keywords, fit_assessment)

            scored_topics.append({
                "topic_id": str(topic.id),
                "topic": topic,
                "keywords": keywords,
                "priority_score": priority_score,
                "priority_factors": factors,
                "fit_assessment": fit_assessment,
                "deterministic_business_alignment": factors["business_alignment"],
                "deterministic_authority": factors["authority"],
                "explanation": explanation,
                "effective_market_mode": effective_market_mode,
            })

        scored_topics.sort(key=lambda x: x["priority_score"], reverse=True)
        await self._update_progress(55, "Evaluating with LLM...")

        agent = PrioritizationAgent()
        prioritizations_by_topic_id: dict[str, dict[str, Any]] = {}
        strategy_notes = ""

        for batch_start in range(0, len(scored_topics), self.LLM_BATCH_SIZE):
            batch_end = min(batch_start + self.LLM_BATCH_SIZE, len(scored_topics))
            batch = scored_topics[batch_start:batch_end]

            progress = 55 + int((batch_start / len(scored_topics)) * 25)
            await self._update_progress(progress, f"Evaluating batch {batch_start // self.LLM_BATCH_SIZE + 1}...")

            try:
                agent_topics = []
                for st in batch:
                    topic = st["topic"]
                    keywords = st["keywords"]
                    primary_kw = next(
                        (kw for kw in keywords if kw.id == topic.primary_keyword_id),
                        keywords[0] if keywords else None,
                    )
                    agent_topics.append({
                        "topic_id": st["topic_id"],
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

                for j, prioritization in enumerate(output.prioritizations):
                    if j >= len(batch):
                        continue
                    topic_id = batch[j]["topic_id"]
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
                    }

                if batch_start == 0:
                    strategy_notes = output.overall_strategy_notes

            except Exception:
                logger.warning(
                    "Prioritization LLM batch failed, using fallback",
                    extra={"batch_start": batch_start},
                )
                for st in batch:
                    topic = st["topic"]
                    prioritizations_by_topic_id[st["topic_id"]] = {
                        "llm_business_alignment": None,
                        "llm_business_alignment_rationale": "",
                        "llm_authority_value": None,
                        "llm_authority_value_rationale": "",
                        "expected_role": self._infer_role(st),
                        "recommended_url_type": self._infer_url_type(topic),
                        "recommended_publish_order": None,
                        "target_money_pages": money_pages[:2] if money_pages else [],
                        "validation_notes": "Auto-assigned (LLM unavailable)",
                    }

        await self._update_progress(80, "Finalizing market-aware scores...")

        for st in scored_topics:
            topic_id = st["topic_id"]
            prioritization = prioritizations_by_topic_id.get(topic_id, {})
            factors = st["priority_factors"].copy()
            factors["llm_enhanced"] = {
                "business_alignment": prioritization.get("llm_business_alignment") is not None,
                "authority": prioritization.get("llm_authority_value") is not None,
            }
            if prioritization:
                factors["llm_rationales"] = {
                    "business_alignment": prioritization.get("llm_business_alignment_rationale", ""),
                    "authority": prioritization.get("llm_authority_value_rationale", ""),
                }
            st["priority_factors"] = factors

        scored_topics.sort(key=lambda x: x["priority_score"], reverse=True)
        self._apply_fit_gating(scored_topics, strategy)

        await self._update_progress(90, "Finalizing rankings...")

        output_topics: list[dict[str, Any]] = []
        next_rank = 1

        for st in scored_topics:
            topic = st["topic"]
            topic_id = st["topic_id"]
            prioritization = prioritizations_by_topic_id.get(topic_id, {})
            fit_assessment = st["fit_assessment"]

            is_eligible = fit_assessment["fit_tier"] == "primary"
            rank = next_rank if is_eligible else None
            if is_eligible:
                next_rank += 1

            needs_serp_validation = (topic.cluster_coherence or 1.0) < LOW_COHERENCE_THRESHOLD

            priority_factors = {
                **st["priority_factors"],
                "fit_score": fit_assessment["fit_score"],
                "fit_tier": fit_assessment["fit_tier"],
                "fit_reasons": fit_assessment["reasons"],
                "fit_threshold_used": fit_assessment["fit_threshold_used"],
            }

            output_topics.append({
                "topic_id": topic_id,
                "name": topic.name,
                "priority_rank": rank,
                "priority_score": st["priority_score"],
                "market_mode": st["effective_market_mode"],
                "demand_fragmentation_index": st["priority_factors"].get("demand_fragmentation_index"),
                "adjusted_volume_sum": st["priority_factors"].get("adjusted_volume_sum"),
                "priority_factors": priority_factors,
                "score_explanation": st["explanation"],
                "expected_role": prioritization.get("expected_role", "authority_builder") if is_eligible else "excluded",
                "recommended_url_type": prioritization.get("recommended_url_type", "blog") if is_eligible else None,
                "recommended_publish_order": (
                    prioritization.get("recommended_publish_order", rank) if is_eligible else None
                ),
                "target_money_pages": prioritization.get("target_money_pages", []) if is_eligible else [],
                "needs_serp_validation": needs_serp_validation,
                "validation_notes": prioritization.get("validation_notes", ""),
            })

        ranked_count = sum(1 for t in output_topics if t["priority_rank"] is not None)
        logger.info(
            "Prioritization complete",
            extra={
                "topics_ranked": ranked_count,
                "primary_topics": sum(1 for t in output_topics if t["priority_factors"].get("fit_tier") == "primary"),
                "secondary_topics": sum(1 for t in output_topics if t["priority_factors"].get("fit_tier") == "secondary"),
                "excluded_topics": sum(1 for t in output_topics if t["priority_factors"].get("fit_tier") == "excluded"),
            },
        )

        await self._update_progress(100, "Prioritization complete")

        return PrioritizationOutput(
            topics_ranked=ranked_count,
            weights_used=weights_used,
            topics=output_topics,
            strategy_notes=strategy_notes,
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
                    "No primary topics passed fit gating this iteration; discovery loop will continue "
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
            return {**ESTABLISHED_PRIORITY_WEIGHTS, **custom}
        return ESTABLISHED_PRIORITY_WEIGHTS.copy()

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
    ) -> float:
        """Calculate cluster score using mode-aware weighting."""
        if effective_market_mode == "fragmented_workflow":
            score = (
                (factors.get("mean_intent_score", 0.0) * 0.45)
                + (factors.get("difficulty_ease", 0.0) * 0.20)
                + (factors.get("adjusted_volume_norm", 0.0) * 0.15)
                + (factors.get("strategic_fit", 0.0) * 0.10)
                + (factors.get("serp_opportunity_signal", 0.0) * 0.10)
            )
        else:
            score = (
                (factors.get("raw_volume_norm", 0.0) * 0.40)
                + (factors.get("difficulty_ease", 0.0) * 0.25)
                + (factors.get("mean_intent_score", 0.0) * 0.20)
                + (factors.get("serp_signal", 0.0) * 0.15)
            )
        return round(score * 100, 2)

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
        """Calculate market-agnostic fit score and reasons for gating."""
        keyword_text = " ".join(kw.keyword for kw in keywords[:20])
        topic_text = f"{topic.name} {keyword_text}".lower()
        topic_tokens = self._tokenize_text(topic_text)

        offer_terms = self._collect_offer_terms(brand, strategy)
        icp_terms = self._collect_icp_terms(strategy)

        if offer_terms:
            overlap_relevance = self._overlap_score(topic_tokens, offer_terms)
            solution_relevance = (overlap_relevance * 0.7) + (business_alignment_score * 0.3)
        else:
            solution_relevance = business_alignment_score
        icp_relevance = self._overlap_score(topic_tokens, icp_terms) if icp_terms else 0.50
        conversion_path = self._conversion_path_score(topic, strategy)
        serp_suitability = self._serp_intent_suitability(topic, keywords)

        fit_score = (
            (solution_relevance * 0.40)
            + (icp_relevance * 0.30)
            + (conversion_path * 0.20)
            + (serp_suitability * 0.10)
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
            hard_exclusion_reason = (
                f"extreme_difficulty:{avg_difficulty:.2f}"
                f">{EXTREME_DIFFICULTY_THRESHOLD:.2f}"
            )

        if strategy.scope_mode == "strict" and solution_relevance < 0.35:
            fit_score *= 0.7
        elif strategy.scope_mode == "broad_education" and topic.dominant_intent == "informational":
            fit_score = min(1.0, fit_score + 0.05)

        if hard_exclusion_reason:
            fit_score = min(fit_score, 0.10)

        reasons = [
            f"Solution relevance: {solution_relevance:.0%}",
            f"ICP relevance: {icp_relevance:.0%}",
            f"Conversion path: {conversion_path:.0%}",
            f"SERP/intent suitability: {serp_suitability:.0%}",
            f"Avg difficulty: {avg_difficulty:.1f}",
        ]
        if hard_exclusion_reason:
            reasons.append(f"Hard exclusion: {hard_exclusion_reason}")

        return {
            "fit_score": round(max(0.0, min(1.0, fit_score)), 4),
            "components": {
                "solution_relevance": round(solution_relevance, 4),
                "icp_relevance": round(icp_relevance, 4),
                "conversion_path_plausibility": round(conversion_path, 4),
                "serp_intent_suitability": round(serp_suitability, 4),
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
                return f"matches_excluded_topic:{excluded_topic}"

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

    def _apply_fit_gating(self, scored_topics: list[dict[str, Any]], strategy: RunStrategy) -> None:
        """Apply hard eligibility gate using primary-fit threshold only."""
        base_threshold = strategy.base_threshold()
        for st in scored_topics:
            fit = st["fit_assessment"]
            if fit["hard_exclusion_reason"]:
                fit["fit_tier"] = "excluded"
                fit["fit_threshold_used"] = base_threshold
                continue

            if fit["fit_score"] >= base_threshold:
                fit["fit_tier"] = "primary"
                fit["fit_threshold_used"] = base_threshold
            else:
                fit["fit_tier"] = "excluded"
                fit["fit_threshold_used"] = base_threshold

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
        """Calculate overlap score for two term sets."""
        if not topic_terms or not reference_terms:
            return 0.0
        overlap = len(topic_terms & reference_terms)
        return min(1.0, overlap / max(len(reference_terms), 1))

    def _calculate_priority_score(self, factors: dict[str, float], weights: dict[str, float]) -> float:
        """Calculate weighted priority score (0-100)."""
        total = 0.0
        for factor_name, weight in weights.items():
            total += factors.get(factor_name, 0.0) * weight
        return round(total * 100, 2)

    def _blend_qualitative_scores(self, deterministic_score: float, llm_score: float | None) -> float:
        """Blend deterministic and LLM qualitative scores."""
        if llm_score is None:
            return deterministic_score
        return (llm_score * LLM_BLEND_WEIGHT) + (deterministic_score * (1 - LLM_BLEND_WEIGHT))

    def _generate_explanation(
        self,
        factors: dict[str, float],
        keywords: list[Keyword],
        fit_assessment: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate auditable explanation for priority + fit."""
        scoring_factors = [
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
            "market_mode": factors.get("effective_market_mode"),
            "demand_fragmentation_index": factors.get("demand_fragmentation_index"),
        }

    def _infer_role(self, scored_topic: dict[str, Any]) -> str:
        """Infer expected role from deterministic metrics."""
        factors = scored_topic["priority_factors"]
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
            topic.market_mode = topic_data.get("market_mode")
            topic.demand_fragmentation_index = topic_data.get("demand_fragmentation_index")
            topic.adjusted_volume_sum = topic_data.get("adjusted_volume_sum")
            topic.priority_factors = {
                **topic_data["priority_factors"],
                "explanation": topic_data["score_explanation"],
            }
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
            "primary_topics": sum(
                1 for t in result.topics if t.get("priority_factors", {}).get("fit_tier") == "primary"
            ),
            "secondary_topics": sum(
                1 for t in result.topics if t.get("priority_factors", {}).get("fit_tier") == "secondary"
            ),
            "excluded_topics": sum(
                1 for t in result.topics if t.get("priority_factors", {}).get("fit_tier") == "excluded"
            ),
            "needs_serp_validation": sum(1 for t in result.topics if t.get("needs_serp_validation")),
            "workflow_topics": sum(1 for t in result.topics if t.get("market_mode") == "fragmented_workflow"),
            "established_topics": sum(1 for t in result.topics if t.get("market_mode") == "established_category"),
        }
        if result.quality_warnings:
            summary["quality_warnings"] = result.quality_warnings
        self.set_result_summary(summary)

        await self.session.commit()
