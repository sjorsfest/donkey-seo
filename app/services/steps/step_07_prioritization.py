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
    weights_used: dict[str, float]
    topics: list[dict[str, Any]] = field(default_factory=list)
    strategy_notes: str = ""
    quality_warnings: list[str] = field(default_factory=list)


DEFAULT_PRIORITY_WEIGHTS = {
    "demand": 0.30,
    "achievability": 0.25,
    "business_alignment": 0.30,
    "authority": 0.15,
}

LOW_COHERENCE_THRESHOLD = 0.7
LLM_BLEND_WEIGHT = 0.70


class Step07PrioritizationService(BaseStepService[PrioritizationInput, PrioritizationOutput]):
    """Step 7: Topic Prioritization with fit gating."""

    step_number = 7
    step_name = "prioritization"
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

        if project.current_step < 6:
            raise ValueError("Step 6 (Clustering) must be completed first")

        topics_result = await self.session.execute(
            select(Topic).where(Topic.project_id == input_data.project_id).limit(1)
        )
        if not topics_result.scalars().first():
            raise ValueError("No topics found. Run Step 6 first.")

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

        weights = self._get_priority_weights(project)
        logger.info(
            "Prioritization starting",
            extra={
                "project_id": input_data.project_id,
                "topic_count": len(all_topics),
                "weights": weights,
                "fit_profile": strategy.fit_threshold_profile,
            },
        )

        if not all_topics:
            return PrioritizationOutput(
                topics_ranked=0,
                weights_used=DEFAULT_PRIORITY_WEIGHTS,
            )

        await self._update_progress(10, f"Scoring {len(all_topics)} topics...")

        brand_context = self._build_brand_context(brand, strategy)
        money_pages = self._extract_money_pages(brand)
        primary_goal = project.primary_goal or ""
        volume_reference = self._calculate_volume_reference(all_topics)

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
                volume_reference=volume_reference,
            )
            fit_assessment = self._calculate_fit_assessment(
                topic=topic,
                keywords=keywords,
                brand=brand,
                strategy=strategy,
                business_alignment_score=factors["business_alignment"],
            )
            factors["fit_score"] = fit_assessment["fit_score"]

            priority_score = self._calculate_priority_score(factors, weights)
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

        await self._update_progress(80, "Blending LLM qualitative scores...")

        for st in scored_topics:
            topic_id = st["topic_id"]
            prioritization = prioritizations_by_topic_id.get(topic_id, {})
            llm_business = prioritization.get("llm_business_alignment")
            llm_authority = prioritization.get("llm_authority_value")

            hybrid_business_alignment = self._blend_qualitative_scores(
                st["deterministic_business_alignment"],
                llm_business,
            )
            hybrid_authority = self._blend_qualitative_scores(
                st["deterministic_authority"],
                llm_authority,
            )

            hybrid_factors = st["priority_factors"].copy()
            hybrid_factors["business_alignment"] = hybrid_business_alignment
            hybrid_factors["authority"] = hybrid_authority
            hybrid_factors["llm_enhanced"] = {
                "business_alignment": llm_business is not None,
                "authority": llm_authority is not None,
            }
            if prioritization:
                hybrid_factors["llm_rationales"] = {
                    "business_alignment": prioritization.get("llm_business_alignment_rationale", ""),
                    "authority": prioritization.get("llm_authority_value_rationale", ""),
                }

            st["priority_score"] = self._calculate_priority_score(hybrid_factors, weights)
            st["priority_factors"] = hybrid_factors

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

            is_eligible = fit_assessment["fit_tier"] in {"primary", "secondary"}
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
            weights_used=weights,
            topics=output_topics,
            strategy_notes=strategy_notes,
        )

    async def _validate_output(self, result: PrioritizationOutput, input_data: PrioritizationInput) -> None:
        """Ensure output can be consumed by Step 12."""
        if result.topics_ranked <= 0:
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
        """Get priority weights from project settings or use defaults."""
        if project.api_budget_caps and "priority_weights" in project.api_budget_caps:
            custom = project.api_budget_caps["priority_weights"]
            return {**DEFAULT_PRIORITY_WEIGHTS, **custom}
        return DEFAULT_PRIORITY_WEIGHTS.copy()

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

    def _calculate_factors(
        self,
        topic: Topic,
        keywords: list[Keyword],
        brand: BrandProfile | None,
        money_pages: list[str],
        volume_reference: float,
    ) -> dict[str, float]:
        """Calculate weighted-priority factors (0-1 normalized)."""
        total_volume = topic.total_volume or sum(kw.search_volume or 0 for kw in keywords)
        if volume_reference == 0:
            demand_score = 0.5
        else:
            demand_score = min(total_volume / volume_reference, 1.0)

        avg_difficulty = topic.avg_difficulty
        if avg_difficulty is None:
            difficulties = [kw.difficulty for kw in keywords if kw.difficulty is not None]
            avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 50.0
        achievability_score = 1.0 - (avg_difficulty / self.DIFFICULTY_MAX)
        achievability_score = max(0.0, min(1.0, achievability_score))

        business_alignment = self._calculate_business_alignment(topic, keywords, brand, money_pages)
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
        """Apply hard eligibility gate with one-pass threshold relaxation."""
        base_threshold = strategy.base_threshold()
        relaxed_threshold = strategy.relaxed_threshold()
        eligible_target = strategy.eligible_target()

        primary_count = 0
        for st in scored_topics:
            fit = st["fit_assessment"]
            if fit["hard_exclusion_reason"]:
                fit["fit_tier"] = "excluded"
                fit["fit_threshold_used"] = base_threshold
                continue

            if fit["fit_score"] >= base_threshold:
                fit["fit_tier"] = "primary"
                fit["fit_threshold_used"] = base_threshold
                primary_count += 1
            else:
                fit["fit_tier"] = "excluded"
                fit["fit_threshold_used"] = base_threshold

        if primary_count >= eligible_target:
            return

        for st in scored_topics:
            fit = st["fit_assessment"]
            if fit["fit_tier"] == "primary" or fit["hard_exclusion_reason"]:
                continue
            if fit["fit_score"] >= relaxed_threshold:
                fit["fit_tier"] = "secondary"
                fit["fit_threshold_used"] = relaxed_threshold
            else:
                fit["fit_tier"] = "excluded"
                fit["fit_threshold_used"] = relaxed_threshold

        eligible_count = sum(
            1 for st in scored_topics if st["fit_assessment"]["fit_tier"] in {"primary", "secondary"}
        )
        if eligible_count >= eligible_target:
            return

        # Adaptive fallback: if strict+relaxed still yields too little coverage,
        # promote best remaining non-hard-excluded topics as secondary.
        # Guard: never promote topics with near-zero ICP relevance — these are
        # completely disconnected from the target audience.
        icp_floor = 0.05
        fallback_floor = max(0.30, relaxed_threshold - 0.15)
        excluded_candidates = [
            st for st in scored_topics
            if st["fit_assessment"]["fit_tier"] == "excluded"
            and not st["fit_assessment"]["hard_exclusion_reason"]
            and st["fit_assessment"]["fit_score"] >= fallback_floor
            and st["fit_assessment"]["components"].get("icp_relevance", 0) >= icp_floor
        ]
        if len(excluded_candidates) < (eligible_target - eligible_count):
            # Final safety net: allow top non-hard-excluded topics even below floor,
            # but still enforce ICP relevance minimum.
            excluded_candidates = [
                st for st in scored_topics
                if st["fit_assessment"]["fit_tier"] == "excluded"
                and not st["fit_assessment"]["hard_exclusion_reason"]
                and st["fit_assessment"]["components"].get("icp_relevance", 0) >= icp_floor
            ]
        excluded_candidates.sort(
            key=lambda st: st["fit_assessment"]["fit_score"],
            reverse=True,
        )

        slots_needed = max(0, eligible_target - eligible_count)
        for st in excluded_candidates[:slots_needed]:
            fit = st["fit_assessment"]
            fit["fit_tier"] = "secondary"
            fit["fit_threshold_used"] = fallback_floor
            fit.setdefault("reasons", []).append(
                "Adaptive fallback promotion: included as secondary to avoid empty/too-small backlog."
            )

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
        sorted_factors = sorted(
            [(k, v) for k, v in factors.items() if k in DEFAULT_PRIORITY_WEIGHTS],
            key=lambda x: x[1],
            reverse=True,
        )

        top_factors: list[str] = []
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

        limiting_factors: list[str] = []
        for name, value in sorted_factors[-2:]:
            if value < 0.5:
                if name == "demand":
                    limiting_factors.append("Low volume")
                elif name == "achievability":
                    limiting_factors.append("High difficulty")
                elif name == "business_alignment":
                    limiting_factors.append("Weak business fit")
                elif name == "authority":
                    limiting_factors.append("Low authority value")

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
        project.current_step = 7

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
        }
        if result.quality_warnings:
            summary["quality_warnings"] = result.quality_warnings
        self.set_result_summary(summary)

        await self.session.commit()
