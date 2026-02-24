"""Synthesize and persist discovery iteration learnings."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.discovery_learning_summarizer import (
    DiscoveryLearningSummarizerAgent,
    DiscoveryLearningSummarizerInput,
    LearningDraft,
)
from app.core.database import rollback_read_only_transaction
from app.core.db_kernel import db_write
from app.models.discovery_learning import DiscoveryIterationLearning
from app.models.generated_dtos import DiscoveryIterationLearningCreateDTO
from app.models.keyword import Keyword, SeedTopic
from app.models.topic import Topic
from app.services.discovery_capabilities import (
    CAPABILITY_CLUSTERING,
    CAPABILITY_INTENT_CLASSIFICATION,
    CAPABILITY_KEYWORD_EXPANSION,
    CAPABILITY_PRIORITIZATION,
    CAPABILITY_SEED_GENERATION,
    CAPABILITY_SERP_VALIDATION,
    DEFAULT_AGENT_BY_CAPABILITY,
)

logger = logging.getLogger(__name__)

STATUS_NEW = "new"
STATUS_CONFIRMED = "confirmed"
STATUS_REGRESSED = "regressed"


@dataclass(slots=True)
class LearningCandidate:
    """Intermediate learning candidate before persistence."""

    learning_key: str
    source_capability: str
    source_agent: str | None
    learning_type: str
    polarity: str
    title: str
    detail: str
    recommendation: str | None
    confidence: float | None
    current_metric: float | None
    applies_to_capabilities: list[str]
    applies_to_agents: list[str] | None
    evidence: dict[str, Any]
    status: str = STATUS_NEW
    baseline_metric: float | None = None
    delta_metric: float | None = None
    novelty_score: float | None = None


class DiscoveryLearningService:
    """Build brand-specific iteration learnings and persist them."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def synthesize_and_persist(
        self,
        *,
        project_id: str,
        pipeline_run_id: str,
        iteration_index: int,
        decisions: list[Any],
        step_summaries: dict[int, dict[str, Any]],
    ) -> list[DiscoveryIterationLearning]:
        """Generate deterministic learnings and persist compact rewritten versions."""
        seed_topics = await self._load_seed_topics(project_id)
        topics = await self._load_topics(project_id)
        keywords = await self._load_keywords(project_id)

        accepted_topic_ids = {
            str(getattr(item, "source_topic_id"))
            for item in decisions
            if getattr(item, "decision", "") == "accepted"
            and getattr(item, "source_topic_id", None)
        }

        usable_by_keyword = self._compute_usable_keywords(
            keywords=keywords,
            topics_by_id={str(topic.id): topic for topic in topics},
            accepted_topic_ids=accepted_topic_ids,
        )
        candidates = self._build_deterministic_candidates(
            decisions=decisions,
            step_summaries=step_summaries,
            keywords=keywords,
            seed_topics=seed_topics,
            topics_by_id={str(topic.id): topic for topic in topics},
            usable_by_keyword=usable_by_keyword,
        )

        if not candidates:
            candidates = [self._build_fallback_candidate(usable_by_keyword, len(keywords))]

        await self._apply_history(
            project_id=project_id,
            pipeline_run_id=pipeline_run_id,
            iteration_index=iteration_index,
            candidates=candidates,
        )
        await rollback_read_only_transaction(
            self.session,
            context="discovery_learning_rewrite",
        )
        await self._rewrite_candidates(project_id=project_id, candidates=candidates)

        payloads: list[DiscoveryIterationLearningCreateDTO] = []
        for candidate in candidates:
            applies_to_capabilities = [
                str(item).strip()
                for item in (candidate.applies_to_capabilities or [])
                if str(item).strip()
            ]
            applies_to_agents = [
                str(item).strip()
                for item in (candidate.applies_to_agents or [])
                if str(item).strip()
            ]
            payloads.append(
                DiscoveryIterationLearningCreateDTO(
                    project_id=project_id,
                    pipeline_run_id=pipeline_run_id,
                    iteration_index=iteration_index,
                    source_capability=candidate.source_capability,
                    source_agent=candidate.source_agent,
                    learning_key=candidate.learning_key,
                    learning_type=candidate.learning_type,
                    polarity=candidate.polarity,
                    status=candidate.status,
                    title=candidate.title,
                    detail=candidate.detail,
                    recommendation=candidate.recommendation,
                    confidence=candidate.confidence,
                    novelty_score=candidate.novelty_score,
                    baseline_metric=candidate.baseline_metric,
                    current_metric=candidate.current_metric,
                    delta_metric=candidate.delta_metric,
                    applies_to_capabilities=applies_to_capabilities or None,
                    applies_to_agents=applies_to_agents or None,
                    evidence=candidate.evidence,
                )
            )

        async def _persist_once(session: AsyncSession) -> list[DiscoveryIterationLearning]:
            persisted_rows: list[DiscoveryIterationLearning] = []
            for payload in payloads:
                persisted_rows.append(
                    DiscoveryIterationLearning.create(
                        session,
                        payload,
                    )
                )
            await session.flush()
            return persisted_rows

        persisted = await db_write(
            _persist_once,
            operation_name="discovery_iteration_learning_persist",
            attempts=3,
        )
        logger.info(
            "Discovery iteration learnings persisted",
            extra={
                "project_id": project_id,
                "run_id": pipeline_run_id,
                "iteration_index": iteration_index,
                "learning_count": len(persisted),
            },
        )
        return persisted

    async def _load_seed_topics(self, project_id: str) -> dict[str, SeedTopic]:
        result = await self.session.execute(
            select(SeedTopic).where(SeedTopic.project_id == project_id)
        )
        return {str(item.id): item for item in result.scalars()}

    async def _load_topics(self, project_id: str) -> list[Topic]:
        result = await self.session.execute(select(Topic).where(Topic.project_id == project_id))
        return list(result.scalars())

    async def _load_keywords(self, project_id: str) -> list[Keyword]:
        result = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == project_id,
                Keyword.status == "active",
            )
        )
        return list(result.scalars())

    def _compute_usable_keywords(
        self,
        *,
        keywords: list[Keyword],
        topics_by_id: dict[str, Topic],
        accepted_topic_ids: set[str],
    ) -> dict[str, bool]:
        usable: dict[str, bool] = {}
        for keyword in keywords:
            topic_id = str(keyword.topic_id) if keyword.topic_id else None
            topic = topics_by_id.get(topic_id or "")
            fit_tier = self._topic_fit_tier(topic)
            has_eligible_rank = topic is not None and topic.priority_rank is not None

            value = False
            if topic_id and topic_id in accepted_topic_ids:
                value = True
            elif accepted_topic_ids:
                value = False
            else:
                value = fit_tier in {"primary", "secondary"} and has_eligible_rank
            usable[str(keyword.id)] = value
        return usable

    def _build_deterministic_candidates(
        self,
        *,
        decisions: list[Any],
        step_summaries: dict[int, dict[str, Any]],
        keywords: list[Keyword],
        seed_topics: dict[str, SeedTopic],
        topics_by_id: dict[str, Topic],
        usable_by_keyword: dict[str, bool],
    ) -> list[LearningCandidate]:
        total_keywords = len(keywords)
        usable_keywords = sum(1 for key in usable_by_keyword.values() if key)
        overall_rate = (usable_keywords / total_keywords) if total_keywords else 0.0
        topics_with_keywords = sum(
            1 for topic in topics_by_id.values() if (getattr(topic, "keyword_count", 0) or 0) > 0
        )
        rank_eligible_topics = sum(
            1
            for topic in topics_by_id.values()
            if self._topic_fit_tier(topic) in {"primary", "secondary"} and topic.priority_rank is not None
        )
        final_cut_candidates = sum(
            1
            for topic in topics_by_id.values()
            if topic.llm_tier_recommendation is not None
            or (topic.final_cut_reason_code is not None and topic.final_cut_reason_code != "outside_final_cut_pool")
        )
        final_cut_selected = sum(
            1
            for topic in topics_by_id.values()
            if self._topic_fit_tier(topic) in {"primary", "secondary"}
            and topic.priority_rank is not None
            and (
                topic.llm_tier_recommendation is not None
                or topic.final_cut_reason_code in {"llm_zero_result_fallback", "deterministic_fallback"}
            )
        )
        diversification_rows: list[dict[str, Any]] = []
        for topic in topics_by_id.values():
            diagnostics = getattr(topic, "prioritization_diagnostics", None)
            if not isinstance(diagnostics, dict):
                continue
            diversification = diagnostics.get("diversification")
            if isinstance(diversification, dict):
                diversification_rows.append(diversification)
        diversification_topic_count = len(diversification_rows)
        suppressed_reason_codes = {
            "exact_pair_duplicate",
            "near_duplicate_hard",
            "overlap_demoted",
            "diversity_cap_demotion",
        }
        diversification_suppressed = sum(
            1
            for topic in topics_by_id.values()
            if topic.final_cut_reason_code in suppressed_reason_codes
        )
        exact_pair_duplicates = sum(
            1
            for topic in topics_by_id.values()
            if topic.final_cut_reason_code == "exact_pair_duplicate"
        )
        sibling_pairs_kept = sum(
            1
            for row in diversification_rows
            if row.get("relationship_type") == "sibling_pair"
            and row.get("action") == "keep"
        )
        sibling_pairs_seen = sum(
            1 for row in diversification_rows if row.get("relationship_type") == "sibling_pair"
        )
        diversification_suppression_rate = (
            diversification_suppressed / diversification_topic_count
            if diversification_topic_count
            else None
        )
        sibling_pair_keep_rate = (
            sibling_pairs_kept / sibling_pairs_seen
            if sibling_pairs_seen
            else None
        )
        exact_pair_duplicate_rate = (
            exact_pair_duplicates / diversification_topic_count
            if diversification_topic_count
            else None
        )
        zero_usable_context = ""
        if overall_rate == 0.0:
            if topics_with_keywords > 0 and rank_eligible_topics == 0:
                if final_cut_candidates > 0:
                    zero_usable_context = (
                        " Keywords were clustered, but all candidates were filtered during the LLM final cut."
                    )
                else:
                    zero_usable_context = (
                        " Keywords were clustered, but no topics were rank-eligible after deterministic gating."
                    )
            elif topics_with_keywords == 0:
                zero_usable_context = " No topics with attached keywords were available."

        candidates: list[LearningCandidate] = [
            LearningCandidate(
                learning_key="overall:usable_keyword_rate",
                source_capability=CAPABILITY_PRIORITIZATION,
                source_agent=DEFAULT_AGENT_BY_CAPABILITY.get(CAPABILITY_PRIORITIZATION),
                learning_type="strategy_effect",
                polarity=(
                    "positive"
                    if overall_rate >= 0.55
                    else ("negative" if overall_rate < 0.35 else "neutral")
                ),
                title="Usable keyword yield this iteration",
                detail=(
                    f"Usable keyword rate is {overall_rate:.1%} "
                    f"({usable_keywords}/{total_keywords})."
                    f"{zero_usable_context}"
                ),
                recommendation=(
                    "Double down on high-yield archetypes from this run."
                    if overall_rate >= 0.55
                    else (
                        "Increase fit-eligible topic yield before expanding volume."
                        if overall_rate == 0.0 and topics_with_keywords > 0 and rank_eligible_topics == 0
                        else "Tighten seed/archetype targeting before expanding volume."
                    )
                ),
                confidence=self._confidence_from_count(total_keywords),
                current_metric=overall_rate,
                applies_to_capabilities=[
                    CAPABILITY_SEED_GENERATION,
                    CAPABILITY_KEYWORD_EXPANSION,
                    CAPABILITY_CLUSTERING,
                    CAPABILITY_PRIORITIZATION,
                ],
                applies_to_agents=[
                    DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_SEED_GENERATION],
                    DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_CLUSTERING],
                    DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_PRIORITIZATION],
                ],
                evidence={
                    "usable_keywords": usable_keywords,
                    "total_keywords": total_keywords,
                    "usable_keyword_rate": round(overall_rate, 4),
                    "rank_eligible_topics": rank_eligible_topics,
                    "llm_final_cut_candidates": final_cut_candidates,
                    "llm_final_cut_selected": final_cut_selected,
                    **(
                        {
                            "diversification_suppression_rate": round(diversification_suppression_rate, 4),
                            "sibling_pair_keep_rate": round(sibling_pair_keep_rate, 4)
                            if sibling_pair_keep_rate is not None
                            else None,
                            "exact_pair_duplicate_rate": (
                                round(exact_pair_duplicate_rate, 4)
                                if exact_pair_duplicate_rate is not None
                                else None
                            ),
                        }
                        if diversification_suppression_rate is not None
                        else {}
                    ),
                },
            )
        ]

        archetype_stats: dict[str, dict[str, int]] = {}
        content_pattern_stats: dict[str, dict[str, int]] = {}
        for keyword in keywords:
            key = str(keyword.id)
            usable = usable_by_keyword.get(key, False)
            archetype = self._keyword_archetype(keyword, seed_topics)
            stats = archetype_stats.setdefault(archetype, {"usable": 0, "total": 0})
            stats["total"] += 1
            if usable:
                stats["usable"] += 1

            topic = topics_by_id.get(str(keyword.topic_id) if keyword.topic_id else "")
            if topic:
                pattern = f"{topic.recommended_url_type or 'unknown'}|{topic.dominant_intent or 'unknown'}"
                cstats = content_pattern_stats.setdefault(pattern, {"usable": 0, "total": 0})
                cstats["total"] += 1
                if usable:
                    cstats["usable"] += 1

        for archetype, stats in archetype_stats.items():
            total = stats["total"]
            if total < 5:
                continue
            rate = stats["usable"] / total
            delta = rate - overall_rate
            if abs(delta) < 0.08:
                continue
            polarity = "positive" if delta > 0 else "negative"
            archetype_label = self._humanize_token(archetype)
            candidates.append(
                LearningCandidate(
                    learning_key=f"archetype:{archetype}:usable_rate",
                    source_capability=CAPABILITY_SEED_GENERATION,
                    source_agent=DEFAULT_AGENT_BY_CAPABILITY.get(CAPABILITY_SEED_GENERATION),
                    learning_type="archetype_performance",
                    polarity=polarity,
                    title=f"{archetype_label} keyword archetype performance",
                    detail=(
                        f"The {archetype_label.lower()} archetype produced {rate:.1%} usable keywords "
                        f"({stats['usable']}/{total}), vs {overall_rate:.1%} overall."
                    ),
                    recommendation=(
                        f"Increase seed coverage for {archetype_label.lower()} terms."
                        if delta > 0
                        else f"Reduce low-signal {archetype_label.lower()} seeds or tighten scope."
                    ),
                    confidence=self._confidence_from_count(total),
                    current_metric=rate,
                    applies_to_capabilities=[
                        CAPABILITY_SEED_GENERATION,
                        CAPABILITY_KEYWORD_EXPANSION,
                    ],
                    applies_to_agents=[
                        DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_SEED_GENERATION],
                    ],
                    evidence={
                        "archetype": archetype,
                        "usable": stats["usable"],
                        "total": total,
                        "usable_rate": round(rate, 4),
                        "overall_rate": round(overall_rate, 4),
                    },
                )
            )

        for pattern, stats in content_pattern_stats.items():
            total = stats["total"]
            if total < 4:
                continue
            rate = stats["usable"] / total
            delta = rate - overall_rate
            if abs(delta) < 0.10:
                continue
            polarity = "positive" if delta > 0 else "negative"
            pattern_label = self._humanize_pattern(pattern)
            candidates.append(
                LearningCandidate(
                    learning_key=f"pattern:{pattern}:usable_rate",
                    source_capability=CAPABILITY_PRIORITIZATION,
                    source_agent=DEFAULT_AGENT_BY_CAPABILITY.get(CAPABILITY_PRIORITIZATION),
                    learning_type="content_pattern_performance",
                    polarity=polarity,
                    title="Content pattern yield signal",
                    detail=(
                        f"{pattern_label} yielded {rate:.1%} usable keywords "
                        f"({stats['usable']}/{total}), vs {overall_rate:.1%} overall."
                    ),
                    recommendation=(
                        f"Prioritize {pattern_label.lower()} in cluster naming and topic ranking."
                        if delta > 0
                        else
                        f"De-emphasize {pattern_label.lower()} unless strategic fit is strong."
                    ),
                    confidence=self._confidence_from_count(total),
                    current_metric=rate,
                    applies_to_capabilities=[CAPABILITY_CLUSTERING, CAPABILITY_PRIORITIZATION],
                    applies_to_agents=[
                        DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_CLUSTERING],
                        DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_PRIORITIZATION],
                    ],
                    evidence={
                        "pattern": pattern,
                        "usable": stats["usable"],
                        "total": total,
                        "usable_rate": round(rate, 4),
                        "overall_rate": round(overall_rate, 4),
                    },
                )
            )

        rejected = [item for item in decisions if getattr(item, "decision", "") == "rejected"]
        reason_counts: dict[str, int] = {}
        for item in rejected:
            for reason in getattr(item, "rejection_reasons", []) or []:
                norm = self._normalize_reason(reason)
                reason_counts[norm] = reason_counts.get(norm, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda pair: pair[1], reverse=True)[:3]:
            rejected_total = max(1, len(rejected))
            rate = count / rejected_total
            capability = self._capability_for_rejection_reason(reason)
            reason_label = self._humanize_rejection_reason(reason)
            applies_to_agents = [
                agent
                for agent in [
                    DEFAULT_AGENT_BY_CAPABILITY.get(capability),
                    DEFAULT_AGENT_BY_CAPABILITY.get(CAPABILITY_PRIORITIZATION),
                ]
                if agent is not None
            ]
            candidates.append(
                LearningCandidate(
                    learning_key=f"bottleneck:{reason}",
                    source_capability=capability,
                    source_agent=DEFAULT_AGENT_BY_CAPABILITY.get(capability),
                    learning_type="quality_bottleneck",
                    polarity="negative",
                    title="Recurring rejection bottleneck",
                    detail=(
                        f"The rejection reason '{reason_label}' appears in {rate:.1%} "
                        f"of rejected topics ({count}/{rejected_total})."
                    ),
                    recommendation=self._recommendation_for_rejection_reason(reason),
                    confidence=self._confidence_from_count(rejected_total),
                    current_metric=rate,
                    applies_to_capabilities=[capability, CAPABILITY_PRIORITIZATION],
                    applies_to_agents=applies_to_agents,
                    evidence={
                        "reason": reason,
                        "count": count,
                        "rejected_total": rejected_total,
                        "rate": round(rate, 4),
                    },
                )
            )

        candidates.extend(self._step_summary_candidates(step_summaries))

        by_key: dict[str, LearningCandidate] = {}
        for candidate in candidates:
            if candidate.learning_key in by_key:
                continue
            by_key[candidate.learning_key] = candidate
        return list(by_key.values())[:20]

    def _step_summary_candidates(
        self,
        step_summaries: dict[int, dict[str, Any]],
    ) -> list[LearningCandidate]:
        candidates: list[LearningCandidate] = []
        summary2 = step_summaries.get(2) or {}
        gaps_count = self._safe_int(summary2.get("known_gaps_count"))
        seeds_created = self._safe_int(summary2.get("seeds_created"))
        if seeds_created > 0 and gaps_count > 0:
            gap_rate = gaps_count / max(1, seeds_created)
            candidates.append(
                LearningCandidate(
                    learning_key="step2:known_gaps_rate",
                    source_capability=CAPABILITY_SEED_GENERATION,
                    source_agent=DEFAULT_AGENT_BY_CAPABILITY.get(CAPABILITY_SEED_GENERATION),
                    learning_type="strategy_effect",
                    polarity="negative" if gap_rate > 0.12 else "neutral",
                    title="Seed generation knowledge gaps",
                    detail=(
                        f"Known gaps ratio is {gap_rate:.1%} "
                        f"({gaps_count} gaps over {seeds_created} seeds)."
                    ),
                    recommendation=(
                        "Feed missing product/audience context before next iteration."
                        if gap_rate > 0.12
                        else "Monitor but keep current seed generation strategy."
                    ),
                    confidence=self._confidence_from_count(seeds_created),
                    current_metric=gap_rate,
                    applies_to_capabilities=[CAPABILITY_SEED_GENERATION],
                    applies_to_agents=[DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_SEED_GENERATION]],
                    evidence={
                        "known_gaps_count": gaps_count,
                        "seeds_created": seeds_created,
                        "gap_rate": round(gap_rate, 4),
                    },
                )
            )

        summary6 = step_summaries.get(6) or {}
        needs_validation = self._safe_int(summary6.get("clusters_needing_validation"))
        clusters_created = self._safe_int(summary6.get("clusters_created"))
        if clusters_created > 0:
            validation_rate = needs_validation / max(1, clusters_created)
            if validation_rate >= 0.25:
                candidates.append(
                    LearningCandidate(
                        learning_key="step6:serp_validation_pressure",
                        source_capability=CAPABILITY_CLUSTERING,
                        source_agent=DEFAULT_AGENT_BY_CAPABILITY.get(CAPABILITY_CLUSTERING),
                        learning_type="quality_bottleneck",
                        polarity="negative",
                        title="Cluster coherence pressure",
                        detail=(
                            f"{validation_rate:.1%} of clusters need extra SERP validation "
                            f"({needs_validation}/{clusters_created})."
                        ),
                        recommendation="Tighten coarse clustering and primary keyword selection.",
                        confidence=self._confidence_from_count(clusters_created),
                        current_metric=validation_rate,
                        applies_to_capabilities=[CAPABILITY_CLUSTERING, CAPABILITY_SERP_VALIDATION],
                        applies_to_agents=[
                            DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_CLUSTERING],
                        ],
                        evidence={
                            "clusters_needing_validation": needs_validation,
                            "clusters_created": clusters_created,
                            "validation_rate": round(validation_rate, 4),
                        },
                    )
                )
        return candidates

    def _build_fallback_candidate(
        self,
        usable_by_keyword: dict[str, bool],
        total_keywords: int,
    ) -> LearningCandidate:
        usable = sum(1 for value in usable_by_keyword.values() if value)
        rate = usable / max(1, total_keywords)
        return LearningCandidate(
            learning_key="iteration:fallback",
            source_capability=CAPABILITY_PRIORITIZATION,
            source_agent=DEFAULT_AGENT_BY_CAPABILITY.get(CAPABILITY_PRIORITIZATION),
            learning_type="iteration_delta",
            polarity="neutral",
            title="Iteration completed with limited signal",
            detail=(
                f"Captured fallback learning with usable keyword rate {rate:.1%} "
                f"({usable}/{total_keywords})."
            ),
            recommendation="Collect one more iteration before making strategy shifts.",
            confidence=self._confidence_from_count(total_keywords),
            current_metric=rate,
            applies_to_capabilities=[CAPABILITY_PRIORITIZATION],
            applies_to_agents=[DEFAULT_AGENT_BY_CAPABILITY[CAPABILITY_PRIORITIZATION]],
            evidence={
                "usable_keywords": usable,
                "total_keywords": total_keywords,
            },
        )

    async def _apply_history(
        self,
        *,
        project_id: str,
        pipeline_run_id: str,
        iteration_index: int,
        candidates: list[LearningCandidate],
    ) -> None:
        result = await self.session.execute(
            select(DiscoveryIterationLearning)
            .where(DiscoveryIterationLearning.project_id == project_id)
            .order_by(DiscoveryIterationLearning.created_at.desc())
            .limit(500)
        )
        history_rows = [
            row
            for row in result.scalars()
            if not (
                row.pipeline_run_id == pipeline_run_id
                and row.iteration_index >= iteration_index
            )
        ]

        latest_by_key: dict[str, DiscoveryIterationLearning] = {}
        for row in history_rows:
            latest_by_key.setdefault(row.learning_key, row)

        for candidate in candidates:
            previous = latest_by_key.get(candidate.learning_key)
            if previous is None:
                candidate.status = STATUS_NEW
                candidate.baseline_metric = None
                candidate.delta_metric = None
                candidate.novelty_score = 1.0
                continue

            prev_metric = self._to_float(previous.current_metric)
            curr_metric = self._to_float(candidate.current_metric)
            candidate.baseline_metric = prev_metric
            if prev_metric is None or curr_metric is None:
                candidate.status = STATUS_CONFIRMED
                candidate.delta_metric = None
                candidate.novelty_score = 0.2
                continue

            delta = curr_metric - prev_metric
            candidate.delta_metric = round(delta, 4)
            threshold = 0.03
            if abs(delta) <= threshold:
                candidate.status = STATUS_CONFIRMED
            else:
                if candidate.polarity == "negative":
                    candidate.status = STATUS_REGRESSED if delta > threshold else STATUS_NEW
                else:
                    candidate.status = STATUS_REGRESSED if delta < -threshold else STATUS_NEW
            candidate.novelty_score = min(1.0, abs(delta) / 0.2)

    async def _rewrite_candidates(
        self,
        *,
        project_id: str,
        candidates: list[LearningCandidate],
    ) -> None:
        drafts = [
            LearningDraft(
                learning_key=item.learning_key,
                polarity=item.polarity,
                title=item.title,
                detail=item.detail,
                recommendation=item.recommendation,
                confidence=item.confidence,
                current_metric=item.current_metric,
                delta_metric=item.delta_metric,
            )
            for item in candidates
        ]
        if not drafts:
            return

        try:
            agent = DiscoveryLearningSummarizerAgent()
            output = await agent.run(
                DiscoveryLearningSummarizerInput(
                    project_context=f"project_id={project_id}",
                    drafts=drafts,
                )
            )
        except Exception:
            logger.warning(
                "Discovery learning summarizer failed; keeping deterministic drafts",
                extra={"project_id": project_id, "draft_count": len(drafts)},
            )
            return

        rewrites = {item.learning_key: item for item in output.rewrites}
        for candidate in candidates:
            rewrite = rewrites.get(candidate.learning_key)
            if rewrite is None:
                continue
            candidate.title = rewrite.title.strip() or candidate.title
            candidate.detail = rewrite.detail.strip() or candidate.detail
            if rewrite.recommendation is not None:
                candidate.recommendation = rewrite.recommendation.strip() or candidate.recommendation
            if rewrite.confidence is not None:
                candidate.confidence = max(0.0, min(1.0, float(rewrite.confidence)))

    def _keyword_archetype(self, keyword: Keyword, seed_topics: dict[str, SeedTopic]) -> str:
        signals = keyword.discovery_signals if isinstance(keyword.discovery_signals, dict) else {}
        if bool(signals.get("is_comparison")) or signals.get("comparison_target"):
            return "comparison"
        if bool(signals.get("has_two_entities")):
            return "integration_pair"
        if signals.get("workflow_verb"):
            return "workflow_action"

        seed = seed_topics.get(str(keyword.seed_topic_id)) if keyword.seed_topic_id else None
        pillar = str(seed.pillar_type if seed else "").strip().lower()
        if "pain" in pillar:
            return "pain_point"
        if "feature" in pillar:
            return "feature"
        if "audience" in pillar:
            return "audience_modifier"
        if "core" in pillar or "offer" in pillar:
            return "core_offer"
        return "general"

    def _topic_fit_tier(self, topic: Topic | None) -> str | None:
        if topic is None:
            return None
        value = topic.fit_tier
        return str(value).strip().lower() if value else None

    def _normalize_reason(self, reason: str) -> str:
        clean = str(reason or "").strip().lower()
        if ":" in clean:
            clean = clean.split(":", 1)[0].strip()
        return re.sub(r"[^a-z0-9_]+", "_", clean).strip("_") or "unknown"

    def _humanize_token(self, token: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(token or "").strip().lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.title() if cleaned else "Unknown"

    def _humanize_pattern(self, pattern: str) -> str:
        left, _, right = str(pattern or "").partition("|")
        url_type = self._humanize_token(left).lower()
        intent = self._humanize_token(right).lower()
        return f"URL type '{url_type}' with '{intent}' intent"

    def _humanize_rejection_reason(self, reason: str) -> str:
        return self._humanize_token(reason).lower()

    def _capability_for_rejection_reason(self, reason: str) -> str:
        if "serp" in reason or "domain_diversity" in reason:
            return CAPABILITY_SERP_VALIDATION
        if "intent" in reason:
            return CAPABILITY_INTENT_CLASSIFICATION
        if "difficulty" in reason:
            return CAPABILITY_KEYWORD_EXPANSION
        return CAPABILITY_PRIORITIZATION

    def _recommendation_for_rejection_reason(self, reason: str) -> str:
        if "serp" in reason:
            return "Prefer less saturated intents and diversify supporting evidence keywords."
        if "intent" in reason:
            return "Improve intent alignment before clustering and prioritization."
        if "difficulty" in reason:
            return "Bias expansion towards lower-difficulty terms and narrower variants."
        return "Refine fit gating and scope filters before the next iteration."

    def _confidence_from_count(self, count: int) -> float:
        return round(min(0.95, 0.4 + (max(count, 0) / 120.0)), 4)

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
