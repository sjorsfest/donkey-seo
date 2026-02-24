"""Closed-loop discovery supervisor for topic opportunity generation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.brand import BrandProfile
from app.models.discovery_snapshot import DiscoveryTopicSnapshot
from app.models.generated_dtos import (
    DiscoveryTopicSnapshotCreateDTO,
    PipelineRunPatchDTO,
)
from app.models.keyword import Keyword
from app.models.pipeline import PipelineRun, StepExecution
from app.models.project import Project
from app.models.topic import Topic
from app.schemas.pipeline import ContentPipelineConfig, DiscoveryLoopConfig
from app.services.run_strategy import (
    build_goal_intent_profile,
    classify_intent_alignment,
    resolve_run_strategy,
)
from app.services.discovery_learning import DiscoveryLearningService
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

ExecuteStepFn = Callable[[PipelineRun, int], Awaitable[StepExecution]]
MarkStepFn = Callable[[str, int, str], Awaitable[dict[str, Any]]]
DispatchTopicsFn = Callable[[list[str], ContentPipelineConfig], Awaitable[None]]

DISCOVERY_STEPS = (1, 2, 3, 4, 5, 6, 7)
SCOPE_SEQUENCE = ("strict", "balanced_adjacent", "broad_education")
FIT_PROFILE_SEQUENCE = ("aggressive", "moderate", "lenient")
LOW_ICP_THRESHOLD = 0.10


@dataclass(slots=True)
class TopicDecision:
    """Single topic decision made during discovery."""

    source_topic_id: str | None
    topic_name: str
    fit_tier: str | None
    fit_score: float | None
    keyword_difficulty: float | None
    domain_diversity: float | None
    validated_intent: str | None
    validated_page_type: str | None
    top_domains: list[str] = field(default_factory=list)
    decision: str = "rejected"
    rejection_reasons: list[str] = field(default_factory=list)
    is_hard_excluded: bool = False
    is_very_low_icp: bool = False


@dataclass(slots=True)
class DiscoveryLoopResult:
    """Outcome of the discovery loop."""

    success: bool
    accepted_topic_ids: list[str]
    accepted_count: int
    target_count: int
    iterations_completed: int
    content_config: ContentPipelineConfig


@dataclass(slots=True)
class AcceptedTopicState:
    """Cumulative accepted topic tracked across discovery iterations."""

    topic_name: str
    source_topic_id: str | None = None


class DiscoveryLoopSupervisor:
    """Runs step 1-7 in adaptive loops until enough accepted topics exist."""

    def __init__(
        self,
        session: AsyncSession,
        project_id: str,
        run: PipelineRun,
        task_manager: TaskManager,
    ) -> None:
        self.session = session
        self.project_id = project_id
        self.run = run
        self.task_manager = task_manager

    async def run_loop(
        self,
        *,
        execute_step: ExecuteStepFn,
        mark_step_completed: MarkStepFn,
        dispatch_accepted_topics: DispatchTopicsFn,
    ) -> DiscoveryLoopResult:
        """Execute the adaptive discovery loop."""
        task_id = str(self.run.id)
        steps_config = dict(self.run.steps_config or {})
        discovery = DiscoveryLoopConfig.model_validate(steps_config.get("discovery") or {})
        content = ContentPipelineConfig.model_validate(steps_config.get("content") or {})

        base_strategy_payload = dict(steps_config.get("strategy") or {})
        immutable_excludes = {
            str(topic).strip().lower()
            for topic in base_strategy_payload.get("exclude_topics", [])
            if str(topic).strip()
        }
        dynamic_excludes: list[str] = []
        target_count = await self._resolve_target_count(discovery, base_strategy_payload)

        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage=(
                f"Discovery loop started: target {target_count} accepted topics "
                f"in <= {discovery.max_iterations} iterations"
            ),
            project_id=self.project_id,
            current_step=1,
            current_step_name="seed_topics",
            error_message=None,
        )

        accepted_topics_by_key: dict[str, AcceptedTopicState] = {}
        accepted_topic_ids: list[str] = []
        accepted_topic_names: list[str] = []
        iterations_completed = 0
        for iteration in range(1, discovery.max_iterations + 1):
            iterations_completed = iteration
            iteration_step_summaries: dict[int, dict[str, Any]] = {}
            strategy_payload = self._build_iteration_strategy_payload(
                base_strategy_payload=base_strategy_payload,
                iteration=iteration,
                dynamic_excludes=dynamic_excludes,
            )
            await self._update_steps_config_for_iteration(
                iteration_index=iteration,
                strategy_payload=strategy_payload,
            )

            await self.task_manager.set_task_state(
                task_id=task_id,
                status="running",
                stage=f"Discovery loop iteration {iteration}/{discovery.max_iterations}",
                current_step=1,
                current_step_name="seed_topics",
                error_message=None,
            )

            for step_num in DISCOVERY_STEPS:
                execution = await execute_step(self.run, step_num)
                self.run.paused_at_step = step_num
                if execution.status == "completed":
                    await mark_step_completed(
                        task_id,
                        step_num,
                        str(execution.step_name or f"step_{step_num}"),
                    )
                iteration_step_summaries = self._collect_iteration_step_summaries(
                    iteration_step_summaries,
                    step_num=step_num,
                    execution=execution,
                )

            decisions = await self._evaluate_topic_decisions(
                iteration_index=iteration,
                discovery=discovery,
            )
            await self._persist_snapshots(iteration, decisions)
            await self._persist_iteration_learnings(
                iteration_index=iteration,
                decisions=decisions,
                step_summaries=iteration_step_summaries,
            )

            accepted_topics_by_key = self._merge_accepted_topics(
                current_pool=accepted_topics_by_key,
                decisions=decisions,
            )
            accepted_topic_ids = self._collect_selected_topic_ids(accepted_topics_by_key)
            accepted_topic_names = self._collect_selected_topic_names(accepted_topics_by_key)
            await self._update_steps_config_selected_topics(
                selected_topic_ids=accepted_topic_ids,
                selected_topic_names=accepted_topic_names,
                accepted_topic_count=len(accepted_topics_by_key),
                iteration_index=iteration,
            )
            await dispatch_accepted_topics(accepted_topic_ids, content)

            accepted_count_current = sum(
                1 for decision in decisions if decision.decision == "accepted"
            )
            accepted_count = len(accepted_topics_by_key)
            if accepted_count >= target_count:
                await self.task_manager.set_task_state(
                    task_id=task_id,
                    status="running",
                    stage=(
                        f"Discovery complete in iteration {iteration}: "
                        f"{accepted_count}/{target_count} accepted"
                    ),
                    current_step=7,
                    current_step_name="serp_validation",
                    error_message=None,
                )
                return DiscoveryLoopResult(
                    success=True,
                    accepted_topic_ids=accepted_topic_ids,
                    accepted_count=accepted_count,
                    target_count=target_count,
                    iterations_completed=iterations_completed,
                    content_config=content,
                )

            dynamic_excludes = self._next_dynamic_excludes(
                current_dynamic_excludes=dynamic_excludes,
                decisions=decisions,
                immutable_excludes=immutable_excludes,
            )
            logger.info(
                "Discovery iteration incomplete",
                extra={
                    "project_id": self.project_id,
                    "run_id": str(self.run.id),
                    "iteration": iteration,
                    "accepted_current": accepted_count_current,
                    "accepted_cumulative": accepted_count,
                    "target": target_count,
                    "dynamic_excludes": len(dynamic_excludes),
                },
            )

        final_count = len(accepted_topics_by_key)
        raise RuntimeError(
            "Insufficient accepted topics after discovery loop "
            f"({final_count}/{target_count} accepted across {iterations_completed} iterations). "
            "Try broadening topic scope, adding include_topics, or lowering difficulty constraints."
        )

    async def _load_project(self) -> Project:
        result = await self.session.execute(select(Project).where(Project.id == self.project_id))
        return result.scalar_one()

    async def _load_brand(self) -> BrandProfile | None:
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == self.project_id)
        )
        return result.scalar_one_or_none()

    async def _resolve_target_count(
        self,
        discovery: DiscoveryLoopConfig,
        strategy_payload: dict[str, Any],
    ) -> int:
        if discovery.min_eligible_topics is not None:
            return max(1, discovery.min_eligible_topics)

        project = await self._load_project()
        brand = await self._load_brand()
        strategy = resolve_run_strategy(
            strategy_payload=strategy_payload,
            brand=brand,
            primary_goal=project.primary_goal,
        )
        return strategy.eligible_target()

    def _build_iteration_strategy_payload(
        self,
        *,
        base_strategy_payload: dict[str, Any],
        iteration: int,
        dynamic_excludes: list[str],
    ) -> dict[str, Any]:
        idx = min(max(iteration - 1, 0), len(SCOPE_SEQUENCE) - 1)
        scope_mode = SCOPE_SEQUENCE[idx]
        fit_profile = FIT_PROFILE_SEQUENCE[idx]

        payload = dict(base_strategy_payload)
        payload["scope_mode"] = scope_mode
        payload["fit_threshold_profile"] = fit_profile
        existing_excludes = [
            str(topic).strip()
            for topic in payload.get("exclude_topics", [])
            if str(topic).strip()
        ]
        merged_excludes = self._dedupe(existing_excludes + dynamic_excludes)
        payload["exclude_topics"] = merged_excludes
        return payload

    async def _update_steps_config_for_iteration(
        self,
        *,
        iteration_index: int,
        strategy_payload: dict[str, Any],
    ) -> None:
        current_steps = dict(self.run.steps_config or {})
        current_steps["iteration_index"] = iteration_index
        current_steps["strategy"] = strategy_payload
        self.run.patch(
            self.session,
            PipelineRunPatchDTO.from_partial({"steps_config": current_steps}),
        )
        await self.session.commit()

    async def _persist_iteration_learnings(
        self,
        *,
        iteration_index: int,
        decisions: list[TopicDecision],
        step_summaries: dict[int, dict[str, Any]],
    ) -> None:
        service = DiscoveryLearningService(self.session)
        await service.synthesize_and_persist(
            project_id=self.project_id,
            pipeline_run_id=str(self.run.id),
            iteration_index=iteration_index,
            decisions=decisions,
            step_summaries=step_summaries,
        )

    def _collect_iteration_step_summaries(
        self,
        existing: dict[int, dict[str, Any]] | None,
        *,
        step_num: int,
        execution: StepExecution,
    ) -> dict[int, dict[str, Any]]:
        summaries = dict(existing or {})
        raw_summary = execution.result_summary if isinstance(execution.result_summary, dict) else {}
        summaries[step_num] = dict(raw_summary)
        return summaries

    async def _update_steps_config_selected_topics(
        self,
        *,
        selected_topic_ids: list[str],
        selected_topic_names: list[str],
        accepted_topic_count: int,
        iteration_index: int,
    ) -> None:
        current_steps = dict(self.run.steps_config or {})
        current_steps["selected_topic_ids"] = selected_topic_ids
        current_steps["selected_topic_names"] = selected_topic_names
        current_steps["accepted_topic_count"] = accepted_topic_count
        current_steps["iteration_index"] = iteration_index
        self.run.patch(
            self.session,
            PipelineRunPatchDTO.from_partial({"steps_config": current_steps}),
        )
        await self.session.commit()

    async def _evaluate_topic_decisions(
        self,
        *,
        iteration_index: int,
        discovery: DiscoveryLoopConfig,
    ) -> list[TopicDecision]:
        steps_config_raw = getattr(self.run, "steps_config", None)
        steps_config = steps_config_raw if isinstance(steps_config_raw, dict) else {}
        strategy_payload = (
            steps_config.get("strategy")
            if isinstance(steps_config.get("strategy"), dict)
            else None
        )
        primary_goal = (
            steps_config.get("primary_goal")
            if isinstance(steps_config.get("primary_goal"), str)
            else None
        )
        run_strategy = resolve_run_strategy(
            strategy_payload=strategy_payload,
            brand=None,
            primary_goal=primary_goal,
        )
        goal_intent_profile = build_goal_intent_profile(run_strategy.conversion_intents)

        topic_result = await self.session.execute(
            select(Topic).where(Topic.project_id == self.project_id)
        )
        all_topics = list(topic_result.scalars())
        candidates = [
            topic
            for topic in all_topics
            if str(topic.fit_tier or "").lower() in {"primary", "secondary"}
        ]
        if not candidates:
            return []

        primary_ids = [topic.primary_keyword_id for topic in candidates if topic.primary_keyword_id]
        evidence_ids = [
            topic.serp_evidence_keyword_id
            for topic in candidates
            if topic.serp_evidence_keyword_id
        ]
        keyword_ids = list({
            str(keyword_id)
            for keyword_id in [*primary_ids, *evidence_ids]
            if keyword_id
        })
        keyword_by_id: dict[str, Keyword] = {}
        if keyword_ids:
            keyword_result = await self.session.execute(
                select(Keyword).where(
                    Keyword.id.in_([keyword_id for keyword_id in keyword_ids]),
                )
            )
            keyword_by_id = {str(keyword.id): keyword for keyword in keyword_result.scalars()}

        decisions: list[TopicDecision] = []
        for topic in candidates:
            fit_tier = str(topic.fit_tier or "").lower()
            is_secondary_tier = fit_tier == "secondary"
            max_keyword_difficulty = (
                max(0.0, discovery.max_keyword_difficulty - 10.0)
                if is_secondary_tier
                else discovery.max_keyword_difficulty
            )
            min_domain_diversity = (
                min(1.0, discovery.min_domain_diversity + 0.10)
                if is_secondary_tier
                else discovery.min_domain_diversity
            )
            min_serp_intent_confidence = (
                min(1.0, discovery.min_serp_intent_confidence + 0.10)
                if is_secondary_tier
                else discovery.min_serp_intent_confidence
            )
            primary_keyword = (
                keyword_by_id.get(str(topic.primary_keyword_id))
                if topic.primary_keyword_id
                else None
            )
            evidence_keyword_id = topic.serp_evidence_keyword_id
            evidence_keyword = (
                keyword_by_id.get(str(evidence_keyword_id))
                if evidence_keyword_id
                else None
            )
            keyword = primary_keyword
            if not (
                keyword
                and isinstance(keyword.serp_top_results, list)
                and len(keyword.serp_top_results) > 0
            ):
                keyword = evidence_keyword
            top_results = (
                list(keyword.serp_top_results or [])[:10]
                if keyword and isinstance(keyword.serp_top_results, list)
                else []
            )
            top_domains = self._extract_top_domains(top_results)
            top10_count = len(top_results)
            domain_diversity = (
                (len(set(top_domains)) / max(1, top10_count))
                if top10_count > 0
                else 0.0
            )
            keyword_difficulty = (
                primary_keyword.difficulty
                if primary_keyword and primary_keyword.difficulty is not None
                else topic.avg_difficulty
            )
            topic_market_mode = (
                topic.market_mode
                or "established_category"
            )
            is_workflow_topic = topic_market_mode == "fragmented_workflow"

            rejection_reasons: list[str] = []
            if discovery.require_serp_gate:
                if is_workflow_topic:
                    servedness = topic.serp_servedness_score
                    competitor_density = topic.serp_competitor_density
                    serp_intent_confidence = (
                        self._to_float(topic.serp_intent_confidence) or 0.0
                    )
                    if servedness is None or competitor_density is None:
                        rejection_reasons.append("missing_cluster_serp_evidence")
                    if keyword_difficulty is None:
                        rejection_reasons.append("missing_keyword_difficulty")
                    if (
                        keyword_difficulty is not None
                        and keyword_difficulty > max_keyword_difficulty
                    ):
                        rejection_reasons.append(
                            "keyword_difficulty_above_threshold:"
                            f"{keyword_difficulty:.2f}>{max_keyword_difficulty:.2f}"
                        )
                    if (
                        servedness is not None
                        and competitor_density is not None
                        and servedness >= discovery.max_serp_servedness
                        and competitor_density >= discovery.max_serp_competitor_density
                    ):
                        rejection_reasons.append(
                            "workflow_serp_saturated:"
                            f"servedness={servedness:.2f},density={competitor_density:.2f}"
                        )
                    if (
                        discovery.require_intent_match
                        and serp_intent_confidence < min_serp_intent_confidence
                    ):
                        rejection_reasons.append(
                            "serp_intent_confidence_below_threshold:"
                            f"{serp_intent_confidence:.2f}<{min_serp_intent_confidence:.2f}"
                        )
                else:
                    if keyword is None:
                        rejection_reasons.append("missing_primary_keyword")
                    if not top_results:
                        rejection_reasons.append("missing_serp_results")
                    if keyword_difficulty is None:
                        rejection_reasons.append("missing_keyword_difficulty")
                    if (
                        keyword_difficulty is not None
                        and keyword_difficulty > max_keyword_difficulty
                    ):
                        rejection_reasons.append(
                            "keyword_difficulty_above_threshold:"
                            f"{keyword_difficulty:.2f}>{max_keyword_difficulty:.2f}"
                        )
                    if domain_diversity < min_domain_diversity:
                        rejection_reasons.append(
                            "domain_diversity_below_threshold:"
                            f"{domain_diversity:.2f}<{min_domain_diversity:.2f}"
                        )

            if (
                not is_workflow_topic
                and
                discovery.require_intent_match
                and keyword
                and isinstance(keyword.serp_mismatch_flags, list)
            ):
                observed_intent = keyword.validated_intent or topic.dominant_intent
                alignment = classify_intent_alignment(observed_intent, goal_intent_profile)
                fit_score = self._to_float(topic.fit_score) or 0.0
                intent_mismatch_flag = "intent_mismatch" in keyword.serp_mismatch_flags
                should_hard_reject = (
                    alignment == "off_goal"
                    and fit_tier != "primary"
                    and fit_score < 0.75
                )
                if should_hard_reject:
                    rejection_reasons.append(
                        "goal_intent_mismatch:"
                        f"{(observed_intent or 'unknown').lower()}"
                        f" not aligned with {goal_intent_profile.profile_name}"
                    )
                elif intent_mismatch_flag and alignment == "off_goal" and fit_tier != "primary":
                    rejection_reasons.append("intent_mismatch_off_goal")

            if is_secondary_tier and rejection_reasons:
                rejection_reasons.append("secondary_tier_strict_gate")

            diagnostics = (
                topic.prioritization_diagnostics
                if isinstance(topic.prioritization_diagnostics, dict)
                else {}
            )
            fit_reasons = [
                str(item)
                for item in diagnostics.get("fit_reasons", [])
                if item is not None
            ]
            icp_relevance = self._parse_icp_relevance(fit_reasons)
            is_hard_excluded = topic.hard_exclusion_reason is not None or any(
                reason.lower().startswith("hard exclusion:")
                for reason in fit_reasons
            )
            is_very_low_icp = icp_relevance is not None and icp_relevance <= LOW_ICP_THRESHOLD

            decision = TopicDecision(
                source_topic_id=str(topic.id),
                topic_name=topic.name,
                fit_tier=topic.fit_tier,
                fit_score=self._to_float(topic.fit_score),
                keyword_difficulty=keyword_difficulty,
                domain_diversity=round(domain_diversity, 4),
                validated_intent=keyword.validated_intent if keyword else None,
                validated_page_type=keyword.validated_page_type if keyword else None,
                top_domains=top_domains,
                decision="accepted" if not rejection_reasons else "rejected",
                rejection_reasons=rejection_reasons,
                is_hard_excluded=is_hard_excluded,
                is_very_low_icp=is_very_low_icp,
            )
            decisions.append(decision)

        logger.info(
            "Discovery topic evaluation complete",
            extra={
                "project_id": self.project_id,
                "run_id": str(self.run.id),
                "iteration": iteration_index,
                "candidates": len(candidates),
                "accepted": sum(1 for item in decisions if item.decision == "accepted"),
                "rejected": sum(1 for item in decisions if item.decision == "rejected"),
            },
        )
        return decisions

    async def _persist_snapshots(
        self,
        iteration_index: int,
        decisions: list[TopicDecision],
    ) -> None:
        for decision in decisions:
            DiscoveryTopicSnapshot.create(
                self.session,
                DiscoveryTopicSnapshotCreateDTO(
                    project_id=self.project_id,
                    pipeline_run_id=str(self.run.id),
                    iteration_index=iteration_index,
                    source_topic_id=decision.source_topic_id,
                    topic_name=decision.topic_name,
                    fit_tier=decision.fit_tier,
                    fit_score=decision.fit_score,
                    keyword_difficulty=decision.keyword_difficulty,
                    domain_diversity=decision.domain_diversity,
                    validated_intent=decision.validated_intent,
                    validated_page_type=decision.validated_page_type,
                    top_domains=decision.top_domains,
                    decision=decision.decision,
                    rejection_reasons=decision.rejection_reasons,
                ),
            )
        await self.session.commit()

    def _next_dynamic_excludes(
        self,
        *,
        current_dynamic_excludes: list[str],
        decisions: list[TopicDecision],
        immutable_excludes: set[str],
    ) -> list[str]:
        promoted = list(current_dynamic_excludes)
        existing = {item.strip().lower() for item in promoted if item.strip()}
        for decision in decisions:
            if not (decision.is_hard_excluded or decision.is_very_low_icp):
                continue
            normalized = decision.topic_name.strip().lower()
            if not normalized or normalized in immutable_excludes or normalized in existing:
                continue
            promoted.append(decision.topic_name)
            existing.add(normalized)
        return promoted

    def _extract_top_domains(self, top_results: list[dict[str, Any]]) -> list[str]:
        domains: list[str] = []
        seen: set[str] = set()
        for result in top_results:
            domain = str(result.get("domain") or "").strip().lower()
            if not domain or domain in seen:
                continue
            seen.add(domain)
            domains.append(domain)
        return domains

    def _parse_icp_relevance(self, fit_reasons: list[str]) -> float | None:
        for reason in fit_reasons:
            match = re.search(r"ICP relevance:\s*(\d{1,3})%", reason, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                return max(0.0, min(1.0, int(match.group(1)) / 100.0))
            except (TypeError, ValueError):
                return None
        return None

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _merge_accepted_topics(
        self,
        *,
        current_pool: dict[str, AcceptedTopicState],
        decisions: list[TopicDecision],
    ) -> dict[str, AcceptedTopicState]:
        merged = dict(current_pool)
        for decision in decisions:
            if decision.decision != "accepted":
                continue
            key = self._accepted_topic_key(decision.topic_name, decision.source_topic_id)
            if not key:
                continue

            existing = merged.get(key)
            cleaned_topic_name = decision.topic_name.strip()
            topic_name = (existing.topic_name if existing else "") or cleaned_topic_name
            source_topic_id = (
                str(decision.source_topic_id)
                if decision.source_topic_id
                else (existing.source_topic_id if existing else None)
            )
            merged[key] = AcceptedTopicState(
                topic_name=topic_name,
                source_topic_id=source_topic_id,
            )
        return merged

    def _accepted_topic_key(self, topic_name: str, source_topic_id: str | None) -> str | None:
        normalized_name = re.sub(r"\s+", " ", str(topic_name).strip().lower())
        if normalized_name:
            return normalized_name
        if source_topic_id:
            return f"id:{source_topic_id}"
        return None

    def _collect_selected_topic_ids(
        self,
        accepted_topics_by_key: dict[str, AcceptedTopicState],
    ) -> list[str]:
        selected_topic_ids: list[str] = []
        seen_ids: set[str] = set()
        for topic in accepted_topics_by_key.values():
            if not topic.source_topic_id:
                continue
            if topic.source_topic_id in seen_ids:
                continue
            seen_ids.add(topic.source_topic_id)
            selected_topic_ids.append(topic.source_topic_id)
        return selected_topic_ids

    def _collect_selected_topic_names(
        self,
        accepted_topics_by_key: dict[str, AcceptedTopicState],
    ) -> list[str]:
        selected_topic_names: list[str] = []
        seen_names: set[str] = set()
        for topic in accepted_topics_by_key.values():
            name = topic.topic_name.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            selected_topic_names.append(name)
        return selected_topic_names

    def _dedupe(self, items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = str(item).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out
