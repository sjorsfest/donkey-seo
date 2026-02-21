"""Step 12: Content Brief Generation.

Generates writer-ready briefs for prioritized topics.
Includes URL slug generation and cannibalization guardrails.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy import or_, select

from app.agents.brief_generator import BriefGeneratorAgent, BriefGeneratorInput
from app.models.brand import BrandProfile
from app.models.content import ContentBrief
from app.models.generated_dtos import ContentBriefCreateDTO, ContentBriefPatchDTO
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class BriefInput:
    """Input for Step 12."""

    project_id: str
    topic_ids: list[str] | None = None  # Specific topics, or None for all prioritized
    max_briefs: int = 20  # Limit number of briefs to generate
    posts_per_week: int = 1
    preferred_weekdays: list[int] | None = None  # 0=Mon ... 6=Sun
    min_lead_days: int = 7
    publication_start_date: date | None = None
    use_llm_timing_hints: bool = True
    llm_timing_flex_days: int = 14
    include_zero_data_topics: bool = True
    zero_data_topic_share: float = 0.2
    zero_data_fit_score_min: float = 0.65


@dataclass
class BriefOutput:
    """Output from Step 12."""

    briefs_generated: int
    briefs_with_warnings: int
    briefs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PublicationScheduleConfig:
    """Publication scheduling controls for content calendar assignment."""

    posts_per_week: int
    weekdays: list[int]
    min_lead_days: int
    start_date: date | None
    use_llm_timing_hints: bool
    llm_timing_flex_days: int


class Step12BriefService(BaseStepService[BriefInput, BriefOutput]):
    """Step 12: Content Brief Generation.

    Generates comprehensive content briefs with:
    - URL slug proposal with collision detection
    - Cannibalization guardrails (do_not_target keywords)
    - Internal linking recommendations
    - Detailed content outlines

    Key features:
    - URL slug generated from primary keyword + page type
    - Collision check against existing inventory (if Step 9 ran)
    - do_not_target list prevents keyword overlap
    - Overlap status indicates if Step 10 ran
    """

    step_number = 12
    step_name = "content_brief"
    is_optional = False

    async def _validate_preconditions(self, input_data: BriefInput) -> None:
        """Validate required upstream artifacts exist."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        # Check prioritized topics exist
        topics_result = await self.session.execute(
            select(Topic).where(
                Topic.project_id == input_data.project_id,
                Topic.priority_rank.isnot(None),
            ).limit(1)
        )
        if not topics_result.scalars().first():
            raise ValueError("No prioritized topics found. Run Step 7 first.")

    async def _execute(self, input_data: BriefInput) -> BriefOutput:
        """Execute content brief generation."""
        # Load brand + run strategy
        strategy = await self.get_run_strategy()

        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()

        await self._update_progress(5, "Loading prioritized topics...")

        # Load topics to generate briefs for
        if input_data.topic_ids:
            # Specific topics requested
            topics_result = await self.session.execute(
                select(Topic).where(
                    Topic.project_id == input_data.project_id,
                    Topic.id.in_(input_data.topic_ids),
                )
            )
        else:
            # All prioritized topics, sorted by priority
            topics_result = await self.session.execute(
                select(Topic).where(
                    Topic.project_id == input_data.project_id,
                    Topic.priority_rank.isnot(None),
                ).order_by(Topic.priority_rank)
            )

        all_topics = list(topics_result.scalars())
        eligible_topics_all = [topic for topic in all_topics if self._is_topic_eligible(topic)]
        skipped_ineligible = len(all_topics) - len(eligible_topics_all)

        selected_topics = eligible_topics_all
        zero_data_candidates = 0
        zero_data_selected = 0
        if input_data.topic_ids:
            selected_topics = eligible_topics_all[:input_data.max_briefs]
        else:
            primary_keywords_by_topic_id = await self._load_primary_keywords_for_topics(
                eligible_topics_all
            )
            zero_data_candidates = sum(
                1
                for topic in eligible_topics_all
                if self._is_zero_data_topic_candidate(
                    topic=topic,
                    primary_keyword=primary_keywords_by_topic_id.get(str(topic.id)),
                    min_fit_score=input_data.zero_data_fit_score_min,
                )
            )
            selected_topics = self._select_topics_for_briefs(
                topics=eligible_topics_all,
                primary_keywords_by_topic_id=primary_keywords_by_topic_id,
                input_data=input_data,
            )
            zero_data_selected = sum(
                1
                for topic in selected_topics
                if self._is_zero_data_topic_candidate(
                    topic=topic,
                    primary_keyword=primary_keywords_by_topic_id.get(str(topic.id)),
                    min_fit_score=input_data.zero_data_fit_score_min,
                )
            )

        existing_brief_topic_ids = await self._load_existing_brief_topic_ids(
            [str(topic.id) for topic in selected_topics]
        )
        topics_to_generate = [
            topic for topic in selected_topics if str(topic.id) not in existing_brief_topic_ids
        ]
        skipped_existing = len(selected_topics) - len(topics_to_generate)
        logger.info(
            "Brief generation starting",
            extra={
                "project_id": input_data.project_id,
                "topic_count": len(all_topics),
                "eligible_topics": len(eligible_topics_all),
                "selected_topics": len(selected_topics),
                "skipped_ineligible": skipped_ineligible,
                "skipped_existing": skipped_existing,
                "zero_data_candidates": zero_data_candidates,
                "zero_data_selected": zero_data_selected,
            },
        )

        if not topics_to_generate:
            return BriefOutput(
                briefs_generated=0,
                briefs_with_warnings=0,
            )

        # Prepare brand context
        brand_context = self._build_brand_context(brand)
        money_pages = self._extract_money_pages(brand)

        # Infer cannibalization coverage from stored topic fields.
        step_10_ran = await self._has_cannibalization_signals(input_data.project_id)

        # Load existing URLs for collision check (if Step 9 ran)
        existing_urls = await self._load_existing_urls(input_data.project_id)

        await self._update_progress(10, f"Generating {len(topics_to_generate)} briefs...")

        # Generate briefs
        agent = BriefGeneratorAgent()
        output_briefs = []
        briefs_with_warnings = 0
        today = date.today()
        schedule_config = self._build_publication_schedule_config(input_data=input_data)
        available_slots = self._build_publication_slots(
            topic_count=len(topics_to_generate),
            today=today,
            config=schedule_config,
        )

        for i, topic in enumerate(topics_to_generate):
            progress = 10 + int((i / len(topics_to_generate)) * 85)
            await self._update_progress(
                progress,
                f"Generating brief {i + 1}/{len(topics_to_generate)}: {topic.name}"
            )

            # Load keywords for this topic
            keywords_result = await self.session.execute(
                select(Keyword).where(Keyword.topic_id == topic.id)
            )
            keywords = list(keywords_result.scalars())

            # Get primary keyword
            primary_kw = next(
                (kw for kw in keywords if kw.id == topic.primary_keyword_id),
                keywords[0] if keywords else None
            )

            if not primary_kw:
                continue

            serp_profile = self._resolve_brief_serp_profile(primary_kw, topic)
            resolved_intent = serp_profile["search_intent"]
            resolved_page_type = serp_profile["page_type"]
            serp_features = serp_profile["serp_features"]
            competitors_content_types = serp_profile["competitors_content_types"]
            serp_mismatch_flags = serp_profile["serp_mismatch_flags"]

            # Generate URL slug
            url_slug = self._generate_url_slug(
                primary_kw.keyword,
                resolved_page_type,
            )

            # Check for URL collision
            collision_status = self._check_url_collision(url_slug, existing_urls)

            # Build do_not_target list (cannibalization guardrails)
            do_not_target = await self._build_do_not_target(
                topic,
                keywords,
                input_data.project_id,
            )

            # Determine overlap status
            overlap_status = "checked" if step_10_ran else "unknown"

            # Check if there are warnings
            has_warnings = (
                collision_status != "safe"
                or len(do_not_target) > 0
                or overlap_status == "unknown"
                or len(serp_mismatch_flags) > 0
            )
            if has_warnings:
                briefs_with_warnings += 1

            try:
                # Generate brief with LLM
                agent_input = BriefGeneratorInput(
                    topic_name=topic.name,
                    primary_keyword=primary_kw.keyword,
                    supporting_keywords=[
                        kw.keyword for kw in keywords if kw.id != primary_kw.id
                    ][:15],
                    search_intent=resolved_intent,
                    page_type=resolved_page_type,
                    funnel_stage=topic.funnel_stage or "tofu",
                    brand_context=brand_context,
                    competitors_content_types=competitors_content_types,
                    serp_features=serp_features,
                    money_pages=topic.target_money_pages or money_pages[:3],
                    conversion_intents=strategy.conversion_intents,
                    recommended_publish_order=topic.recommended_publish_order,
                )

                output = await agent.run(agent_input)
                brief_data = output.brief
                proposed_publication_date = self._select_proposed_publication_date(
                    llm_date=brief_data.proposed_publication_date,
                    available_slots=available_slots,
                    today=today,
                    config=schedule_config,
                )

                output_briefs.append({
                    "topic_id": str(topic.id),
                    "topic_name": topic.name,
                    "primary_keyword": primary_kw.keyword,
                    "search_intent": resolved_intent,
                    "page_type": resolved_page_type,
                    "funnel_stage": topic.funnel_stage,
                    "serp_features": serp_features,
                    "competitors_content_types": competitors_content_types,
                    "serp_mismatch_flags": serp_mismatch_flags,
                    # URL architecture
                    "proposed_url_slug": url_slug,
                    "url_collision_check": collision_status,
                    # Cannibalization guardrails
                    "do_not_target": do_not_target,
                    "do_not_cover_intent": self._get_conflicting_intent(topic),
                    "overlap_status": overlap_status,
                    # Brief content
                    "working_titles": brief_data.working_titles,
                    "target_audience": brief_data.target_audience,
                    "reader_job_to_be_done": brief_data.reader_job_to_be_done,
                    "outline": [
                        {
                            "heading": s.heading,
                            "level": s.level,
                            "purpose": s.purpose,
                            "key_points": s.key_points,
                            "supporting_keywords": s.supporting_keywords,
                        }
                        for s in brief_data.outline
                    ],
                    "supporting_keywords": [
                        kw.keyword for kw in keywords if kw.id != primary_kw.id
                    ],
                    "supporting_keywords_map": self._build_keyword_section_map(
                        brief_data.outline,
                        [kw.keyword for kw in keywords if kw.id != primary_kw.id],
                    ),
                    "examples_required": brief_data.examples_required,
                    "faq_questions": brief_data.faq_questions,
                    "internal_links_out": self._build_internal_links(topic, keywords),
                    "money_page_links": self._build_money_page_links(topic, money_pages),
                    "meta_title_guidelines": brief_data.meta_title_template,
                    "meta_description_guidelines": brief_data.meta_description_template,
                    "target_word_count": {
                        "min": brief_data.target_word_count_min,
                        "max": brief_data.target_word_count_max,
                    },
                    "must_include_sections": brief_data.must_include_sections,
                    "recommended_schema_type": brief_data.recommended_schema_type,
                    "proposed_publication_date": proposed_publication_date,
                    "has_warnings": has_warnings,
                    "warnings": self._collect_warnings(
                        collision_status,
                        do_not_target,
                        overlap_status,
                        serp_mismatch_flags,
                    ),
                })

            except Exception:
                logger.warning(
                    "Brief LLM generation failed",
                    extra={
                        "topic_name": topic.name,
                        "primary_keyword": primary_kw.keyword,
                    },
                )
                # Fallback: create basic brief structure
                fallback_warnings = self._collect_warnings(
                    collision_status,
                    do_not_target,
                    overlap_status,
                    serp_mismatch_flags,
                )
                fallback_warnings.append("LLM generation failed - manual completion required")
                fallback_publication_date = self._select_proposed_publication_date(
                    llm_date=None,
                    available_slots=available_slots,
                    today=today,
                    config=schedule_config,
                )
                output_briefs.append({
                    "topic_id": str(topic.id),
                    "topic_name": topic.name,
                    "primary_keyword": primary_kw.keyword,
                    "search_intent": resolved_intent,
                    "page_type": resolved_page_type,
                    "funnel_stage": topic.funnel_stage,
                    "serp_features": serp_features,
                    "competitors_content_types": competitors_content_types,
                    "serp_mismatch_flags": serp_mismatch_flags,
                    "proposed_url_slug": url_slug,
                    "url_collision_check": collision_status,
                    "do_not_target": do_not_target,
                    "overlap_status": overlap_status,
                    "working_titles": [f"[Title for: {primary_kw.keyword}]"],
                    "target_audience": "To be defined",
                    "reader_job_to_be_done": "To be defined",
                    "outline": [],
                    "supporting_keywords": [
                        kw.keyword for kw in keywords if kw.id != primary_kw.id
                    ],
                    "target_word_count": {"min": 1500, "max": 2500},
                    "proposed_publication_date": fallback_publication_date,
                    "has_warnings": True,
                    "warnings": fallback_warnings,
                })
                briefs_with_warnings += 1

        logger.info(
            "Brief generation complete",
            extra={
                "briefs_generated": len(output_briefs),
                "with_warnings": briefs_with_warnings,
                "skipped_ineligible_topics": skipped_ineligible,
                "skipped_existing_topics": skipped_existing,
            },
        )

        await self._update_progress(100, "Brief generation complete")

        return BriefOutput(
            briefs_generated=len(output_briefs),
            briefs_with_warnings=briefs_with_warnings,
            briefs=output_briefs,
        )

    def _is_topic_eligible(self, topic: Topic) -> bool:
        """Only generate briefs for primary/secondary fit tiers."""
        factors = topic.priority_factors or {}
        fit_tier = factors.get("fit_tier")
        if fit_tier is None:
            return topic.priority_rank is not None
        return fit_tier in {"primary", "secondary"}

    def _select_topics_for_briefs(
        self,
        *,
        topics: list[Topic],
        primary_keywords_by_topic_id: dict[str, Keyword | None],
        input_data: BriefInput,
    ) -> list[Topic]:
        """Select topics for briefs with optional zero-data exploratory reservation."""
        max_briefs = max(1, int(input_data.max_briefs))
        if len(topics) <= max_briefs:
            return topics

        if not input_data.include_zero_data_topics:
            return topics[:max_briefs]

        share = max(0.0, min(float(input_data.zero_data_topic_share), 0.5))
        reserved_slots = int(round(max_briefs * share))
        if share > 0 and reserved_slots <= 0:
            reserved_slots = 1
        if reserved_slots <= 0:
            return topics[:max_briefs]

        zero_data_candidates = [
            topic
            for topic in topics
            if self._is_zero_data_topic_candidate(
                topic=topic,
                primary_keyword=primary_keywords_by_topic_id.get(str(topic.id)),
                min_fit_score=input_data.zero_data_fit_score_min,
            )
        ]
        if not zero_data_candidates:
            return topics[:max_briefs]

        reserved_slots = min(reserved_slots, len(zero_data_candidates), max_briefs)
        selected_zero_data = sorted(
            zero_data_candidates,
            key=self._zero_data_sort_key,
        )[:reserved_slots]
        selected_zero_data_ids = {str(topic.id) for topic in selected_zero_data}

        core_slots = max(0, max_briefs - reserved_slots)
        selected_core = [
            topic for topic in topics if str(topic.id) not in selected_zero_data_ids
        ][:core_slots]
        selected_ids = {str(topic.id) for topic in selected_core}
        selected_ids.update(selected_zero_data_ids)

        selected: list[Topic] = []
        for topic in topics:
            if str(topic.id) in selected_ids:
                selected.append(topic)
            if len(selected) >= max_briefs:
                break

        return selected

    def _is_zero_data_topic_candidate(
        self,
        *,
        topic: Topic,
        primary_keyword: Keyword | None,
        min_fit_score: float,
    ) -> bool:
        """Return True if topic should be considered for zero-data exploratory briefs."""
        if primary_keyword is None:
            return False

        fit_score = self._topic_fit_score(topic)
        if fit_score is None or fit_score < min_fit_score:
            return False

        return self._is_keyword_zero_data(primary_keyword)

    def _is_keyword_zero_data(self, keyword: Keyword) -> bool:
        """Treat missing volume/metrics as zero-data demand signal."""
        if keyword.search_volume is None:
            return True
        if keyword.metrics_confidence is not None and keyword.metrics_confidence <= 0.2:
            return True
        return False

    def _zero_data_sort_key(self, topic: Topic) -> tuple[float, int]:
        """Sort high-fit zero-data topics by fit first, then existing priority rank."""
        fit_score = self._topic_fit_score(topic) or 0.0
        rank = topic.priority_rank if topic.priority_rank is not None else 999_999
        return (-fit_score, rank)

    def _topic_fit_score(self, topic: Topic) -> float | None:
        """Read fit_score from topic priority factors, if available."""
        factors = topic.priority_factors or {}
        fit_score = factors.get("fit_score")
        try:
            if fit_score is None:
                return None
            return float(fit_score)
        except (TypeError, ValueError):
            return None

    async def _load_primary_keywords_for_topics(
        self,
        topics: list[Topic],
    ) -> dict[str, Keyword | None]:
        """Load primary keyword models for topic selection logic."""
        topic_ids = [str(topic.id) for topic in topics]
        primary_ids = [
            topic.primary_keyword_id
            for topic in topics
            if topic.primary_keyword_id
        ]
        if not topic_ids:
            return {}
        mapping: dict[str, Keyword | None] = {topic_id: None for topic_id in topic_ids}
        if not primary_ids:
            return mapping

        result = await self.session.execute(
            select(Keyword).where(Keyword.id.in_(primary_ids))
        )
        keywords = list(result.scalars())
        by_id = {str(keyword.id): keyword for keyword in keywords}
        for topic in topics:
            if topic.primary_keyword_id:
                mapping[str(topic.id)] = by_id.get(str(topic.primary_keyword_id))

        return mapping

    async def _has_cannibalization_signals(self, project_id: str) -> bool:
        """Return True if topic rows contain cannibalization artifacts."""
        result = await self.session.execute(
            select(Topic.id)
            .where(
                Topic.project_id == project_id,
                or_(
                    Topic.cannibalization_risk.isnot(None),
                    Topic.overlapping_topic_ids.isnot(None),
                ),
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    def _resolve_brief_serp_profile(self, primary_kw: Keyword, topic: Topic) -> dict[str, Any]:
        """Resolve brief search profile, preferring Step 8 validation when available."""
        return {
            "search_intent": (
                primary_kw.validated_intent
                or topic.dominant_intent
                or "informational"
            ),
            "page_type": primary_kw.validated_page_type or topic.dominant_page_type or "guide",
            "serp_features": [
                str(item) for item in (primary_kw.serp_features or []) if item is not None
            ],
            "competitors_content_types": [
                str(item) for item in (primary_kw.format_requirements or []) if item is not None
            ],
            "serp_mismatch_flags": [
                str(item) for item in (primary_kw.serp_mismatch_flags or []) if item is not None
            ],
        }

    def _build_publication_schedule_config(
        self,
        input_data: BriefInput,
    ) -> PublicationScheduleConfig:
        """Normalize publication scheduling controls from step input."""
        posts_per_week = max(1, min(input_data.posts_per_week, 7))
        min_lead_days = max(1, min(input_data.min_lead_days, 60))
        llm_timing_flex_days = max(0, min(input_data.llm_timing_flex_days, 90))
        preferred = self._sanitize_weekdays(input_data.preferred_weekdays or [])
        weekdays = self._resolve_weekdays_for_cadence(
            posts_per_week=posts_per_week,
            preferred_weekdays=preferred,
        )
        return PublicationScheduleConfig(
            posts_per_week=posts_per_week,
            weekdays=weekdays,
            min_lead_days=min_lead_days,
            start_date=input_data.publication_start_date,
            use_llm_timing_hints=input_data.use_llm_timing_hints,
            llm_timing_flex_days=llm_timing_flex_days,
        )

    def _sanitize_weekdays(self, weekdays: list[int]) -> list[int]:
        """Return sorted unique weekday integers in [0..6]."""
        clean_values: set[int] = set()
        for day in weekdays:
            try:
                parsed = int(day)
            except (TypeError, ValueError):
                continue
            if 0 <= parsed <= 6:
                clean_values.add(parsed)
        sanitized = sorted(clean_values)
        return sanitized

    def _resolve_weekdays_for_cadence(
        self,
        posts_per_week: int,
        preferred_weekdays: list[int],
    ) -> list[int]:
        """Resolve effective publish weekdays, filling missing days for cadence."""
        default_order = [0, 2, 4, 1, 3, 5, 6]  # Mon, Wed, Fri, Tue, Thu, Sat, Sun
        chosen = list(preferred_weekdays)

        if not chosen:
            chosen = default_order[:posts_per_week]

        if len(chosen) < posts_per_week:
            for weekday in default_order:
                if weekday not in chosen:
                    chosen.append(weekday)
                if len(chosen) >= posts_per_week:
                    break

        return sorted(chosen[:posts_per_week])

    def _build_publication_slots(
        self,
        topic_count: int,
        today: date,
        config: PublicationScheduleConfig,
    ) -> list[date]:
        """Generate deterministic future publication slots based on cadence settings."""
        if topic_count <= 0:
            return []

        earliest = today + timedelta(days=config.min_lead_days)
        if config.start_date is not None and config.start_date > earliest:
            earliest = config.start_date

        # Start from Monday of the anchor week, then emit selected weekdays per week.
        week_start = earliest - timedelta(days=earliest.weekday())
        slots: list[date] = []
        while len(slots) < topic_count:
            for weekday in config.weekdays:
                candidate = week_start + timedelta(days=weekday)
                if candidate < earliest:
                    continue
                slots.append(candidate)
                if len(slots) >= topic_count:
                    break
            week_start += timedelta(days=7)

        return slots

    def _is_llm_date_eligible(
        self,
        llm_date: date,
        today: date,
        config: PublicationScheduleConfig,
    ) -> bool:
        min_allowed = today + timedelta(days=max(3, config.min_lead_days - 2))
        max_allowed = today + timedelta(days=180)
        return min_allowed <= llm_date <= max_allowed

    def _select_proposed_publication_date(
        self,
        llm_date: date | None,
        available_slots: list[date],
        today: date,
        config: PublicationScheduleConfig,
    ) -> date:
        """Assign a publish slot, using LLM timing hint when it aligns with schedule."""
        if not available_slots:
            fallback = today + timedelta(days=config.min_lead_days)
            return fallback

        chosen_index = 0

        if (
            config.use_llm_timing_hints
            and llm_date is not None
            and self._is_llm_date_eligible(llm_date=llm_date, today=today, config=config)
        ):
            nearest_index: int | None = None
            nearest_score: tuple[int, int, int] | None = None
            for idx, slot in enumerate(available_slots):
                delta_days = abs((slot - llm_date).days)
                if delta_days > config.llm_timing_flex_days:
                    continue
                # Prefer non-earlier slots, then closest distance, then earliest index.
                score = (0 if slot >= llm_date else 1, delta_days, idx)
                if nearest_score is None or score < nearest_score:
                    nearest_index = idx
                    nearest_score = score
            if nearest_index is not None:
                chosen_index = nearest_index

        return available_slots.pop(chosen_index if chosen_index < len(available_slots) else 0)

    async def _validate_output(self, result: BriefOutput, input_data: BriefInput) -> None:
        """Ensure output can be consumed by Step 13."""
        if result.briefs_generated > 0:
            return

        stmt = select(ContentBrief).where(ContentBrief.project_id == input_data.project_id)
        if input_data.topic_ids:
            stmt = stmt.where(
                ContentBrief.topic_id.in_(input_data.topic_ids)
            )
        existing = await self.session.execute(stmt.limit(1))
        if existing.scalars().first():
            return

        raise ValueError(
            "Step 12 generated 0 content briefs and no existing briefs "
            "were found for the requested scope."
        )

    def _build_brand_context(self, brand: BrandProfile | None) -> str:
        """Build brand context string for LLM."""
        if not brand:
            return ""

        parts = []
        if brand.company_name:
            parts.append(f"Company: {brand.company_name}")
        if brand.tagline:
            parts.append(f"Tagline: {brand.tagline}")

        # Full product details so the LLM knows what the brand actually does
        if brand.products_services:
            for p in brand.products_services[:5]:
                name = p.get("name", "")
                desc = p.get("description", "")
                benefits = p.get("core_benefits", [])
                audience = p.get("target_audience", "")
                product_parts = [f"  - Name: {name}"]
                if desc:
                    product_parts.append(f"    Description: {desc}")
                if benefits:
                    product_parts.append(f"    Core Benefits: {', '.join(benefits[:6])}")
                if audience:
                    product_parts.append(f"    Target Audience: {audience}")
                parts.append("Product:\n" + "\n".join(product_parts))

        if brand.unique_value_props:
            parts.append(f"Value Props: {', '.join(brand.unique_value_props[:5])}")
        if brand.differentiators:
            parts.append(f"Differentiators: {', '.join(brand.differentiators[:5])}")

        # ICP data so briefs target the right audience
        if brand.target_roles:
            parts.append(f"Target Roles: {', '.join(brand.target_roles[:5])}")
        if brand.target_industries:
            parts.append(f"Target Industries: {', '.join(brand.target_industries[:5])}")
        if brand.company_sizes:
            parts.append(f"Company Sizes: {', '.join(brand.company_sizes[:5])}")
        if brand.primary_pains:
            parts.append(f"Primary Pains: {', '.join(brand.primary_pains[:5])}")

        # Topic boundaries
        if brand.in_scope_topics:
            parts.append(f"In-Scope Topics: {', '.join(brand.in_scope_topics[:10])}")
        if brand.out_of_scope_topics:
            parts.append(f"Out-of-Scope Topics: {', '.join(brand.out_of_scope_topics[:10])}")

        # Claim guardrails
        if brand.restricted_claims:
            parts.append(
                "Restricted Claims (DO NOT make these): "
                f"{', '.join(brand.restricted_claims[:5])}"
            )

        if brand.tone_attributes:
            parts.append(f"Tone: {', '.join(brand.tone_attributes[:5])}")

        return "\n".join(parts)

    def _extract_money_pages(self, brand: BrandProfile | None) -> list[str]:
        """Extract money page URLs from brand profile."""
        if not brand or not brand.money_pages:
            return []
        return [mp.get("url", "") for mp in brand.money_pages if mp.get("url")]

    def _generate_url_slug(self, keyword: str, page_type: str) -> str:
        """Generate URL slug from keyword and page type.

        Examples:
        - "how to choose crm" + "guide" -> "/blog/how-to-choose-crm"
        - "best crm software" + "list" -> "/blog/best-crm-software"
        - "crm pricing" + "landing" -> "/crm-pricing"
        """
        # Normalize keyword to slug format
        slug = keyword.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars
        slug = re.sub(r'[\s_]+', '-', slug)  # Replace spaces with hyphens
        slug = re.sub(r'-+', '-', slug)  # Collapse multiple hyphens
        slug = slug.strip('-')

        # Add path prefix based on page type
        if page_type in ("landing",):
            return f"/{slug}"
        elif page_type in ("tool", "calculator"):
            return f"/tools/{slug}"
        elif page_type in ("comparison", "alternatives"):
            return f"/compare/{slug}"
        else:
            return f"/blog/{slug}"

    def _check_url_collision(
        self,
        url_slug: str,
        existing_urls: set[str],
    ) -> str:
        """Check if URL slug collides with existing content.

        Returns:
        - "safe": No collision detected
        - "warning": Similar URL exists
        - "conflict": Exact URL already exists
        """
        if not existing_urls:
            return "safe"

        # Normalize for comparison
        normalized_slug = url_slug.lower().strip('/')

        for existing in existing_urls:
            existing_normalized = existing.lower().strip('/')

            # Exact match
            if normalized_slug == existing_normalized:
                return "conflict"

            # Partial match (similar topic)
            slug_words = set(normalized_slug.split('-'))
            existing_words = set(existing_normalized.split('-'))
            overlap = len(slug_words & existing_words) / max(len(slug_words), 1)
            if overlap > 0.7:
                return "warning"

        return "safe"

    async def _load_existing_urls(self, project_id: str) -> set[str]:
        """Load existing content URLs from inventory (if Step 9 ran)."""
        # TODO: Load from ContentInventory model when implemented
        # For now, return empty set
        return set()

    async def _load_existing_brief_topic_ids(self, topic_ids: list[str]) -> set[str]:
        """Load topic IDs that already have persisted briefs."""
        if not topic_ids:
            return set()
        result = await self.session.execute(
            select(ContentBrief.topic_id).where(
                ContentBrief.project_id == self.project_id,
                ContentBrief.topic_id.in_(topic_ids),
            )
        )
        return {str(topic_id) for topic_id in result.scalars().all() if topic_id is not None}

    async def _build_do_not_target(
        self,
        topic: Topic,
        keywords: list[Keyword],
        project_id: str,
    ) -> list[str]:
        """Build list of keywords to NOT target (cannibalization prevention).

        Returns keywords that:
        - Already have content ranking
        - Belong to another topic's primary set
        - Have conflicting intent
        """
        do_not_target = []

        # If topic has overlapping_topic_ids from Step 10, get their primary keywords
        if topic.overlapping_topic_ids:
            for other_topic_id in topic.overlapping_topic_ids:
                try:
                    other_result = await self.session.execute(
                        select(Topic).where(Topic.id == other_topic_id)
                    )
                    other_topic = other_result.scalar_one_or_none()
                    if other_topic and other_topic.primary_keyword_id:
                        kw_result = await self.session.execute(
                            select(Keyword).where(Keyword.id == other_topic.primary_keyword_id)
                        )
                        kw = kw_result.scalar_one_or_none()
                        if kw:
                            do_not_target.append(kw.keyword)
                except Exception:
                    pass

        return do_not_target

    def _get_conflicting_intent(self, topic: Topic) -> str | None:
        """Get intent that should NOT be covered by this topic.

        E.g., if topic is informational, don't cover transactional intent.
        """
        intent = topic.dominant_intent or ""

        if intent == "informational":
            return "transactional"
        elif intent == "transactional":
            return "informational"
        elif intent == "commercial":
            return None  # Commercial can blend

        return None

    def _build_keyword_section_map(
        self,
        outline: list,
        keywords: list[str],
    ) -> dict[str, list[str]]:
        """Map keywords to outline sections."""
        section_map = {}

        # Simple heuristic: distribute keywords across H2 sections
        h2_sections = [s for s in outline if s.level == 2]

        if not h2_sections:
            return {"intro": keywords[:5], "body": keywords[5:10]}

        keywords_per_section = max(1, len(keywords) // max(len(h2_sections), 1))

        for i, section in enumerate(h2_sections):
            start = i * keywords_per_section
            end = start + keywords_per_section
            section_keywords = keywords[start:end]
            if section_keywords:
                section_map[section.heading] = section_keywords

        return section_map

    def _build_internal_links(
        self,
        topic: Topic,
        keywords: list[Keyword],
    ) -> list[dict[str, str]]:
        """Build internal link recommendations.

        Note: This now only creates basic pillar links. Step 2 (interlinking_enrichment)
        will add semantic cross-links and sitemap-based links.
        """
        links = []

        # Link to pillar page
        if topic.pillar_seed_topic_id:
            links.append({
                "target_type": "pillar_page",
                "target_url": None,
                "target_brief_id": None,
                "anchor_text": topic.name,
                "placement_section": "In introduction",
                "relevance_score": 1.0,
                "intent_alignment": "pillar_link",
                "funnel_relationship": "pillar_link",
            })

        # Generic "related content" placeholder removed - Step 2 will add semantic links

        return links

    def _build_money_page_links(
        self,
        topic: Topic,
        money_pages: list[str],
    ) -> list[dict[str, str]]:
        """Build money page link recommendations."""
        links = []

        target_pages = topic.target_money_pages or money_pages[:2]

        for page in target_pages:
            links.append({
                "url": page,
                "anchor_suggestion": "Learn more",
                "placement": "In relevant section or CTA",
            })

        return links

    def _collect_warnings(
        self,
        collision_status: str,
        do_not_target: list[str],
        overlap_status: str,
        serp_mismatch_flags: list[str] | None = None,
    ) -> list[str]:
        """Collect all warnings for this brief."""
        warnings = []

        if collision_status == "conflict":
            warnings.append("URL already exists - consider updating existing page instead")
        elif collision_status == "warning":
            warnings.append("Similar URL exists - check for potential overlap")

        if do_not_target:
            warnings.append(f"Avoid targeting: {', '.join(do_not_target[:3])}")

        if overlap_status == "unknown":
            warnings.append("Cannibalization not checked - run Step 10 for full analysis")

        for flag in serp_mismatch_flags or []:
            if flag == "intent_mismatch":
                warnings.append("SERP intent differs from Step 5 intent - review targeting")
            elif flag == "page_type_mismatch":
                warnings.append("SERP page type differs from Step 5 recommendation")
            elif flag == "serp_fetch_failed":
                warnings.append("SERP validation fetch failed for this keyword")
            elif flag == "no_organic_results":
                warnings.append("SERP returned no organic results for this keyword")
            else:
                warnings.append(f"SERP validation flag: {flag}")

        return warnings

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        coerced = str(value).strip()
        return coerced or None

    def _str_list_or_none(self, value: Any) -> list[str] | None:
        if not isinstance(value, list):
            return None
        cleaned = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
        return cleaned or None

    def _dict_list_or_none(self, value: Any) -> list[dict] | None:
        if not isinstance(value, list):
            return None
        cleaned = [item for item in value if isinstance(item, dict)]
        return cleaned or None

    def _dict_or_none(self, value: Any) -> dict | None:
        if isinstance(value, dict):
            return value
        return None

    def _optional_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _optional_date(self, value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    async def _persist_results(self, result: BriefOutput) -> None:
        """Save content briefs to database."""
        await self._lock_project_publication_schedule()

        topic_ids = [
            brief_data["topic_id"]
            for brief_data in result.briefs
            if brief_data.get("topic_id")
        ]

        existing_by_topic_id: dict[str, ContentBrief] = {}
        if topic_ids:
            existing_result = await self.session.execute(
                select(ContentBrief).where(
                    ContentBrief.project_id == self.project_id,
                    ContentBrief.topic_id.in_(topic_ids),
                )
            )
            existing_by_topic_id = {
                str(brief.topic_id): brief
                for brief in existing_result.scalars().all()
                if brief.topic_id is not None
            }

        reserved_dates_result = await self.session.execute(
            select(ContentBrief.proposed_publication_date).where(
                ContentBrief.project_id == self.project_id,
                ContentBrief.proposed_publication_date.isnot(None),
            )
        )
        reserved_publication_dates = Counter(
            publication_date
            for publication_date in reserved_dates_result.scalars().all()
            if publication_date is not None
        )

        for brief_data in result.briefs:
            topic_id = self._optional_str(brief_data.get("topic_id"))
            if topic_id is None:
                continue

            existing = existing_by_topic_id.get(topic_id)
            existing_publication_date = (
                existing.proposed_publication_date
                if existing is not None
                else None
            )
            if existing_publication_date is not None:
                self._decrement_reserved_date_count(
                    reserved_publication_dates,
                    existing_publication_date,
                )
            target_word_count_raw = brief_data.get("target_word_count")
            target_word_count = (
                target_word_count_raw
                if isinstance(target_word_count_raw, dict)
                else {}
            )
            primary_keyword = str(brief_data.get("primary_keyword") or "")
            search_intent = self._optional_str(brief_data.get("search_intent"))
            page_type = self._optional_str(brief_data.get("page_type"))
            funnel_stage = self._optional_str(brief_data.get("funnel_stage"))
            working_titles = self._str_list_or_none(brief_data.get("working_titles"))
            target_audience = self._optional_str(brief_data.get("target_audience"))
            reader_job_to_be_done = self._optional_str(brief_data.get("reader_job_to_be_done"))
            outline = self._dict_list_or_none(brief_data.get("outline"))
            supporting_keywords = self._str_list_or_none(brief_data.get("supporting_keywords"))
            supporting_keywords_map = self._dict_or_none(brief_data.get("supporting_keywords_map"))
            examples_required = self._str_list_or_none(brief_data.get("examples_required"))
            faq_questions = self._str_list_or_none(brief_data.get("faq_questions"))
            recommended_schema_type = self._optional_str(brief_data.get("recommended_schema_type"))
            internal_links_out = self._dict_list_or_none(brief_data.get("internal_links_out"))
            money_page_links = self._dict_list_or_none(brief_data.get("money_page_links"))
            meta_title_guidelines = self._optional_str(brief_data.get("meta_title_guidelines"))
            meta_description_guidelines = self._optional_str(
                brief_data.get("meta_description_guidelines")
            )
            target_word_count_min = self._optional_int(target_word_count.get("min"))
            target_word_count_max = self._optional_int(target_word_count.get("max"))
            must_include_sections = self._str_list_or_none(brief_data.get("must_include_sections"))
            proposed_publication_date = self._optional_date(
                brief_data.get("proposed_publication_date")
            )
            assigned_publication_date = self._resolve_unique_publication_date(
                desired_date=proposed_publication_date,
                existing_date=existing_publication_date,
                reserved_date_counts=reserved_publication_dates,
            )
            brief_data["proposed_publication_date"] = assigned_publication_date
            if assigned_publication_date is not None:
                reserved_publication_dates[assigned_publication_date] += 1
            payload = {
                "primary_keyword": primary_keyword,
                "search_intent": search_intent,
                "page_type": page_type,
                "funnel_stage": funnel_stage,
                "working_titles": working_titles,
                "target_audience": target_audience,
                "reader_job_to_be_done": reader_job_to_be_done,
                "outline": outline,
                "supporting_keywords": supporting_keywords,
                "supporting_keywords_map": supporting_keywords_map,
                "examples_required": examples_required,
                "faq_questions": faq_questions,
                "recommended_schema_type": recommended_schema_type,
                "internal_links_out": internal_links_out,
                "money_page_links": money_page_links,
                "meta_title_guidelines": meta_title_guidelines,
                "meta_description_guidelines": meta_description_guidelines,
                "target_word_count_min": target_word_count_min,
                "target_word_count_max": target_word_count_max,
                "must_include_sections": must_include_sections,
                "proposed_publication_date": assigned_publication_date,
                "status": "draft",
            }
            if existing:
                existing.patch(
                    self.session,
                    ContentBriefPatchDTO.from_partial(payload),
                )
                continue

            create_dto = ContentBriefCreateDTO(
                project_id=self.project_id,
                topic_id=topic_id,
                primary_keyword=primary_keyword,
                search_intent=search_intent,
                page_type=page_type,
                funnel_stage=funnel_stage,
                working_titles=working_titles,
                target_audience=target_audience,
                reader_job_to_be_done=reader_job_to_be_done,
                outline=outline,
                supporting_keywords=supporting_keywords,
                supporting_keywords_map=supporting_keywords_map,
                examples_required=examples_required,
                faq_questions=faq_questions,
                recommended_schema_type=recommended_schema_type,
                internal_links_out=internal_links_out,
                money_page_links=money_page_links,
                meta_title_guidelines=meta_title_guidelines,
                meta_description_guidelines=meta_description_guidelines,
                target_word_count_min=target_word_count_min,
                target_word_count_max=target_word_count_max,
                must_include_sections=must_include_sections,
                proposed_publication_date=assigned_publication_date,
                status="draft",
            )

            ContentBrief.create(
                self.session,
                create_dto,
            )

        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, 12)

        # Set result summary
        self.set_result_summary({
            "briefs_generated": result.briefs_generated,
            "briefs_with_warnings": result.briefs_with_warnings,
            "url_conflicts": sum(
                1 for b in result.briefs if b.get("url_collision_check") == "conflict"
            ),
            "url_warnings": sum(
                1 for b in result.briefs if b.get("url_collision_check") == "warning"
            ),
            "unchecked_overlap": sum(
                1 for b in result.briefs if b.get("overlap_status") == "unknown"
            ),
        })

        await self.session.commit()

    async def _lock_project_publication_schedule(self) -> None:
        """Serialize publication-date allocation per project."""
        lock_result = await self.session.execute(
            select(Project.id).where(Project.id == self.project_id).with_for_update()
        )
        if lock_result.scalar_one_or_none() is None:
            raise ValueError(f"Project not found: {self.project_id}")

    def _decrement_reserved_date_count(
        self,
        reserved_date_counts: Counter[date],
        assigned_date: date,
    ) -> None:
        current_count = reserved_date_counts.get(assigned_date, 0)
        if current_count <= 1:
            reserved_date_counts.pop(assigned_date, None)
            return
        reserved_date_counts[assigned_date] = current_count - 1

    def _resolve_unique_publication_date(
        self,
        *,
        desired_date: date | None,
        existing_date: date | None,
        reserved_date_counts: Counter[date],
    ) -> date | None:
        candidate = desired_date or existing_date
        if candidate is None:
            return None
        return self._next_available_publication_date(
            candidate,
            reserved_date_counts=reserved_date_counts,
        )

    def _next_available_publication_date(
        self,
        candidate: date,
        *,
        reserved_date_counts: Counter[date],
    ) -> date:
        max_shift_days = 365 * 3
        current = candidate
        for _ in range(max_shift_days):
            if reserved_date_counts.get(current, 0) == 0:
                return current
            current += timedelta(days=1)
        return current
