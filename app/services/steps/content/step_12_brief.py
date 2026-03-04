"""Step 12: Content Brief Generation.

Generates writer-ready briefs for prioritized topics.
Includes URL slug generation and cannibalization guardrails.
"""

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import or_, select

from app.agents.brief_diversifier import BriefDiversifierAgent, BriefDiversifierInput
from app.agents.brief_generator import BriefGeneratorAgent, BriefGeneratorInput
from app.agents.serp_briefing import (
    SerpBriefingAgent,
    SerpBriefingInput,
    SerpPageSnapshot,
)
from app.integrations.scraper import WebsiteScraper
from app.models.brand import BrandProfile
from app.models.content import ContentBrief
from app.models.content_pillar import ContentBriefPillarAssignment, ContentPillar
from app.models.generated_dtos import (
    ContentBriefCreateDTO,
    ContentBriefPatchDTO,
    ContentBriefPillarAssignmentCreateDTO,
    ContentPillarCreateDTO,
    ContentPillarPatchDTO,
)
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.content_keyword_tracking import sync_brief_keywords
from app.services.discovery.topic_overlap import (
    build_comparison_key,
    build_family_key,
    compute_topic_overlap,
    is_exact_pair_duplicate,
    is_sibling_pair,
    jaccard,
    normalize_text_tokens,
)
from app.services.run_strategy import (
    RunStrategy,
    build_adaptive_target_mix,
    funnel_from_intent,
    normalize_funnel_label,
    normalize_intent_label,
)
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)
EXISTING_CONTENT_SKIP_THRESHOLD = 0.82
IN_BATCH_DIVERSIFICATION_OVERLAP_THRESHOLD = 0.82
BRIEF_DIVERSIFIER_RETRY_ATTEMPTS = 2
SERP_SNAPSHOT_TOP_RESULTS = 8
SERP_SNAPSHOT_MAX_FETCHED_BLOG_PAGES = 3
SERP_SNAPSHOT_MAX_HEADINGS_PER_PAGE = 14
SERP_SNAPSHOT_FETCH_TIMEOUT_SECONDS = 12.0
SERP_NON_CONTENT_GUIDANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(author|byline|author bio|writer bio|bio box|author profile|avatar)\b"),
    re.compile(r"\b(published date|publish date|updated date|last updated|read time)\b"),
    re.compile(r"\b(meta title|meta description|title tag|open graph|schema(?:\.org)?|json-ld)\b"),
    re.compile(r"\b(slug|permalink|url structure|canonical)\b"),
    re.compile(r"\b(hero image|featured image|cover image|banner image)\b"),
    re.compile(r"\b(nav(?:igation)?|sidebar|footer|site layout|page layout|font|color palette)\b"),
    re.compile(r"\b(call to action|cta button|subscribe|newsletter|popup|pop-up)\b"),
    re.compile(r"\b(internal linking strategy|anchor text strategy|schema markup)\b"),
)
ALLOWED_PILLAR_CONFIG: dict[str, tuple[str, str]] = {
    "blog": (
        "Blog",
        "General informational and educational content for broad awareness topics.",
    ),
    "tools": (
        "Tools",
        "Software, templates, calculators, comparisons, alternatives, and product-focused content.",
    ),
    "guides": (
        "Guides",
        "How-to, implementation walkthroughs, use cases, and educational playbooks.",
    ),
}


@dataclass
class BriefInput:
    """Input for Step 12."""

    project_id: str
    topic_ids: list[str] | None = None  # Specific topics, or None for all prioritized
    max_briefs: int = 20  # Limit number of briefs to generate
    posts_per_week: int = 1
    preferred_weekdays: list[int] | None = None  # 0=Mon ... 6=Sun
    min_lead_days: int = 3
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
    skipped_existing_pair: int = 0
    skipped_existing_overlap: int = 0
    sibling_pairs_allowed: int = 0
    skipped_batch_diversification: int = 0


@dataclass(frozen=True)
class ExistingBriefSignature:
    """Existing-brief signature used for semantic duplicate suppression."""

    comparison_key: str | None
    family_key: str
    intent: str
    page_type: str
    keyword_tokens: set[str]
    text_tokens: set[str]


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

    step_number = 1
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
            raise ValueError("No prioritized topics found. Run discovery step 6 first.")

    async def _execute(self, input_data: BriefInput) -> BriefOutput:
        """Execute content brief generation."""
        # Load brand + run strategy
        strategy = await self.get_run_strategy()

        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == input_data.project_id)
        )
        brand = brand_result.scalar_one_or_none()
        project_result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = project_result.scalar_one()
        project_posts_per_week = self._coerce_posts_per_week(project.posts_per_week)

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
        mix_diagnostics: dict[str, Any] = {}
        ordered_topics_all = eligible_topics_all

        selected_topics = eligible_topics_all
        zero_data_candidates = 0
        zero_data_selected = 0
        primary_keywords_by_topic_id: dict[str, Keyword | None] = {}
        if input_data.topic_ids:
            selected_topics = eligible_topics_all[:input_data.max_briefs]
        else:
            ordered_topics_all, mix_diagnostics = self._soft_balance_topic_order(
                topics=eligible_topics_all,
                strategy=strategy,
            )
            primary_keywords_by_topic_id = await self._load_primary_keywords_for_topics(
                ordered_topics_all
            )
            zero_data_candidates = sum(
                1
                for topic in ordered_topics_all
                if self._is_zero_data_topic_candidate(
                    topic=topic,
                    primary_keyword=primary_keywords_by_topic_id.get(str(topic.id)),
                    min_fit_score=input_data.zero_data_fit_score_min,
                )
            )
            selected_topics = self._select_topics_for_briefs(
                topics=ordered_topics_all,
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
        if not primary_keywords_by_topic_id:
            primary_keywords_by_topic_id = await self._load_primary_keywords_for_topics(
                selected_topics
            )
        if mix_diagnostics:
            logger.info(
                "Step 12 soft intent/funnel balancing applied",
                extra={
                    "project_id": input_data.project_id,
                    "selected_topics": len(selected_topics),
                    "target_intent_mix": mix_diagnostics.get("target_intent_mix"),
                    "observed_intent_mix": mix_diagnostics.get("observed_intent_mix"),
                    "target_funnel_mix": mix_diagnostics.get("target_funnel_mix"),
                    "observed_funnel_mix": mix_diagnostics.get("observed_funnel_mix"),
                },
            )

        existing_brief_topic_ids = await self._load_existing_brief_topic_ids(
            [str(topic.id) for topic in selected_topics]
        )
        existing_signatures = await self._load_existing_brief_signatures(input_data.project_id)
        existing_brief_history = await self._load_existing_brief_history(input_data.project_id)
        skipped_existing_pair = 0
        skipped_existing_overlap = 0
        sibling_pairs_allowed = 0
        skipped_batch_diversification = 0
        diversification_notes = ""
        topics_to_generate: list[Topic] = []
        for topic in selected_topics:
            topic_id = str(topic.id)
            if topic_id in existing_brief_topic_ids:
                continue
            primary_keyword = primary_keywords_by_topic_id.get(topic_id)
            should_skip, reason_code, sibling_allowed = self._should_skip_as_covered(
                topic=topic,
                primary_keyword=primary_keyword,
                existing_signatures=existing_signatures,
            )
            if sibling_allowed:
                sibling_pairs_allowed += 1
            if should_skip:
                if reason_code == "existing_pair_covered":
                    skipped_existing_pair += 1
                elif reason_code == "existing_content_overlap":
                    skipped_existing_overlap += 1
                continue
            topics_to_generate.append(topic)

        topics_to_generate, skipped_batch_diversification, diversification_notes = (
            await self._apply_batch_diversification_selection(
                topics=topics_to_generate,
                primary_keywords_by_topic_id=primary_keywords_by_topic_id,
                existing_history=existing_brief_history,
                target_count=input_data.max_briefs,
            )
        )
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
                "skipped_existing_pair": skipped_existing_pair,
                "skipped_existing_overlap": skipped_existing_overlap,
                "sibling_pairs_allowed": sibling_pairs_allowed,
                "skipped_batch_diversification": skipped_batch_diversification,
                "diversification_notes": diversification_notes or None,
            },
        )

        if not topics_to_generate:
            return BriefOutput(
                briefs_generated=0,
                briefs_with_warnings=0,
                skipped_existing_pair=skipped_existing_pair,
                skipped_existing_overlap=skipped_existing_overlap,
                sibling_pairs_allowed=sibling_pairs_allowed,
                skipped_batch_diversification=skipped_batch_diversification,
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
        serp_briefing_agent = SerpBriefingAgent()
        output_briefs = []
        briefs_with_warnings = 0
        today = date.today()
        schedule_config = self._build_publication_schedule_config(
            input_data=input_data,
            project_posts_per_week=project_posts_per_week,
        )
        reserved_dates_result = await self.session.execute(
            select(ContentBrief.topic_id, ContentBrief.proposed_publication_date).where(
                ContentBrief.project_id == input_data.project_id,
                ContentBrief.proposed_publication_date.isnot(None),
            )
        )
        reserved_rows = list(reserved_dates_result.all())
        reserved_publication_dates = Counter(
            publication_date
            for _, publication_date in reserved_rows
            if publication_date is not None
        )
        topic_ids_to_refresh = {
            str(topic.id)
            for topic in topics_to_generate
            if getattr(topic, "id", None) is not None
        }
        for topic_id, publication_date in reserved_rows:
            if publication_date is None:
                continue
            if str(topic_id) not in topic_ids_to_refresh:
                continue
            self._decrement_reserved_date_count(
                reserved_publication_dates,
                publication_date,
            )
        available_slots = self._build_publication_slots(
            topic_count=len(topics_to_generate),
            today=today,
            config=schedule_config,
            reserved_date_counts=reserved_publication_dates,
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

            supporting_keyword_models = [kw for kw in keywords if kw.id != primary_kw.id]
            supporting_keyword_texts = [kw.keyword for kw in supporting_keyword_models]
            supporting_keyword_ids = [
                str(kw.id)
                for kw in supporting_keyword_models
            ]

            serp_profile = self._resolve_brief_serp_profile(primary_kw, topic)
            resolved_intent = serp_profile["search_intent"]
            resolved_page_type = serp_profile["page_type"]
            serp_features = serp_profile["serp_features"]
            competitors_content_types = serp_profile["competitors_content_types"]
            serp_mismatch_flags = serp_profile["serp_mismatch_flags"]
            serp_briefing = await self._build_serp_briefing(
                primary_keyword=primary_kw.keyword,
                search_intent=resolved_intent,
                page_type=resolved_page_type,
                serp_features=serp_features,
                serp_top_results=primary_kw.serp_top_results,
                serp_briefing_agent=serp_briefing_agent,
            )
            top_ranking_pages = serp_briefing["top_pages"]
            serp_best_practices = serp_briefing["best_practices"]
            serp_recommended_sections = serp_briefing["recommended_sections"]
            serp_outperform_opportunities = serp_briefing["opportunities_to_outperform"]
            serp_analysis_summary = serp_briefing["summary"]

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
                    supporting_keywords=supporting_keyword_texts[:15],
                    search_intent=resolved_intent,
                    page_type=resolved_page_type,
                    funnel_stage=topic.funnel_stage or "tofu",
                    brand_context=brand_context,
                    competitors_content_types=competitors_content_types,
                    serp_features=serp_features,
                    top_ranking_pages=top_ranking_pages,
                    serp_best_practices=serp_best_practices,
                    serp_recommended_sections=serp_recommended_sections,
                    serp_outperform_opportunities=serp_outperform_opportunities,
                    serp_analysis_summary=serp_analysis_summary,
                    money_pages=topic.target_money_pages or money_pages[:3],
                    conversion_intents=strategy.conversion_intents,
                    recommended_publish_order=topic.recommended_publish_order,
                )

                output = await agent.run(agent_input)
                brief_data = output.brief
                merged_must_include_sections = self._merge_unique_str_items(
                    brief_data.must_include_sections,
                    serp_recommended_sections,
                    limit=12,
                )
                proposed_publication_date = self._select_proposed_publication_date(
                    llm_date=brief_data.proposed_publication_date,
                    available_slots=available_slots,
                    today=today,
                    config=schedule_config,
                )
                pillar_slug = self._resolve_allowed_pillar_slug(brief_data.pillar_slug)

                output_briefs.append({
                    "topic_id": str(topic.id),
                    "topic_name": topic.name,
                    "primary_keyword": primary_kw.keyword,
                    "primary_keyword_id": str(primary_kw.id),
                    "search_intent": resolved_intent,
                    "page_type": resolved_page_type,
                    "funnel_stage": topic.funnel_stage,
                    "serp_features": serp_features,
                    "competitors_content_types": competitors_content_types,
                    "serp_mismatch_flags": serp_mismatch_flags,
                    "serp_analysis_summary": serp_analysis_summary,
                    "serp_best_practices": serp_best_practices,
                    "serp_recommended_sections": serp_recommended_sections,
                    "serp_outperform_opportunities": serp_outperform_opportunities,
                    "top_ranking_pages_snapshot": top_ranking_pages,
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
                    "supporting_keywords": supporting_keyword_texts,
                    "supporting_keyword_ids": supporting_keyword_ids,
                    "supporting_keywords_map": self._build_keyword_section_map(
                        brief_data.outline,
                        supporting_keyword_texts,
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
                    "must_include_sections": merged_must_include_sections,
                    "recommended_schema_type": brief_data.recommended_schema_type,
                    "proposed_publication_date": proposed_publication_date,
                    "pillar_slug": pillar_slug,
                    "pillar_confidence": 1.0,
                    "pillar_assignment_method": "ai_brief",
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
                    "primary_keyword_id": str(primary_kw.id),
                    "search_intent": resolved_intent,
                    "page_type": resolved_page_type,
                    "funnel_stage": topic.funnel_stage,
                    "serp_features": serp_features,
                    "competitors_content_types": competitors_content_types,
                    "serp_mismatch_flags": serp_mismatch_flags,
                    "serp_analysis_summary": serp_analysis_summary,
                    "serp_best_practices": serp_best_practices,
                    "serp_recommended_sections": serp_recommended_sections,
                    "serp_outperform_opportunities": serp_outperform_opportunities,
                    "top_ranking_pages_snapshot": top_ranking_pages,
                    "proposed_url_slug": url_slug,
                    "url_collision_check": collision_status,
                    "do_not_target": do_not_target,
                    "overlap_status": overlap_status,
                    "working_titles": [f"[Title for: {primary_kw.keyword}]"],
                    "target_audience": "To be defined",
                    "reader_job_to_be_done": "To be defined",
                    "outline": [],
                    "supporting_keywords": supporting_keyword_texts,
                    "supporting_keyword_ids": supporting_keyword_ids,
                    "target_word_count": {"min": 1500, "max": 2500},
                    "must_include_sections": serp_recommended_sections[:8],
                    "proposed_publication_date": fallback_publication_date,
                    "pillar_slug": "blog",
                    "pillar_confidence": 0.0,
                    "pillar_assignment_method": "fallback_default",
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
                "skipped_existing_pair": skipped_existing_pair,
                "skipped_existing_overlap": skipped_existing_overlap,
                "sibling_pairs_allowed": sibling_pairs_allowed,
                "skipped_batch_diversification": skipped_batch_diversification,
            },
        )

        await self._update_progress(100, "Brief generation complete")

        return BriefOutput(
            briefs_generated=len(output_briefs),
            briefs_with_warnings=briefs_with_warnings,
            briefs=output_briefs,
            skipped_existing_pair=skipped_existing_pair,
            skipped_existing_overlap=skipped_existing_overlap,
            sibling_pairs_allowed=sibling_pairs_allowed,
            skipped_batch_diversification=skipped_batch_diversification,
        )

    def _is_topic_eligible(self, topic: Topic) -> bool:
        """Only generate briefs for primary/secondary fit tiers."""
        fit_tier = topic.fit_tier
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
        """Read typed fit_score value."""
        fit_score = topic.fit_score
        try:
            if fit_score is None:
                return None
            return float(fit_score)
        except (TypeError, ValueError):
            return None

    def _soft_balance_topic_order(
        self,
        *,
        topics: list[Topic],
        strategy: RunStrategy,
    ) -> tuple[list[Topic], dict[str, Any]]:
        """Softly rerank topics to improve intent/funnel distribution without hard quotas."""
        if len(topics) <= 1:
            return topics, {}

        observed_intent_mix = self._observed_intent_mix(topics)
        base_intent_mix = strategy.intent_mix.to_shares()
        target_intent_mix = build_adaptive_target_mix(
            base_mix=base_intent_mix,
            observed_mix=observed_intent_mix,
            influence=strategy.intent_mix.influence,
        )

        observed_funnel_mix = self._observed_funnel_mix(topics)
        configured_funnel_mix = strategy.funnel_mix.to_shares()
        derived_funnel_mix = {
            "tofu": target_intent_mix["informational"],
            "mofu": target_intent_mix["commercial"],
            "bofu": target_intent_mix["transactional"],
        }
        base_funnel_mix = {
            key: round((derived_funnel_mix[key] * 0.7) + (configured_funnel_mix[key] * 0.3), 6)
            for key in ("tofu", "mofu", "bofu")
        }
        target_funnel_mix = build_adaptive_target_mix(
            base_mix=base_funnel_mix,
            observed_mix=observed_funnel_mix,
            influence=strategy.funnel_mix.influence,
        )

        scored: list[tuple[float, int, Topic]] = []
        for topic in topics:
            rank = topic.priority_rank if topic.priority_rank is not None else 999_999
            quality_score = float(topic.priority_score or topic.final_priority_score or 0.0)
            if quality_score <= 0 and rank < 999_999:
                quality_score = max(0.0, 100.0 - float(rank))

            intent_label = str(topic.dominant_intent or "").strip().lower()
            funnel_label = str(topic.funnel_stage or "").strip().lower()
            intent_key = normalize_intent_label(intent_label)
            funnel_key = normalize_funnel_label(funnel_label) or funnel_from_intent(intent_label)

            intent_bonus = 0.0
            if intent_key is not None:
                gap = float(target_intent_mix[intent_key]) - float(observed_intent_mix[intent_key])
                intent_bonus = max(-1.0, min(1.0, gap)) * strategy.intent_mix.influence

            funnel_bonus = 0.0
            if funnel_key is not None:
                gap = float(target_funnel_mix[funnel_key]) - float(observed_funnel_mix[funnel_key])
                funnel_bonus = max(-1.0, min(1.0, gap)) * strategy.funnel_mix.influence

            nav_penalty = 0.0
            if intent_label == "navigational":
                fit_score = self._topic_fit_score(topic) or 0.0
                if fit_score < 0.82:
                    nav_penalty = 0.50 * strategy.intent_mix.influence

            soft_bonus = (intent_bonus * 3.0) + (funnel_bonus * 2.5) - nav_penalty
            sort_score = quality_score + soft_bonus
            scored.append((sort_score, -rank, topic))

        scored.sort(key=lambda item: (-item[0], -item[1]))
        ordered_topics = [item[2] for item in scored]
        diagnostics = {
            "base_intent_mix": {
                key: round(float(value), 4)
                for key, value in base_intent_mix.items()
            },
            "observed_intent_mix": {
                key: round(float(value), 4)
                for key, value in observed_intent_mix.items()
            },
            "target_intent_mix": {
                key: round(float(value), 4)
                for key, value in target_intent_mix.items()
            },
            "base_funnel_mix": {
                key: round(float(value), 4)
                for key, value in base_funnel_mix.items()
            },
            "observed_funnel_mix": {
                key: round(float(value), 4)
                for key, value in observed_funnel_mix.items()
            },
            "target_funnel_mix": {
                key: round(float(value), 4)
                for key, value in target_funnel_mix.items()
            },
        }
        return ordered_topics, diagnostics

    def _observed_intent_mix(self, topics: list[Topic]) -> dict[str, float]:
        weighted: dict[str, float] = {
            "informational": 0.0,
            "commercial": 0.0,
            "transactional": 0.0,
        }
        total = 0.0
        for topic in topics:
            intent_key = normalize_intent_label(topic.dominant_intent)
            if intent_key is None:
                continue
            fit_score = self._topic_fit_score(topic) or 0.0
            quality_score = float(topic.priority_score or topic.final_priority_score or 0.0)
            weight = max(0.05, (fit_score * 0.7) + ((quality_score / 100.0) * 0.3))
            weighted[intent_key] += weight
            total += weight
        if total <= 0:
            return {key: 0.0 for key in weighted}
        return {key: weighted[key] / total for key in weighted}

    def _observed_funnel_mix(self, topics: list[Topic]) -> dict[str, float]:
        weighted: dict[str, float] = {"tofu": 0.0, "mofu": 0.0, "bofu": 0.0}
        total = 0.0
        for topic in topics:
            funnel_key = normalize_funnel_label(topic.funnel_stage) or funnel_from_intent(topic.dominant_intent)
            if funnel_key is None:
                continue
            fit_score = self._topic_fit_score(topic) or 0.0
            quality_score = float(topic.priority_score or topic.final_priority_score or 0.0)
            weight = max(0.05, (fit_score * 0.7) + ((quality_score / 100.0) * 0.3))
            weighted[funnel_key] += weight
            total += weight
        if total <= 0:
            return {key: 0.0 for key in weighted}
        return {key: weighted[key] / total for key in weighted}

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

    async def _build_serp_briefing(
        self,
        *,
        primary_keyword: str,
        search_intent: str,
        page_type: str,
        serp_features: list[str],
        serp_top_results: Any,
        serp_briefing_agent: SerpBriefingAgent,
    ) -> dict[str, Any]:
        """Build SERP best practices from top pages and a focused SERP agent."""
        top_pages = self._normalize_serp_top_results(serp_top_results)
        deterministic = self._derive_deterministic_serp_briefing(
            top_pages=[],
            search_intent=search_intent,
            page_type=page_type,
            serp_features=serp_features,
        )
        if not top_pages:
            return {
                "top_pages": [],
                "summary": deterministic["summary"],
                "best_practices": self._sanitize_serp_guidance_items(
                    deterministic["best_practices"],
                    limit=12,
                ),
                "recommended_sections": self._sanitize_serp_guidance_items(
                    deterministic["recommended_sections"],
                    limit=12,
                ),
                "opportunities_to_outperform": self._sanitize_serp_guidance_items(
                    deterministic["opportunities_to_outperform"],
                    limit=10,
                ),
            }

        fetch_candidates = [
            page
            for page in top_pages
            if self._is_blog_like_snapshot(page)
        ][:SERP_SNAPSHOT_MAX_FETCHED_BLOG_PAGES]
        if not fetch_candidates:
            fetch_candidates = top_pages[:1]

        fetched_headings_by_url = await self._fetch_serp_page_headings(fetch_candidates)
        enriched_pages: list[dict[str, Any]] = []
        for page in top_pages:
            page_url = str(page.get("url") or "")
            headings = fetched_headings_by_url.get(page_url, [])
            structural_signals = self._derive_serp_structural_signals(
                title=str(page.get("title") or ""),
                snippet=str(page.get("snippet") or ""),
                headings=headings,
                serp_features=serp_features,
            )
            enriched_pages.append({
                **page,
                "headings": headings,
                "structural_signals": structural_signals,
            })

        deterministic = self._derive_deterministic_serp_briefing(
            top_pages=enriched_pages,
            search_intent=search_intent,
            page_type=page_type,
            serp_features=serp_features,
        )

        snapshots: list[SerpPageSnapshot] = []
        for page in enriched_pages[:SERP_SNAPSHOT_TOP_RESULTS]:
            try:
                snapshots.append(
                    SerpPageSnapshot(
                        position=int(page.get("position") or 1),
                        title=str(page.get("title") or ""),
                        url=str(page.get("url") or ""),
                        domain=str(page.get("domain") or ""),
                        content_type_hint=str(page.get("content_type_hint") or "unknown"),
                        headings=[
                            str(item).strip()
                            for item in page.get("headings", [])
                            if str(item).strip()
                        ][:SERP_SNAPSHOT_MAX_HEADINGS_PER_PAGE],
                        structural_signals=[
                            str(item).strip()
                            for item in page.get("structural_signals", [])
                            if str(item).strip()
                        ],
                    )
                )
            except Exception:
                continue

        if not snapshots:
            return {
                "top_pages": enriched_pages,
                "summary": deterministic["summary"],
                "best_practices": self._sanitize_serp_guidance_items(
                    deterministic["best_practices"],
                    limit=12,
                ),
                "recommended_sections": self._sanitize_serp_guidance_items(
                    deterministic["recommended_sections"],
                    limit=12,
                ),
                "opportunities_to_outperform": self._sanitize_serp_guidance_items(
                    deterministic["opportunities_to_outperform"],
                    limit=10,
                ),
            }

        try:
            briefing_input = SerpBriefingInput(
                primary_keyword=primary_keyword,
                search_intent=search_intent,
                page_type=page_type,
                serp_features=serp_features,
                top_pages=snapshots,
            )
            briefing_output = await serp_briefing_agent.run(briefing_input)
            insight = briefing_output.insight
        except Exception as exc:
            logger.warning(
                "SERP briefing agent failed; using deterministic fallback",
                extra={
                    "primary_keyword": primary_keyword,
                    "error": str(exc),
                },
            )
            return {
                "top_pages": enriched_pages,
                "summary": deterministic["summary"],
                "best_practices": self._sanitize_serp_guidance_items(
                    deterministic["best_practices"],
                    limit=12,
                ),
                "recommended_sections": self._sanitize_serp_guidance_items(
                    deterministic["recommended_sections"],
                    limit=12,
                ),
                "opportunities_to_outperform": self._sanitize_serp_guidance_items(
                    deterministic["opportunities_to_outperform"],
                    limit=10,
                ),
            }

        return {
            "top_pages": enriched_pages,
            "summary": self._optional_str(insight.summary) or deterministic["summary"],
            "best_practices": self._sanitize_serp_guidance_items(
                self._merge_unique_str_items(
                    insight.best_practices,
                    deterministic["best_practices"],
                    limit=12,
                ),
                limit=12,
            ),
            "recommended_sections": self._sanitize_serp_guidance_items(
                self._merge_unique_str_items(
                    insight.recommended_sections,
                    deterministic["recommended_sections"],
                    limit=12,
                ),
                limit=12,
            ),
            "opportunities_to_outperform": self._sanitize_serp_guidance_items(
                self._merge_unique_str_items(
                    insight.opportunities_to_outperform,
                    deterministic["opportunities_to_outperform"],
                    limit=10,
                ),
                limit=10,
            ),
        }

    def _normalize_serp_top_results(self, serp_top_results: Any) -> list[dict[str, Any]]:
        """Normalize Step 8 SERP rows into deterministic top-page snapshots."""
        if not isinstance(serp_top_results, list):
            return []

        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(serp_top_results[:SERP_SNAPSHOT_TOP_RESULTS], start=1):
            if not isinstance(row, dict):
                continue

            url = str(row.get("url") or "").strip()
            if not self._is_http_url(url):
                continue

            title = str(row.get("title") or "").strip()
            snippet = str(row.get("snippet") or "").strip()
            raw_position = self._optional_int(row.get("position"))
            position = raw_position if raw_position is not None and raw_position > 0 else index
            domain = str(row.get("domain") or "").strip().lower()
            if not domain:
                domain = self._extract_domain_from_url(url)

            normalized.append({
                "position": position,
                "title": title,
                "url": url,
                "domain": domain,
                "snippet": snippet,
                "content_type_hint": self._infer_serp_content_type_hint(
                    title=title,
                    url=url,
                    snippet=snippet,
                ),
            })

        return normalized

    def _infer_serp_content_type_hint(self, *, title: str, url: str, snippet: str) -> str:
        """Infer a coarse result type from URL/title/snippet tokens."""
        normalized = f"{title} {snippet} {url}".lower()
        url_lower = url.lower()
        if any(token in url_lower for token in ("/docs", "/documentation", "/help", "/kb/")):
            return "docs"
        if any(
            token in url_lower
            for token in ("/pricing", "/product", "/features", "/solutions", "/platform")
        ):
            return "product"
        if any(token in normalized for token in (" vs ", "versus", "alternatives", "comparison")):
            return "comparison"
        if re.search(r"\b(top|best)\s+\d+\b", normalized):
            return "list"
        if any(
            token in url_lower
            for token in ("/blog/", "/post/", "/posts/", "/article/", "/articles/", "/guides/")
        ) or any(
            token in normalized
            for token in ("how to", "guide", "tutorial", "checklist", "playbook")
        ):
            return "blog"
        return "other"

    def _is_blog_like_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Return True if a SERP snapshot is likely a blog/editorial page."""
        hint = str(snapshot.get("content_type_hint") or "").strip().lower()
        return hint in {"blog", "comparison", "list"}

    def _is_http_url(self, value: str) -> bool:
        normalized = value.strip().lower()
        return normalized.startswith("http://") or normalized.startswith("https://")

    def _extract_domain_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        if domain.startswith("www."):
            return domain[4:]
        return domain

    async def _fetch_serp_page_headings(
        self,
        pages: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Fetch top pages and return extracted H2/H3 headings by URL."""
        urls = [
            str(page.get("url") or "").strip()
            for page in pages
            if self._is_http_url(str(page.get("url") or ""))
        ]
        if not urls:
            return {}

        try:
            async with WebsiteScraper(
                timeout=SERP_SNAPSHOT_FETCH_TIMEOUT_SECONDS,
                max_pages=max(1, len(urls)),
            ) as scraper:
                results = await asyncio.gather(
                    *(scraper.scrape_page(url) for url in urls),
                    return_exceptions=True,
                )
        except Exception as exc:
            logger.warning(
                "SERP page structure fetch failed",
                extra={"urls": len(urls), "error": str(exc)},
            )
            return {}

        headings_by_url: dict[str, list[str]] = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                continue
            if not isinstance(result, dict) or result.get("error"):
                continue

            headings: list[str] = []
            for item in result.get("headings", []):
                if not isinstance(item, dict):
                    continue
                level = self._optional_int(item.get("level"))
                text = str(item.get("text") or "").strip()
                if level not in {2, 3} or not text:
                    continue
                headings.append(text)

            headings_by_url[url] = self._merge_unique_str_items(
                headings,
                [],
                limit=SERP_SNAPSHOT_MAX_HEADINGS_PER_PAGE,
            )

        return headings_by_url

    def _derive_serp_structural_signals(
        self,
        *,
        title: str,
        snippet: str,
        headings: list[str],
        serp_features: list[str],
    ) -> list[str]:
        """Derive structural cues from title/snippet/headings."""
        corpus = " ".join([title, snippet, " ".join(headings)]).lower()
        signals: list[str] = []

        if headings:
            signals.append("explicit_h2_h3_structure")
        if "featured_snippet" in serp_features:
            signals.append("snippet_answer_block")
        if "paa" in serp_features:
            signals.append("faq_section")
        if "images" in serp_features or "video" in serp_features:
            signals.append("visual_examples")

        if re.search(r"\b(step|how to|walkthrough|process)\b", corpus):
            signals.append("step_by_step_flow")
        if re.search(r"\b(vs|versus|comparison|alternatives?)\b", corpus):
            signals.append("comparison_elements")
        if re.search(r"\b(checklist|template|worksheet)\b", corpus):
            signals.append("template_or_checklist")
        if re.search(r"\b(example|case study|real world)\b", corpus):
            signals.append("examples_or_case_studies")
        if re.search(r"\b(faq|questions?)\b", corpus):
            signals.append("faq_section")
        if re.search(r"\b(price|pricing|cost)\b", corpus):
            signals.append("pricing_context")
        if re.search(r"\b(202[0-9]|2030)\b", corpus):
            signals.append("freshness_year_markers")

        return self._merge_unique_str_items(signals, [], limit=12)

    def _derive_deterministic_serp_briefing(
        self,
        *,
        top_pages: list[dict[str, Any]],
        search_intent: str,
        page_type: str,
        serp_features: list[str],
    ) -> dict[str, Any]:
        """Fallback deterministic SERP guidance when live/LLM analysis is incomplete."""
        signal_counts: Counter[str] = Counter()
        for page in top_pages:
            signals = page.get("structural_signals", [])
            if not isinstance(signals, list):
                continue
            for signal in signals:
                signal_key = str(signal).strip().lower()
                if signal_key:
                    signal_counts[signal_key] += 1

        total_pages = len(top_pages)
        blog_like_count = sum(1 for page in top_pages if self._is_blog_like_snapshot(page))
        top_signals = ", ".join(name.replace("_", " ") for name, _ in signal_counts.most_common(3))

        best_practices: list[str] = []
        recommended_sections: list[str] = []
        opportunities: list[str] = []

        if signal_counts.get("snippet_answer_block", 0) > 0 or "featured_snippet" in serp_features:
            best_practices.append(
                "Open with a concise answer block in the first section to match snippet-style SERP behavior."
            )
            recommended_sections.append("Quick Answer / TL;DR")
        if signal_counts.get("step_by_step_flow", 0) > 0 or page_type == "guide":
            best_practices.append("Use a clear step-by-step flow with explicit sub-steps and outcomes.")
            recommended_sections.append("Step-by-Step Implementation")
        if signal_counts.get("comparison_elements", 0) > 0 or page_type in {"comparison", "alternatives"}:
            best_practices.append(
                "Include a scannable comparison framework (criteria, trade-offs, and recommendations)."
            )
            recommended_sections.append("Comparison Table")
        if signal_counts.get("examples_or_case_studies", 0) > 0:
            best_practices.append("Anchor key claims in practical examples or scenario-style evidence.")
            recommended_sections.append("Practical Examples")
        if signal_counts.get("template_or_checklist", 0) > 0:
            best_practices.append("Add reusable checklists or templates that help readers apply the advice.")
            recommended_sections.append("Checklist / Template")
        if signal_counts.get("faq_section", 0) > 0 or "paa" in serp_features:
            best_practices.append("Cover high-intent follow-up questions in a dedicated FAQ block.")
            recommended_sections.append("FAQ")
        if signal_counts.get("freshness_year_markers", 0) > 0:
            best_practices.append("Include freshness cues and up-to-date context for year-sensitive queries.")
            recommended_sections.append("What's New / Current Context")

        normalized_intent = search_intent.strip().lower()
        if normalized_intent in {"commercial", "transactional"}:
            best_practices.append(
                "Provide explicit decision criteria and bridge sections that support evaluation and conversion."
            )
            recommended_sections.append("Decision Criteria")
        if page_type in {"comparison", "alternatives", "list"}:
            recommended_sections.append("Top Options Summary")

        if signal_counts.get("examples_or_case_studies", 0) == 0:
            opportunities.append("Add scenario-based examples competitors do not clearly provide.")
        if signal_counts.get("template_or_checklist", 0) == 0:
            opportunities.append("Include a practical checklist/template to increase actionability.")
        if signal_counts.get("faq_section", 0) == 0 and "paa" in serp_features:
            opportunities.append("Close PAA-style intent gaps with concise objection-handling FAQs.")
        opportunities.append(
            "Differentiate with stronger examples, source-backed claims, and deeper edge-case coverage."
        )

        if total_pages > 0:
            summary = (
                f"{blog_like_count}/{total_pages} top results appear blog/editorial; "
                f"dominant structural patterns: {top_signals or 'mixed signals'}."
            )
        else:
            summary = (
                "No fetchable top-page structures; applying intent and page-type based SERP best practices."
            )

        return {
            "summary": summary,
            "best_practices": self._merge_unique_str_items(best_practices, [], limit=12),
            "recommended_sections": self._merge_unique_str_items(
                recommended_sections,
                [],
                limit=12,
            ),
            "opportunities_to_outperform": self._merge_unique_str_items(opportunities, [], limit=10),
        }

    def _merge_unique_str_items(
        self,
        primary: list[str] | None,
        secondary: list[str] | None,
        *,
        limit: int = 20,
    ) -> list[str]:
        """Merge two string lists with case-insensitive de-duplication."""
        merged: list[str] = []
        seen: set[str] = set()
        for item in (primary or []) + (secondary or []):
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
            if len(merged) >= limit:
                break
        return merged

    def _sanitize_serp_guidance_items(
        self,
        items: list[str] | None,
        *,
        limit: int,
    ) -> list[str]:
        """Drop non-content guidance so SERP advice stays focused on article body content."""
        sanitized: list[str] = []
        for item in items or []:
            text = self._optional_str(item)
            if text is None:
                continue
            normalized = text.casefold()
            if any(pattern.search(normalized) for pattern in SERP_NON_CONTENT_GUIDANCE_PATTERNS):
                continue
            sanitized.append(text)
            if len(sanitized) >= limit:
                break
        return sanitized

    def _build_publication_schedule_config(
        self,
        input_data: BriefInput,
        *,
        project_posts_per_week: int,
    ) -> PublicationScheduleConfig:
        """Normalize publication scheduling controls from step input."""
        requested_posts_per_week = self._coerce_posts_per_week(input_data.posts_per_week)
        posts_per_week = min(
            requested_posts_per_week,
            self._coerce_posts_per_week(project_posts_per_week),
        )
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
        reserved_date_counts: Counter[date] | None = None,
    ) -> list[date]:
        """Generate deterministic future publication slots based on cadence settings."""
        if topic_count <= 0:
            return []

        reserved_counts = Counter(reserved_date_counts or {})
        weekly_reserved_counts = self._build_weekly_reserved_counts(reserved_counts)

        earliest = today + timedelta(days=config.min_lead_days)
        if config.start_date is not None and config.start_date > earliest:
            earliest = config.start_date

        # Start from Monday of the anchor week, then emit selected weekdays per week.
        week_start = earliest - timedelta(days=earliest.weekday())
        slots: list[date] = []
        max_weeks_to_scan = max(topic_count * 8, 520)  # 10-year guardrail for sparse calendars.
        scanned_weeks = 0
        while len(slots) < topic_count and scanned_weeks < max_weeks_to_scan:
            week_key = self._iso_week_key(week_start)
            remaining_capacity = max(
                0,
                config.posts_per_week - weekly_reserved_counts.get(week_key, 0),
            )
            if remaining_capacity <= 0:
                week_start += timedelta(days=7)
                scanned_weeks += 1
                continue

            for weekday in config.weekdays:
                if remaining_capacity <= 0:
                    break
                candidate = week_start + timedelta(days=weekday)
                if candidate < earliest:
                    continue
                if reserved_counts.get(candidate, 0) > 0:
                    continue
                slots.append(candidate)
                reserved_counts[candidate] += 1
                weekly_reserved_counts[week_key] += 1
                remaining_capacity -= 1
                if len(slots) >= topic_count:
                    break
            week_start += timedelta(days=7)
            scanned_weeks += 1

        if len(slots) >= topic_count:
            return slots

        fallback = earliest
        max_shift_days = 365 * 10
        for _ in range(max_shift_days):
            week_key = self._iso_week_key(fallback)
            if (
                reserved_counts.get(fallback, 0) == 0
                and weekly_reserved_counts.get(week_key, 0) < config.posts_per_week
            ):
                slots.append(fallback)
                reserved_counts[fallback] += 1
                weekly_reserved_counts[week_key] += 1
                if len(slots) >= topic_count:
                    break
            fallback += timedelta(days=1)

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

    async def _load_existing_brief_signatures(self, project_id: str) -> list[ExistingBriefSignature]:
        """Load semantic signatures for already-persisted briefs."""
        result = await self.session.execute(
            select(ContentBrief).where(ContentBrief.project_id == project_id)
        )
        signatures: list[ExistingBriefSignature] = []
        for brief in result.scalars().all():
            primary_keyword = str(brief.primary_keyword or "").strip()
            supporting = [
                str(item).strip()
                for item in (brief.supporting_keywords or [])
                if str(item).strip()
            ]
            keyword_texts = [primary_keyword, *supporting]
            keyword_text = " ".join(keyword_texts)
            title_text = " ".join(
                str(item).strip()
                for item in (brief.working_titles or [])
                if str(item).strip()
            )
            comparison_key = build_comparison_key("", primary_keyword)
            family_key = build_family_key("", primary_keyword, supporting)
            signatures.append(
                ExistingBriefSignature(
                    comparison_key=comparison_key,
                    family_key=family_key,
                    intent=str(brief.search_intent or "").strip().lower(),
                    page_type=str(brief.page_type or "").strip().lower(),
                    keyword_tokens=normalize_text_tokens(keyword_text),
                    text_tokens=normalize_text_tokens(f"{title_text} {primary_keyword}".strip()),
                )
            )
        return signatures

    async def _load_existing_brief_history(
        self,
        project_id: str,
        *,
        limit: int = 80,
    ) -> list[dict[str, str]]:
        """Load existing brief history rows used for LLM diversity memory."""
        result = await self.session.execute(
            select(ContentBrief)
            .where(ContentBrief.project_id == project_id)
            .order_by(ContentBrief.created_at.desc())
            .limit(limit)
        )
        rows: list[dict[str, str]] = []
        for brief in result.scalars().all():
            topic_name = ""
            titles = brief.working_titles or []
            if titles:
                topic_name = str(titles[0] or "").strip()
            rows.append({
                "topic_name": topic_name,
                "primary_keyword": str(brief.primary_keyword or "").strip(),
                "search_intent": str(brief.search_intent or "").strip(),
                "page_type": str(brief.page_type or "").strip(),
                "created_at": (
                    brief.created_at.isoformat()
                    if getattr(brief, "created_at", None) is not None
                    else ""
                ),
            })
        return rows

    async def _apply_batch_diversification_selection(
        self,
        *,
        topics: list[Topic],
        primary_keywords_by_topic_id: dict[str, Keyword | None],
        existing_history: list[dict[str, str]],
        target_count: int,
    ) -> tuple[list[Topic], int, str]:
        """Apply LLM-assisted in-batch diversification with deterministic fallback."""
        if len(topics) <= 1:
            return topics, 0, ""

        max_count = max(1, min(target_count, len(topics)))
        candidates_payload = self._build_diversifier_candidates_payload(
            topics=topics,
            primary_keywords_by_topic_id=primary_keywords_by_topic_id,
        )
        if not candidates_payload:
            return topics[:max_count], max(0, len(topics) - max_count), "no_candidates_payload"

        agent = BriefDiversifierAgent()
        output = await self._run_brief_diversifier_with_retry(
            agent=agent,
            candidates=candidates_payload,
            existing_history=existing_history,
            target_count=max_count,
        )

        if output is not None:
            topic_by_id = {str(topic.id): topic for topic in topics}
            included_ids: list[str] = []
            for decision in output.decisions:
                decision_topic_id = str(decision.topic_id or "").strip()
                if not decision_topic_id or decision_topic_id not in topic_by_id:
                    continue
                if decision.decision.strip().lower() != "include":
                    continue
                if decision_topic_id not in included_ids:
                    included_ids.append(decision_topic_id)
                if len(included_ids) >= max_count:
                    break
            if included_ids:
                selected = [topic_by_id[topic_id] for topic_id in included_ids if topic_id in topic_by_id]
                return selected, len(topics) - len(selected), output.overall_notes or "llm_diversifier_selected"

        fallback_selected = self._deterministic_batch_diversification_fallback(
            topics=topics,
            primary_keywords_by_topic_id=primary_keywords_by_topic_id,
            target_count=max_count,
        )
        if fallback_selected:
            return (
                fallback_selected,
                len(topics) - len(fallback_selected),
                "deterministic_diversification_fallback",
            )
        return topics[:max_count], max(0, len(topics) - max_count), "diversification_noop"

    async def _run_brief_diversifier_with_retry(
        self,
        *,
        agent: BriefDiversifierAgent,
        candidates: list[dict[str, Any]],
        existing_history: list[dict[str, str]],
        target_count: int,
    ) -> Any | None:
        """Run brief diversifier with compact retry."""
        for attempt in range(BRIEF_DIVERSIFIER_RETRY_ATTEMPTS):
            compact_mode = attempt > 0
            try:
                return await agent.run(
                    BriefDiversifierInput(
                        candidates=candidates,
                        existing_topics=existing_history,
                        target_count=target_count,
                        compact_mode=compact_mode,
                    )
                )
            except Exception:
                logger.warning(
                    "Brief diversification LLM attempt failed",
                    extra={"attempt": attempt + 1, "compact_mode": compact_mode},
                )
        return None

    def _build_diversifier_candidates_payload(
        self,
        *,
        topics: list[Topic],
        primary_keywords_by_topic_id: dict[str, Keyword | None],
    ) -> list[dict[str, Any]]:
        """Build compact candidate payload for diversification selector."""
        payload: list[dict[str, Any]] = []
        for topic in topics:
            topic_id = str(topic.id)
            primary_kw = primary_keywords_by_topic_id.get(topic_id)
            payload.append({
                "topic_id": topic_id,
                "name": topic.name,
                "primary_keyword": str(primary_kw.keyword).strip() if primary_kw is not None else "",
                "intent": str(topic.dominant_intent or "").strip(),
                "page_type": str(topic.dominant_page_type or "").strip(),
                "funnel_stage": str(topic.funnel_stage or "").strip(),
                "fit_tier": str(topic.fit_tier or "").strip(),
                "fit_score": self._topic_fit_score(topic),
                "priority_rank": topic.priority_rank,
                "priority_score": topic.priority_score,
            })
        return payload

    def _deterministic_batch_diversification_fallback(
        self,
        *,
        topics: list[Topic],
        primary_keywords_by_topic_id: dict[str, Keyword | None],
        target_count: int,
    ) -> list[Topic]:
        """Greedy fallback that suppresses near-duplicates within this batch."""
        if not topics:
            return []
        max_count = max(1, min(target_count, len(topics)))

        signatures: dict[str, dict[str, Any]] = {}
        for topic in topics:
            topic_id = str(topic.id)
            primary_kw = primary_keywords_by_topic_id.get(topic_id)
            primary_keyword_text = str(primary_kw.keyword).strip() if primary_kw is not None else ""
            signatures[topic_id] = {
                "comparison_key": build_comparison_key(topic.name or "", primary_keyword_text),
                "family_key": build_family_key(topic.name or "", primary_keyword_text, []),
                "keyword_tokens": normalize_text_tokens(primary_keyword_text),
                "text_tokens": normalize_text_tokens(
                    f"{topic.name or ''} {topic.description or ''} {primary_keyword_text}".strip()
                ),
                "keyword_norm_text": re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", primary_keyword_text.lower())).strip(),
                "intent": str(topic.dominant_intent or "").strip().lower(),
                "page_type": str(topic.dominant_page_type or "").strip().lower(),
            }

        selected: list[Topic] = []
        for topic in topics:
            if len(selected) >= max_count:
                break
            topic_id = str(topic.id)
            signature = signatures[topic_id]
            is_duplicate = False
            for kept in selected:
                kept_signature = signatures[str(kept.id)]
                if is_exact_pair_duplicate(
                    signature.get("comparison_key"),
                    kept_signature.get("comparison_key"),
                ):
                    is_duplicate = True
                    break
                if is_sibling_pair(
                    signature.get("comparison_key"),
                    kept_signature.get("comparison_key"),
                ):
                    continue
                keyword_norm_text = str(signature.get("keyword_norm_text") or "")
                kept_keyword_norm_text = str(kept_signature.get("keyword_norm_text") or "")
                if (
                    signature.get("intent") == kept_signature.get("intent")
                    and signature.get("page_type") == kept_signature.get("page_type")
                    and keyword_norm_text
                    and kept_keyword_norm_text
                    and (
                        keyword_norm_text in kept_keyword_norm_text
                        or kept_keyword_norm_text in keyword_norm_text
                    )
                ):
                    is_duplicate = True
                    break
                keyword_jaccard = jaccard(
                    signature.get("keyword_tokens", set()),
                    kept_signature.get("keyword_tokens", set()),
                )
                text_jaccard = jaccard(
                    signature.get("text_tokens", set()),
                    kept_signature.get("text_tokens", set()),
                )
                if (
                    signature.get("intent") == kept_signature.get("intent")
                    and signature.get("page_type") == kept_signature.get("page_type")
                    and keyword_jaccard >= 0.66
                    and text_jaccard >= 0.66
                ):
                    is_duplicate = True
                    break
                overlap_score = compute_topic_overlap(
                    keyword_tokens_a=signature.get("keyword_tokens", set()),
                    keyword_tokens_b=kept_signature.get("keyword_tokens", set()),
                    text_tokens_a=signature.get("text_tokens", set()),
                    text_tokens_b=kept_signature.get("text_tokens", set()),
                    serp_domains_a={"no_serp_evidence"},
                    serp_domains_b={"no_serp_evidence"},
                    intent_a=signature.get("intent"),
                    intent_b=kept_signature.get("intent"),
                    page_type_a=signature.get("page_type"),
                    page_type_b=kept_signature.get("page_type"),
                )
                if overlap_score >= IN_BATCH_DIVERSIFICATION_OVERLAP_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(topic)

        if not selected:
            return topics[:1]
        return selected

    def _should_skip_as_covered(
        self,
        *,
        topic: Topic,
        primary_keyword: Keyword | None,
        existing_signatures: list[ExistingBriefSignature],
    ) -> tuple[bool, str | None, bool]:
        """Return skip decision for already-covered semantic signatures."""
        if primary_keyword is None:
            return False, None, False

        primary_keyword_text = str(primary_keyword.keyword or "").strip()
        comparison_key = build_comparison_key(topic.name or "", primary_keyword_text)
        family_key = build_family_key(topic.name or "", primary_keyword_text, [])
        candidate_keyword_tokens = normalize_text_tokens(primary_keyword_text)
        candidate_text_tokens = normalize_text_tokens(
            f"{topic.name or ''} {topic.description or ''} {primary_keyword_text}".strip()
        )
        candidate_intent = str(topic.dominant_intent or "").strip().lower()
        candidate_page_type = str(topic.dominant_page_type or "").strip().lower()

        sibling_pairs_allowed = False
        for signature in existing_signatures:
            if comparison_key and signature.comparison_key:
                if is_exact_pair_duplicate(comparison_key, signature.comparison_key):
                    return True, "existing_pair_covered", sibling_pairs_allowed
                if is_sibling_pair(comparison_key, signature.comparison_key):
                    sibling_pairs_allowed = True
                    continue

            if not candidate_intent or not candidate_page_type:
                continue
            if candidate_intent != signature.intent or candidate_page_type != signature.page_type:
                continue
            if family_key != signature.family_key and comparison_key is not None:
                continue

            overlap_score = compute_topic_overlap(
                keyword_tokens_a=candidate_keyword_tokens,
                keyword_tokens_b=signature.keyword_tokens,
                text_tokens_a=candidate_text_tokens,
                text_tokens_b=signature.text_tokens,
                serp_domains_a={"no_serp_evidence"},
                serp_domains_b={"no_serp_evidence"},
                intent_a=candidate_intent,
                intent_b=signature.intent,
                page_type_a=candidate_page_type,
                page_type_b=signature.page_type,
            )
            if overlap_score >= EXISTING_CONTENT_SKIP_THRESHOLD:
                return True, "existing_content_overlap", sibling_pairs_allowed

        return False, None, sibling_pairs_allowed

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
        """Deprecated strict guard; cross-intent coverage is now allowed when relevant."""
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
        pillars_by_slug = await self._ensure_allowed_pillars()

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
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project_posts_per_week = self._coerce_posts_per_week(project.posts_per_week)

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
            primary_keyword_id = self._optional_str(brief_data.get("primary_keyword_id"))
            search_intent = self._optional_str(brief_data.get("search_intent"))
            page_type = self._optional_str(brief_data.get("page_type"))
            funnel_stage = self._optional_str(brief_data.get("funnel_stage"))
            working_titles = self._str_list_or_none(brief_data.get("working_titles"))
            target_audience = self._optional_str(brief_data.get("target_audience"))
            reader_job_to_be_done = self._optional_str(brief_data.get("reader_job_to_be_done"))
            outline = self._dict_list_or_none(brief_data.get("outline"))
            supporting_keywords = self._str_list_or_none(brief_data.get("supporting_keywords"))
            supporting_keyword_ids = self._str_list_or_none(brief_data.get("supporting_keyword_ids"))
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
                posts_per_week_limit=project_posts_per_week,
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
                await sync_brief_keywords(
                    self.session,
                    brief=existing,
                    primary_keyword=primary_keyword,
                    supporting_keywords=supporting_keywords,
                    primary_keyword_id=primary_keyword_id,
                    supporting_keyword_ids=supporting_keyword_ids,
                )
                await self._persist_pillar_assignment(
                    brief_id=str(existing.id),
                    brief_data=brief_data,
                    pillars_by_slug=pillars_by_slug,
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

            created_brief = ContentBrief.create(
                self.session,
                create_dto,
            )
            await self.session.flush()
            await sync_brief_keywords(
                self.session,
                brief=created_brief,
                primary_keyword=primary_keyword,
                supporting_keywords=supporting_keywords,
                primary_keyword_id=primary_keyword_id,
                supporting_keyword_ids=supporting_keyword_ids,
            )
            await self._persist_pillar_assignment(
                brief_id=str(created_brief.id),
                brief_data=brief_data,
                pillars_by_slug=pillars_by_slug,
            )

        # Update project step
        project.current_step = max(project.current_step, self.step_number)
        strategy = await self.get_run_strategy()

        intent_counts = Counter(
            str(brief.get("search_intent") or "").strip().lower()
            for brief in result.briefs
            if str(brief.get("search_intent") or "").strip().lower() in {
                "informational",
                "commercial",
                "transactional",
            }
        )
        funnel_counts = Counter(
            str(brief.get("funnel_stage") or "").strip().lower()
            for brief in result.briefs
            if str(brief.get("funnel_stage") or "").strip().lower() in {"tofu", "mofu", "bofu"}
        )
        intent_total = sum(intent_counts.values())
        funnel_total = sum(funnel_counts.values())
        observed_intent_mix = {
            key: round(intent_counts.get(key, 0) / max(intent_total, 1), 4)
            for key in ("informational", "commercial", "transactional")
        }
        observed_funnel_mix = {
            key: round(funnel_counts.get(key, 0) / max(funnel_total, 1), 4)
            for key in ("tofu", "mofu", "bofu")
        }

        # Set result summary
        self.set_result_summary({
            "briefs_generated": result.briefs_generated,
            "briefs_with_warnings": result.briefs_with_warnings,
            "target_intent_mix": strategy.intent_mix.to_shares(),
            "observed_intent_mix": observed_intent_mix,
            "target_funnel_mix": strategy.funnel_mix.to_shares(),
            "observed_funnel_mix": observed_funnel_mix,
            "url_conflicts": sum(
                1 for b in result.briefs if b.get("url_collision_check") == "conflict"
            ),
            "url_warnings": sum(
                1 for b in result.briefs if b.get("url_collision_check") == "warning"
            ),
            "unchecked_overlap": sum(
                1 for b in result.briefs if b.get("overlap_status") == "unknown"
            ),
            "skipped_existing_pair": result.skipped_existing_pair,
            "skipped_existing_overlap": result.skipped_existing_overlap,
            "sibling_pairs_allowed": result.sibling_pairs_allowed,
            "skipped_batch_diversification": result.skipped_batch_diversification,
        })

        await self.session.commit()

    async def _persist_pillar_assignment(
        self,
        *,
        brief_id: str,
        brief_data: dict[str, Any],
        pillars_by_slug: dict[str, ContentPillar],
    ) -> None:
        pillar_slug = self._resolve_allowed_pillar_slug(brief_data.get("pillar_slug"))
        pillar = pillars_by_slug.get(pillar_slug)
        if pillar is None:
            return

        existing_result = await self.session.execute(
            select(ContentBriefPillarAssignment).where(
                ContentBriefPillarAssignment.project_id == self.project_id,
                ContentBriefPillarAssignment.brief_id == brief_id,
            )
        )
        for assignment in existing_result.scalars().all():
            await assignment.delete(self.session)

        confidence_value = brief_data.get("pillar_confidence")
        confidence = None
        try:
            if confidence_value is not None:
                confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = None
        assignment_method = self._optional_str(brief_data.get("pillar_assignment_method")) or "ai_brief"

        ContentBriefPillarAssignment.create(
            self.session,
            ContentBriefPillarAssignmentCreateDTO(
                project_id=self.project_id,
                brief_id=brief_id,
                pillar_id=str(pillar.id),
                relationship_type="primary",
                confidence_score=confidence,
                assignment_method=assignment_method,
            ),
        )

    def _resolve_allowed_pillar_slug(self, value: Any) -> str:
        slug = str(value or "").strip().lower()
        if slug in ALLOWED_PILLAR_CONFIG:
            return slug
        return "blog"

    async def _ensure_allowed_pillars(self) -> dict[str, ContentPillar]:
        slugs = list(ALLOWED_PILLAR_CONFIG.keys())
        result = await self.session.execute(
            select(ContentPillar).where(
                ContentPillar.project_id == self.project_id,
                ContentPillar.slug.in_(slugs),
            )
        )
        existing_by_slug = {str(pillar.slug).strip().lower(): pillar for pillar in result.scalars().all()}

        for slug, (name, description) in ALLOWED_PILLAR_CONFIG.items():
            pillar = existing_by_slug.get(slug)
            if pillar is None:
                created = ContentPillar.create(
                    self.session,
                    ContentPillarCreateDTO(
                        project_id=self.project_id,
                        name=name,
                        slug=slug,
                        description=description,
                        status="active",
                        source="ai",
                        locked=False,
                    ),
                )
                existing_by_slug[slug] = created
                continue

            patch_payload: dict[str, str] = {}
            if pillar.name != name:
                patch_payload["name"] = name
            if pillar.description != description:
                patch_payload["description"] = description
            if pillar.status != "active":
                patch_payload["status"] = "active"
            if patch_payload:
                pillar.patch(
                    self.session,
                    ContentPillarPatchDTO.from_partial(patch_payload),
                )

        await self.session.flush()
        refreshed_result = await self.session.execute(
            select(ContentPillar).where(
                ContentPillar.project_id == self.project_id,
                ContentPillar.slug.in_(slugs),
                ContentPillar.status == "active",
            )
        )
        return {
            str(pillar.slug).strip().lower(): pillar
            for pillar in refreshed_result.scalars().all()
        }

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

    def _coerce_posts_per_week(self, value: Any) -> int:
        """Clamp posts-per-week values into supported bounds."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 1
        return max(1, min(parsed, 7))

    def _iso_week_key(self, value: date) -> tuple[int, int]:
        """Return ISO year/week tuple used for weekly publication caps."""
        iso = value.isocalendar()
        return (iso.year, iso.week)

    def _build_weekly_reserved_counts(
        self,
        reserved_date_counts: Counter[date],
    ) -> Counter[tuple[int, int]]:
        weekly_counts: Counter[tuple[int, int]] = Counter()
        for publication_date, count in reserved_date_counts.items():
            if publication_date is None or count <= 0:
                continue
            weekly_counts[self._iso_week_key(publication_date)] += int(count)
        return weekly_counts

    def _resolve_unique_publication_date(
        self,
        *,
        desired_date: date | None,
        existing_date: date | None,
        reserved_date_counts: Counter[date],
        posts_per_week_limit: int | None = None,
    ) -> date | None:
        candidate = desired_date or existing_date
        if candidate is None:
            return None
        return self._next_available_publication_date(
            candidate,
            reserved_date_counts=reserved_date_counts,
            posts_per_week_limit=posts_per_week_limit,
        )

    def _next_available_publication_date(
        self,
        candidate: date,
        *,
        reserved_date_counts: Counter[date],
        posts_per_week_limit: int | None = None,
    ) -> date:
        max_shift_days = 365 * 3
        current = candidate
        weekly_limit = 7 if posts_per_week_limit is None else self._coerce_posts_per_week(
            posts_per_week_limit
        )
        weekly_counts = self._build_weekly_reserved_counts(reserved_date_counts)
        for _ in range(max_shift_days):
            week_key = self._iso_week_key(current)
            if (
                reserved_date_counts.get(current, 0) == 0
                and weekly_counts.get(week_key, 0) < weekly_limit
            ):
                return current
            current += timedelta(days=1)
        return current
