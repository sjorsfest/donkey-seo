"""Step 12: Content Brief Generation.

Generates writer-ready briefs for prioritized topics.
Includes URL slug generation and cannibalization guardrails.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.brief_generator import BriefGeneratorAgent, BriefGeneratorInput
from app.models.brand import BrandProfile
from app.models.content import ContentBrief
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.topic import Topic
from app.services.steps.base_step import BaseStepService, StepResult


@dataclass
class BriefInput:
    """Input for Step 12."""

    project_id: str
    topic_ids: list[str] | None = None  # Specific topics, or None for all prioritized
    max_briefs: int = 20  # Limit number of briefs to generate


@dataclass
class BriefOutput:
    """Output from Step 12."""

    briefs_generated: int
    briefs_with_warnings: int
    briefs: list[dict[str, Any]] = field(default_factory=list)


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
        """Validate Step 7 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 7:
            raise ValueError("Step 7 (Prioritization) must be completed first")

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
        # Load project and brand
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()

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
                    Topic.id.in_([uuid.UUID(tid) for tid in input_data.topic_ids]),
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

        all_topics = list(topics_result.scalars())[:input_data.max_briefs]

        if not all_topics:
            return BriefOutput(
                briefs_generated=0,
                briefs_with_warnings=0,
            )

        # Prepare brand context
        brand_context = self._build_brand_context(brand)
        money_pages = self._extract_money_pages(brand)

        # Check if Step 10 (cannibalization) ran
        step_10_ran = project.current_step >= 10

        # Load existing URLs for collision check (if Step 9 ran)
        existing_urls = await self._load_existing_urls(input_data.project_id)

        await self._update_progress(10, f"Generating {len(all_topics)} briefs...")

        # Generate briefs
        agent = BriefGeneratorAgent()
        output_briefs = []
        briefs_with_warnings = 0

        for i, topic in enumerate(all_topics):
            progress = 10 + int((i / len(all_topics)) * 85)
            await self._update_progress(
                progress,
                f"Generating brief {i + 1}/{len(all_topics)}: {topic.name}"
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

            # Generate URL slug
            url_slug = self._generate_url_slug(
                primary_kw.keyword,
                topic.dominant_page_type or "blog",
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
            )
            if has_warnings:
                briefs_with_warnings += 1

            try:
                # Generate brief with LLM
                agent_input = BriefGeneratorInput(
                    topic_name=topic.name,
                    primary_keyword=primary_kw.keyword,
                    supporting_keywords=[kw.keyword for kw in keywords if kw.id != primary_kw.id][:15],
                    search_intent=topic.dominant_intent or "informational",
                    page_type=topic.dominant_page_type or "guide",
                    funnel_stage=topic.funnel_stage or "tofu",
                    brand_context=brand_context,
                    money_pages=topic.target_money_pages or money_pages[:3],
                )

                output = await agent.run(agent_input)
                brief_data = output.brief

                output_briefs.append({
                    "topic_id": str(topic.id),
                    "topic_name": topic.name,
                    "primary_keyword": primary_kw.keyword,
                    "search_intent": topic.dominant_intent,
                    "page_type": topic.dominant_page_type,
                    "funnel_stage": topic.funnel_stage,
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
                    "supporting_keywords": [kw.keyword for kw in keywords if kw.id != primary_kw.id],
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
                    "has_warnings": has_warnings,
                    "warnings": self._collect_warnings(
                        collision_status,
                        do_not_target,
                        overlap_status,
                    ),
                })

            except Exception:
                # Fallback: create basic brief structure
                output_briefs.append({
                    "topic_id": str(topic.id),
                    "topic_name": topic.name,
                    "primary_keyword": primary_kw.keyword,
                    "search_intent": topic.dominant_intent,
                    "page_type": topic.dominant_page_type,
                    "funnel_stage": topic.funnel_stage,
                    "proposed_url_slug": url_slug,
                    "url_collision_check": collision_status,
                    "do_not_target": do_not_target,
                    "overlap_status": overlap_status,
                    "working_titles": [f"[Title for: {primary_kw.keyword}]"],
                    "target_audience": "To be defined",
                    "reader_job_to_be_done": "To be defined",
                    "outline": [],
                    "supporting_keywords": [kw.keyword for kw in keywords if kw.id != primary_kw.id],
                    "target_word_count": {"min": 1500, "max": 2500},
                    "has_warnings": True,
                    "warnings": ["LLM generation failed - manual completion required"],
                })
                briefs_with_warnings += 1

        await self._update_progress(100, "Brief generation complete")

        return BriefOutput(
            briefs_generated=len(output_briefs),
            briefs_with_warnings=briefs_with_warnings,
            briefs=output_briefs,
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
        if brand.products_services:
            products = [p.get("name", "") for p in brand.products_services[:5]]
            parts.append(f"Products/Services: {', '.join(products)}")
        if brand.unique_value_props:
            parts.append(f"Value Props: {', '.join(brand.unique_value_props[:3])}")
        if brand.tone_attributes:
            parts.append(f"Tone: {', '.join(brand.tone_attributes[:3])}")

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
        """Build internal link recommendations."""
        links = []

        # Link to pillar page
        if topic.pillar_seed_topic_id:
            links.append({
                "target": "pillar",
                "anchor_suggestion": topic.name,
                "context": "In introduction, link to pillar content",
            })

        # Link to related topics (based on same funnel stage)
        links.append({
            "target": "related_topic",
            "anchor_suggestion": "related content",
            "context": "In conclusion or sidebar, link to related topics",
        })

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

        return warnings

    async def _persist_results(self, result: BriefOutput) -> None:
        """Save content briefs to database."""
        for brief_data in result.briefs:
            # Create ContentBrief record
            brief = ContentBrief(
                id=uuid.uuid4(),
                project_id=uuid.UUID(self.project_id),
                topic_id=uuid.UUID(brief_data["topic_id"]),
                primary_keyword=brief_data["primary_keyword"],
                search_intent=brief_data.get("search_intent"),
                page_type=brief_data.get("page_type"),
                funnel_stage=brief_data.get("funnel_stage"),
                working_titles=brief_data.get("working_titles"),
                target_audience=brief_data.get("target_audience"),
                reader_job_to_be_done=brief_data.get("reader_job_to_be_done"),
                outline=brief_data.get("outline"),
                supporting_keywords=brief_data.get("supporting_keywords"),
                supporting_keywords_map=brief_data.get("supporting_keywords_map"),
                examples_required=brief_data.get("examples_required"),
                faq_questions=brief_data.get("faq_questions"),
                recommended_schema_type=brief_data.get("recommended_schema_type"),
                internal_links_out=brief_data.get("internal_links_out"),
                money_page_links=brief_data.get("money_page_links"),
                meta_title_guidelines=brief_data.get("meta_title_guidelines"),
                meta_description_guidelines=brief_data.get("meta_description_guidelines"),
                target_word_count_min=brief_data.get("target_word_count", {}).get("min"),
                target_word_count_max=brief_data.get("target_word_count", {}).get("max"),
                must_include_sections=brief_data.get("must_include_sections"),
                status="draft",
            )
            self.session.add(brief)

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
