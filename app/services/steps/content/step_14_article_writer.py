"""Step 14: Generate modular SEO articles from briefs and writing templates."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.models.brand import BrandProfile
from app.models.content import (
    ContentArticle,
    ContentArticleVersion,
    ContentBrief,
    WriterInstructions,
)
from app.models.generated_dtos import (
    ContentArticleCreateDTO,
    ContentArticleVersionCreateDTO,
)
from app.models.project import Project
from app.models.style_guide import BriefDelta
from app.services.article_generation import ArticleGenerationService
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class ArticleWriterInput:
    """Input for Step 14 article generation."""

    project_id: str
    brief_ids: list[str] | None = None


@dataclass
class ArticleWriterOutput:
    """Output for Step 14 article generation."""

    articles_generated: int
    articles_needing_review: int
    articles_skipped_existing: int
    articles_failed: int
    article_ids: list[str] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)


class Step14ArticleWriterService(BaseStepService[ArticleWriterInput, ArticleWriterOutput]):
    """Step 14: Generate publish-ready modular content artifacts."""

    step_number = 14
    step_name = "article_generation"
    is_optional = False

    async def _validate_preconditions(self, input_data: ArticleWriterInput) -> None:
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        brief_stmt = select(ContentBrief).where(ContentBrief.project_id == input_data.project_id)
        if input_data.brief_ids:
            brief_stmt = brief_stmt.where(
                ContentBrief.id.in_([uuid.UUID(brief_id) for brief_id in input_data.brief_ids])
            )

        brief_result = await self.session.execute(brief_stmt.limit(1))
        if not brief_result.scalars().first():
            raise ValueError("No content briefs found. Run Step 12 first.")

        instructions_result = await self.session.execute(
            select(WriterInstructions)
            .join(ContentBrief, WriterInstructions.brief_id == ContentBrief.id)
            .where(ContentBrief.project_id == input_data.project_id)
            .limit(1)
        )
        if not instructions_result.scalars().first():
            raise ValueError("No writer instructions found. Run Step 13 first.")

    async def _execute(self, input_data: ArticleWriterInput) -> ArticleWriterOutput:
        project = await self._load_project(input_data.project_id)
        brand = await self._load_brand(input_data.project_id)
        strategy = await self.get_run_strategy()

        await self._update_progress(5, "Loading brief artifacts for article generation...")

        briefs = await self._load_briefs(input_data)
        if not briefs:
            return ArticleWriterOutput(
                articles_generated=0,
                articles_needing_review=0,
                articles_skipped_existing=0,
                articles_failed=0,
            )

        instructions = await self._load_writer_instructions(briefs)
        deltas = await self._load_brief_deltas(briefs)
        existing_article_brief_ids = await self._load_existing_article_brief_ids(briefs)

        brand_context = self._build_brand_context(brand)
        generator = ArticleGenerationService(project.domain)

        missing_instruction_failures: list[dict[str, str]] = []
        eligible_briefs: list[ContentBrief] = []
        skipped_existing = 0

        for brief in briefs:
            if brief.id in existing_article_brief_ids:
                skipped_existing += 1
                continue
            if brief.id not in instructions:
                missing_instruction_failures.append(
                    {
                        "brief_id": str(brief.id),
                        "reason": "missing_writer_instructions",
                    }
                )
                continue
            eligible_briefs.append(brief)

        if not eligible_briefs:
            return ArticleWriterOutput(
                articles_generated=0,
                articles_needing_review=0,
                articles_skipped_existing=skipped_existing,
                articles_failed=len(missing_instruction_failures),
                failures=missing_instruction_failures,
            )

        await self._update_progress(
            20,
            f"Generating modular articles for {len(eligible_briefs)} briefs...",
        )

        semaphore = asyncio.Semaphore(4)

        async def _generate_for_brief(brief: ContentBrief) -> tuple[ContentBrief, Any | None, str | None]:
            async with semaphore:
                try:
                    brief_payload = self._brief_payload(brief)
                    instructions_payload = instructions[brief.id]
                    delta_payload = deltas.get(brief.id, {})
                    artifact = await generator.generate_with_repair(
                        brief=brief_payload,
                        writer_instructions=instructions_payload,
                        brief_delta=delta_payload,
                        brand_context=brand_context,
                        conversion_intents=strategy.conversion_intents,
                    )
                    return brief, artifact, None
                except Exception as exc:
                    logger.warning(
                        "Article generation failed for brief",
                        extra={"brief_id": str(brief.id), "error": str(exc)},
                    )
                    return brief, None, str(exc)

        results = await asyncio.gather(*[_generate_for_brief(brief) for brief in eligible_briefs])

        await self._update_progress(75, "Saving generated articles and versions...")

        generated = 0
        needs_review = 0
        failed = len(missing_instruction_failures)
        failures = list(missing_instruction_failures)
        article_ids: list[str] = []

        for brief, artifact, error in results:
            if artifact is None:
                failed += 1
                failures.append(
                    {
                        "brief_id": str(brief.id),
                        "reason": error or "unknown_error",
                    }
                )
                continue

            article = ContentArticle.create(
                self.session,
                ContentArticleCreateDTO(
                    project_id=self.project_id,
                    brief_id=str(brief.id),
                    title=artifact.title,
                    slug=artifact.slug,
                    primary_keyword=artifact.primary_keyword,
                    modular_document=artifact.modular_document,
                    rendered_html=artifact.rendered_html,
                    qa_report=artifact.qa_report,
                    status=artifact.status,
                    current_version=1,
                    generation_model=artifact.generation_model,
                    generation_temperature=artifact.generation_temperature,
                ),
            )
            await self.session.flush()

            ContentArticleVersion.create(
                self.session,
                ContentArticleVersionCreateDTO(
                    article_id=str(article.id),
                    version_number=1,
                    title=artifact.title,
                    slug=artifact.slug,
                    primary_keyword=artifact.primary_keyword,
                    modular_document=artifact.modular_document,
                    rendered_html=artifact.rendered_html,
                    qa_report=artifact.qa_report,
                    status=artifact.status,
                    change_reason="initial_generation",
                    generation_model=artifact.generation_model,
                    generation_temperature=artifact.generation_temperature,
                    created_by_regeneration=False,
                ),
            )

            generated += 1
            article_ids.append(str(article.id))
            if artifact.status == "needs_review":
                needs_review += 1

        if eligible_briefs and generated == 0:
            raise ValueError("Step 14 generated 0 articles for eligible briefs")

        await self._update_progress(100, "Article generation complete")

        return ArticleWriterOutput(
            articles_generated=generated,
            articles_needing_review=needs_review,
            articles_skipped_existing=skipped_existing,
            articles_failed=failed,
            article_ids=article_ids,
            failures=failures,
        )

    async def _persist_results(self, result: ArticleWriterOutput) -> None:
        project = await self._load_project(self.project_id)
        project.current_step = max(project.current_step, 14)

        self.set_result_summary(
            {
                "articles_generated": result.articles_generated,
                "articles_needing_review": result.articles_needing_review,
                "articles_skipped_existing": result.articles_skipped_existing,
                "articles_failed": result.articles_failed,
            }
        )

        await self.session.commit()

    async def _load_project(self, project_id: str) -> Project:
        result = await self.session.execute(select(Project).where(Project.id == project_id))
        return result.scalar_one()

    async def _load_brand(self, project_id: str) -> BrandProfile | None:
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == project_id)
        )
        return result.scalar_one_or_none()

    async def _load_briefs(self, input_data: ArticleWriterInput) -> list[ContentBrief]:
        stmt = select(ContentBrief).where(ContentBrief.project_id == input_data.project_id)
        if input_data.brief_ids:
            stmt = stmt.where(
                ContentBrief.id.in_([uuid.UUID(brief_id) for brief_id in input_data.brief_ids])
            )
        result = await self.session.execute(stmt.order_by(ContentBrief.created_at.asc()))
        return list(result.scalars())

    async def _load_writer_instructions(self, briefs: list[ContentBrief]) -> dict[str, dict[str, Any]]:
        result = await self.session.execute(
            select(WriterInstructions).where(
                WriterInstructions.brief_id.in_([brief.id for brief in briefs])
            )
        )
        payload: dict[str, dict[str, Any]] = {}
        for item in result.scalars():
            payload[item.brief_id] = {
                "voice_tone_constraints": item.voice_tone_constraints or {},
                "forbidden_claims": item.forbidden_claims or [],
                "compliance_notes": item.compliance_notes or [],
                "formatting_requirements": item.formatting_requirements or {},
                "h1_h2_usage": item.h1_h2_usage or {},
                "internal_linking_minimums": item.internal_linking_minimums or {},
                "schema_guidance": item.schema_guidance or "",
                "qa_checklist": item.qa_checklist or [],
                "pass_fail_thresholds": item.pass_fail_thresholds or {},
                "common_failure_modes": item.common_failure_modes or [],
            }
        return payload

    async def _load_brief_deltas(self, briefs: list[ContentBrief]) -> dict[str, dict[str, Any]]:
        result = await self.session.execute(
            select(BriefDelta).where(BriefDelta.brief_id.in_([brief.id for brief in briefs]))
        )
        payload: dict[str, dict[str, Any]] = {}
        for item in result.scalars():
            payload[item.brief_id] = {
                "page_type_rules": item.page_type_rules or {},
                "must_include_sections": item.must_include_sections or [],
                "h1_h2_usage": item.h1_h2_usage or {},
                "schema_type": item.schema_type or "Article",
                "additional_qa_items": item.additional_qa_items or [],
            }
        return payload

    async def _load_existing_article_brief_ids(self, briefs: list[ContentBrief]) -> set[str]:
        result = await self.session.execute(
            select(ContentArticle.brief_id).where(
                ContentArticle.brief_id.in_([brief.id for brief in briefs])
            )
        )
        return set(result.scalars())

    def _brief_payload(self, brief: ContentBrief) -> dict[str, Any]:
        return {
            "id": str(brief.id),
            "primary_keyword": brief.primary_keyword,
            "search_intent": brief.search_intent,
            "page_type": brief.page_type,
            "funnel_stage": brief.funnel_stage,
            "working_titles": brief.working_titles or [],
            "target_audience": brief.target_audience or "",
            "proposed_publication_date": (
                brief.proposed_publication_date.isoformat()
                if brief.proposed_publication_date is not None
                else None
            ),
            "reader_job_to_be_done": brief.reader_job_to_be_done or "",
            "outline": brief.outline or [],
            "supporting_keywords": brief.supporting_keywords or [],
            "examples_required": brief.examples_required or [],
            "faq_questions": brief.faq_questions or [],
            "recommended_schema_type": brief.recommended_schema_type or "Article",
            "internal_links_out": brief.internal_links_out or [],
            "money_page_links": brief.money_page_links or [],
            "meta_title_guidelines": brief.meta_title_guidelines or "",
            "meta_description_guidelines": brief.meta_description_guidelines or "",
            "target_word_count_min": brief.target_word_count_min,
            "target_word_count_max": brief.target_word_count_max,
            "must_include_sections": brief.must_include_sections or [],
        }

    def _build_brand_context(self, brand: BrandProfile | None) -> str:
        if not brand:
            return ""

        parts: list[str] = []
        if brand.company_name:
            parts.append(f"Company: {brand.company_name}")
        if brand.tagline:
            parts.append(f"Tagline: {brand.tagline}")
        if brand.products_services:
            for product in brand.products_services[:5]:
                if not isinstance(product, dict):
                    continue
                name = product.get("name")
                description = product.get("description")
                if name:
                    parts.append(f"Product: {name}")
                if description:
                    parts.append(f"Description: {description}")
                benefits = product.get("core_benefits")
                if isinstance(benefits, list) and benefits:
                    parts.append(f"Core Benefits: {', '.join(str(b) for b in benefits[:6])}")

        if brand.unique_value_props:
            parts.append(f"UVPs: {', '.join(brand.unique_value_props[:6])}")
        if brand.differentiators:
            parts.append(f"Differentiators: {', '.join(brand.differentiators[:6])}")
        if brand.target_roles:
            parts.append(f"Target Roles: {', '.join(brand.target_roles[:6])}")
        if brand.target_industries:
            parts.append(f"Target Industries: {', '.join(brand.target_industries[:6])}")
        if brand.primary_pains:
            parts.append(f"Primary Pains: {', '.join(brand.primary_pains[:6])}")
        if brand.allowed_claims:
            parts.append(f"Allowed Claims: {', '.join(brand.allowed_claims[:6])}")
        if brand.restricted_claims:
            parts.append(f"Restricted Claims: {', '.join(brand.restricted_claims[:6])}")

        return "\n".join(parts)
