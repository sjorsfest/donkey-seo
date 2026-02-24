"""Step 4: Generate featured images from LLM template specs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.models.brand import BrandProfile
from app.models.content import ContentArticle, ContentBrief, ContentFeaturedImage
from app.models.project import Project
from app.services.featured_image_generation import (
    FeaturedImageGenerationService,
    generation_retry_settings,
    retry_with_backoff,
)
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class FeaturedImageInput:
    """Input for Step 4 featured image generation."""

    project_id: str
    brief_ids: list[str] | None = None
    force_regenerate: bool = False


@dataclass
class FeaturedImageOutput:
    """Output for Step 4 featured image generation."""

    briefs_processed: int
    images_generated: int
    images_reused: int
    images_skipped_existing_articles: int
    images_failed: int
    failures: list[dict[str, str]] = field(default_factory=list)


class Step4FeaturedImageService(BaseStepService[FeaturedImageInput, FeaturedImageOutput]):
    """Step 4: Produce deterministic featured images for briefs."""

    step_number = 4
    step_name = "featured_image_generation"
    is_optional = False

    async def _validate_preconditions(self, input_data: FeaturedImageInput) -> None:
        project_result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        if not project_result.scalar_one_or_none():
            raise ValueError(f"Project not found: {input_data.project_id}")

        brief_stmt = select(ContentBrief).where(ContentBrief.project_id == input_data.project_id)
        if input_data.brief_ids:
            brief_stmt = brief_stmt.where(ContentBrief.id.in_(input_data.brief_ids))
        brief_result = await self.session.execute(brief_stmt.limit(1))
        if not brief_result.scalars().first():
            raise ValueError("No content briefs found. Run Step 1 first.")

    async def _execute(self, input_data: FeaturedImageInput) -> FeaturedImageOutput:
        await self._update_progress(5, "Loading briefs and brand context...")

        brand = await self._load_brand(input_data.project_id)
        briefs = await self._load_briefs(input_data)
        if not briefs:
            return FeaturedImageOutput(
                briefs_processed=0,
                images_generated=0,
                images_reused=0,
                images_skipped_existing_articles=0,
                images_failed=0,
            )

        existing_images = await self._load_existing_images(briefs)
        existing_article_brief_ids = await self._load_existing_article_brief_ids(briefs)

        attempts, backoff_ms = generation_retry_settings()
        generator = FeaturedImageGenerationService()

        generated = 0
        reused = 0
        skipped_existing_articles = 0
        failures: list[dict[str, str]] = []

        await self._update_progress(20, f"Generating featured images for {len(briefs)} briefs...")

        for index, brief in enumerate(briefs, start=1):
            brief_id = str(brief.id)
            title_text = generator.locked_title_for_brief(brief)

            if brief.id in existing_article_brief_ids and not input_data.force_regenerate:
                skipped_existing_articles += 1
                continue

            existing = existing_images.get(brief_id)
            was_reused = (
                existing is not None
                and existing.status == "ready"
                and bool(existing.object_key)
                and existing.title_text == title_text
                and not input_data.force_regenerate
            )

            try:
                image = await retry_with_backoff(
                    attempts=attempts,
                    backoff_ms=backoff_ms,
                    coro_factory=lambda: generator.generate_for_brief(
                        session=self.session,
                        project_id=self.project_id,
                        brief=brief,
                        brand=brand,
                        existing=existing,
                        force_regenerate=input_data.force_regenerate,
                    ),
                )
                existing_images[brief_id] = image
                if was_reused:
                    reused += 1
                else:
                    generated += 1
            except Exception as exc:
                logger.warning(
                    "Featured image generation failed",
                    extra={"brief_id": brief_id, "error": str(exc)},
                )
                await generator.mark_failure(
                    session=self.session,
                    project_id=self.project_id,
                    brief_id=brief_id,
                    title_text=title_text,
                    error_message=str(exc),
                    existing=existing,
                )
                failures.append({"brief_id": brief_id, "reason": str(exc)})

            step_progress = 20 + (70 * index / max(1, len(briefs)))
            await self._update_progress(step_progress, f"Processed {index}/{len(briefs)} featured images...")

        if failures:
            raise ValueError(
                f"Featured image generation failed for {len(failures)} briefs"
            )

        await self._update_progress(100, "Featured image generation complete")

        return FeaturedImageOutput(
            briefs_processed=len(briefs),
            images_generated=generated,
            images_reused=reused,
            images_skipped_existing_articles=skipped_existing_articles,
            images_failed=len(failures),
            failures=failures,
        )

    async def _persist_results(self, result: FeaturedImageOutput) -> None:
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = max(project.current_step, self.step_number)

        self.set_result_summary(
            {
                "briefs_processed": result.briefs_processed,
                "images_generated": result.images_generated,
                "images_reused": result.images_reused,
                "images_skipped_existing_articles": result.images_skipped_existing_articles,
                "images_failed": result.images_failed,
            }
        )

        await self.session.commit()

    async def _load_brand(self, project_id: str) -> BrandProfile | None:
        result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == project_id)
        )
        return result.scalar_one_or_none()

    async def _load_briefs(self, input_data: FeaturedImageInput) -> list[ContentBrief]:
        stmt = select(ContentBrief).where(ContentBrief.project_id == input_data.project_id)
        if input_data.brief_ids:
            stmt = stmt.where(ContentBrief.id.in_(input_data.brief_ids))
        result = await self.session.execute(stmt.order_by(ContentBrief.created_at.asc()))
        return list(result.scalars())

    async def _load_existing_images(self, briefs: list[ContentBrief]) -> dict[str, ContentFeaturedImage]:
        result = await self.session.execute(
            select(ContentFeaturedImage).where(
                ContentFeaturedImage.brief_id.in_([brief.id for brief in briefs])
            )
        )
        return {str(item.brief_id): item for item in result.scalars()}

    async def _load_existing_article_brief_ids(self, briefs: list[ContentBrief]) -> set[str]:
        result = await self.session.execute(
            select(ContentArticle.brief_id).where(
                ContentArticle.brief_id.in_([brief.id for brief in briefs])
            )
        )
        return set(result.scalars())
