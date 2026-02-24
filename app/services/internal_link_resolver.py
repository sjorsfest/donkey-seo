"""Resolve deferred internal links when target articles receive published URLs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentArticle, ContentArticleVersion
from app.models.generated_dtos import ContentArticlePatchDTO, ContentArticleVersionCreateDTO
from app.services.content_renderer import render_modular_document


def _normalize_published_url(url: str) -> str:
    stripped = str(url).strip()
    if not stripped:
        return ""
    parsed = urlparse(stripped)
    if parsed.scheme in {"http", "https"}:
        return stripped
    return f"https://{stripped.lstrip('/')}"


def replace_target_brief_links_in_document(
    *,
    document: dict[str, Any],
    target_brief_id: str,
    published_url: str,
) -> tuple[dict[str, Any], int]:
    """Replace hrefs for links that point to a target brief id."""
    if not isinstance(document, dict):
        return document, 0

    blocks = document.get("blocks")
    if not isinstance(blocks, list):
        return document, 0

    updated_count = 0
    updated_document = deepcopy(document)
    updated_blocks = updated_document.get("blocks")
    if not isinstance(updated_blocks, list):
        return document, 0

    for block in updated_blocks:
        if not isinstance(block, dict):
            continue

        links = block.get("links")
        if isinstance(links, list):
            for link in links:
                if not isinstance(link, dict):
                    continue
                if str(link.get("target_brief_id") or "").strip() != target_brief_id:
                    continue
                if str(link.get("href") or "").strip() == published_url:
                    continue
                link["href"] = published_url
                updated_count += 1

        cta = block.get("cta")
        if isinstance(cta, dict):
            if str(cta.get("target_brief_id") or "").strip() == target_brief_id:
                if str(cta.get("href") or "").strip() != published_url:
                    cta["href"] = published_url
                    updated_count += 1

    return updated_document, updated_count


class InternalLinkResolverService:
    """Backfill deferred internal links across draft/scheduled articles."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def resolve_for_published_brief(
        self,
        *,
        project_id: str,
        published_brief_id: str,
        published_url: str,
    ) -> int:
        normalized_url = _normalize_published_url(published_url)
        if not normalized_url:
            return 0

        result = await self.session.execute(
            select(ContentArticle).where(
                ContentArticle.project_id == project_id,
                ContentArticle.brief_id != published_brief_id,
            )
        )
        articles = list(result.scalars())

        updated_articles = 0
        for article in articles:
            if article.publish_status == "published" or article.published_at is not None:
                continue

            document = article.modular_document if isinstance(article.modular_document, dict) else {}
            updated_document, replacements = replace_target_brief_links_in_document(
                document=document,
                target_brief_id=published_brief_id,
                published_url=normalized_url,
            )
            if replacements <= 0:
                continue

            rendered_html = render_modular_document(updated_document)
            next_version = int(article.current_version or 1) + 1

            article.patch(
                self.session,
                ContentArticlePatchDTO.from_partial(
                    {
                        "modular_document": updated_document,
                        "rendered_html": rendered_html,
                        "current_version": next_version,
                    }
                ),
            )

            ContentArticleVersion.create(
                self.session,
                ContentArticleVersionCreateDTO(
                    article_id=str(article.id),
                    version_number=next_version,
                    title=article.title,
                    slug=article.slug,
                    primary_keyword=article.primary_keyword,
                    modular_document=updated_document,
                    rendered_html=rendered_html,
                    qa_report=article.qa_report,
                    status=article.status,
                    change_reason="internal_link_resolution",
                    generation_model=article.generation_model,
                    generation_temperature=article.generation_temperature,
                    created_by_regeneration=False,
                ),
            )
            updated_articles += 1

        return updated_articles


async def resolve_deferred_internal_links_for_published_article(
    session: AsyncSession,
    *,
    project_id: str,
    published_brief_id: str,
    published_url: str,
) -> int:
    """Convenience wrapper used by API routes."""
    service = InternalLinkResolverService(session)
    return await service.resolve_for_published_brief(
        project_id=project_id,
        published_brief_id=published_brief_id,
        published_url=published_url,
    )
