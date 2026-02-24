"""Routes for the external integration API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.integration.dependencies import require_integration_api_key
from app.api.integration.docs import (
    DONKEY_CLIENT_GUIDE_MARKDOWN,
    MODULAR_DOCUMENT_CONTRACT,
)
from app.api.integration.schemas import (
    IntegrationArticlePublicationPatchRequest,
    IntegrationArticlePublicationResponse,
    IntegrationArticleVersionResponse,
    IntegrationGuideResponse,
    IntegrationIndexResponse,
)
from app.api.v1.content.constants import (
    CONTENT_ARTICLE_NOT_FOUND_DETAIL,
    CONTENT_ARTICLE_VERSION_NOT_FOUND_DETAIL,
)
from app.core.database import get_session
from app.models.content import ContentArticle, ContentArticleVersion
from app.models.generated_dtos import ContentArticlePatchDTO
from app.integrations.content_image_store import ContentImageStore
from app.services.internal_link_resolver import (
    resolve_deferred_internal_links_for_published_article,
)
from app.services.publication_webhook import (
    cancel_pending_publication_webhook_deliveries,
)

public_router = APIRouter()
protected_router = APIRouter(dependencies=[Depends(require_integration_api_key)])
logger = logging.getLogger(__name__)


def _resolve_integration_path(request: Request, suffix: str) -> str:
    root_path = str(request.scope.get("root_path") or "")
    return f"{root_path}{suffix}"


async def _get_article_version(
    *,
    session: AsyncSession,
    project_id: str,
    article_id: str,
    version_number: int | None,
) -> tuple[ContentArticle, ContentArticleVersion]:
    article_result = await session.execute(
        select(ContentArticle).where(
            ContentArticle.id == article_id,
            ContentArticle.project_id == project_id,
        )
    )
    article = article_result.scalar_one_or_none()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_ARTICLE_NOT_FOUND_DETAIL,
        )

    resolved_version = version_number or article.current_version
    version_result = await session.execute(
        select(ContentArticleVersion).where(
            ContentArticleVersion.article_id == article_id,
            ContentArticleVersion.version_number == resolved_version,
        )
    )
    article_version = version_result.scalar_one_or_none()
    if not article_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_ARTICLE_VERSION_NOT_FOUND_DETAIL,
        )

    return article, article_version


def _serialize_article_version(
    article: ContentArticle,
    article_version: ContentArticleVersion,
) -> IntegrationArticleVersionResponse:
    modular_document = _enrich_modular_document_with_signed_featured_image(
        article_version.modular_document or {}
    )
    return IntegrationArticleVersionResponse(
        id=str(article_version.id),
        article_id=str(article_version.article_id),
        project_id=str(article.project_id),
        version_number=article_version.version_number,
        title=article_version.title,
        slug=article_version.slug,
        primary_keyword=article_version.primary_keyword,
        modular_document=modular_document,
        rendered_html=article_version.rendered_html,
        qa_report=article_version.qa_report,
        status=article_version.status,
        change_reason=article_version.change_reason,
        generation_model=article_version.generation_model,
        generation_temperature=article_version.generation_temperature,
        created_by_regeneration=article_version.created_by_regeneration,
        created_at=article_version.created_at,
        updated_at=article_version.updated_at,
    )


def _enrich_modular_document_with_signed_featured_image(
    modular_document: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(modular_document, dict):
        return {}

    featured_image = modular_document.get("featured_image")
    if not isinstance(featured_image, dict):
        return dict(modular_document)

    object_key = str(featured_image.get("object_key") or "").strip()
    if not object_key:
        return dict(modular_document)

    payload = dict(modular_document)
    enriched_featured_image = dict(featured_image)
    try:
        store = ContentImageStore()
        enriched_featured_image["signed_url"] = store.create_signed_read_url(object_key=object_key)
    except Exception as exc:
        logger.warning(
            "Failed to enrich featured image with signed URL",
            extra={"object_key": object_key, "error": str(exc)},
        )
    payload["featured_image"] = enriched_featured_image
    return payload


@public_router.get(
    "/",
    response_model=IntegrationIndexResponse,
    summary="Integration API index",
    description="Return discoverable docs and route templates for external integrations.",
)
async def integration_index(request: Request) -> IntegrationIndexResponse:
    """Expose integration docs and route templates."""
    return IntegrationIndexResponse(
        service="DonkeySEO Integration API",
        docs_path=_resolve_integration_path(request, "/docs"),
        openapi_path=_resolve_integration_path(request, "/openapi.json"),
        guide_path=_resolve_integration_path(request, "/guide/donkey-client"),
        guide_markdown_path=_resolve_integration_path(request, "/guide/donkey-client.md"),
        article_latest_path_template="/article/{article_id}?project_id={project_id}",
        article_version_path_template=(
            "/article/{article_id}/versions/{version_number}?project_id={project_id}"
        ),
        article_publication_patch_path_template=(
            "/article/{article_id}/publication?project_id={project_id}"
        ),
        auth_header="X-API-Key",
    )


@public_router.get(
    "/guide/donkey-client",
    response_model=IntegrationGuideResponse,
    summary="Donkey SEO client implementation guide",
    description="Return a thorough implementation guide for building Donkey SEO clients.",
)
async def integration_client_guide() -> IntegrationGuideResponse:
    """Return JSON guide payload for coding agents."""
    return IntegrationGuideResponse(
        title="Donkey SEO Client Implementation Guide",
        schema_version="1.0",
        markdown=DONKEY_CLIENT_GUIDE_MARKDOWN.strip(),
        modular_document_contract=MODULAR_DOCUMENT_CONTRACT,
    )


@public_router.get(
    "/guide/donkey-client.md",
    response_class=PlainTextResponse,
    summary="Donkey SEO client guide (markdown)",
    description="Return the same implementation guide as markdown text.",
)
async def integration_client_guide_markdown() -> str:
    """Return markdown version of coding-agent implementation guide."""
    return DONKEY_CLIENT_GUIDE_MARKDOWN.strip()


@protected_router.get(
    "/article/{article_id}",
    response_model=IntegrationArticleVersionResponse,
    summary="Get latest article version",
    description=(
        "Fetch the latest immutable article version from `content_article_versions` "
        "for an article/project pair."
    ),
)
async def get_latest_article_version(
    article_id: str,
    project_id: str = Query(..., min_length=1),
    version_number: int | None = Query(None, ge=1),
    session: AsyncSession = Depends(get_session),
) -> IntegrationArticleVersionResponse:
    """Fetch latest or explicit version snapshot for an article."""
    article, article_version = await _get_article_version(
        session=session,
        project_id=project_id,
        article_id=article_id,
        version_number=version_number,
    )
    return _serialize_article_version(article, article_version)


@protected_router.get(
    "/article/{article_id}/versions/{version_number}",
    response_model=IntegrationArticleVersionResponse,
    summary="Get explicit article version",
    description="Fetch a specific immutable article version by version number.",
)
async def get_specific_article_version(
    article_id: str,
    version_number: int,
    project_id: str = Query(..., min_length=1),
    session: AsyncSession = Depends(get_session),
) -> IntegrationArticleVersionResponse:
    """Fetch explicit immutable version snapshot for an article."""
    article, article_version = await _get_article_version(
        session=session,
        project_id=project_id,
        article_id=article_id,
        version_number=version_number,
    )
    return _serialize_article_version(article, article_version)


@protected_router.patch(
    "/article/{article_id}/publication",
    response_model=IntegrationArticlePublicationResponse,
    summary="Update article publication metadata",
    description=(
        "Persist publication status, publish timestamp, and live URL for an article/project pair."
    ),
)
async def update_article_publication(
    article_id: str,
    payload: IntegrationArticlePublicationPatchRequest,
    project_id: str = Query(..., min_length=1),
    session: AsyncSession = Depends(get_session),
) -> IntegrationArticlePublicationResponse:
    """Update mutable publication fields for a content article."""
    article_result = await session.execute(
        select(ContentArticle).where(
            ContentArticle.id == article_id,
            ContentArticle.project_id == project_id,
        )
    )
    article = article_result.scalar_one_or_none()
    if article is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_ARTICLE_NOT_FOUND_DETAIL,
        )

    update_data = payload.model_dump(exclude_unset=True)
    article.patch(session, ContentArticlePatchDTO.from_partial(update_data))
    if update_data.get("publish_status") == "published":
        await cancel_pending_publication_webhook_deliveries(
            session,
            article_id=str(article.id),
        )
    published_url = str(article.published_url or "").strip()
    if published_url:
        await resolve_deferred_internal_links_for_published_article(
            session,
            project_id=str(article.project_id),
            published_brief_id=str(article.brief_id),
            published_url=published_url,
        )

    await session.flush()
    await session.refresh(article)

    return IntegrationArticlePublicationResponse(
        article_id=str(article.id),
        project_id=str(article.project_id),
        publish_status=article.publish_status,
        published_at=article.published_at,
        published_url=article.published_url,
        updated_at=article.updated_at,
    )
