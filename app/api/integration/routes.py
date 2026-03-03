"""Routes for the external integration API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.integration.dependencies import require_integration_api_key
from app.api.integration.docs import (
    DONKEY_CLIENT_GUIDE_MARKDOWN,
    INTEGRATION_CLIENT_ENV_TEMPLATE,
    INTEGRATION_CLIENT_ENV_VARS,
    MODULAR_BLOCK_TYPE_REFERENCE,
    MODULAR_DOCUMENT_CONTRACT,
    MODULAR_DOCUMENT_FIELD_REFERENCE,
    PUBLICATION_WEBHOOK_CONTRACT,
)
from app.api.integration.schemas import (
    IntegrationArticleListResponse,
    IntegrationArticlePublicationPatchRequest,
    IntegrationArticlePublicationResponse,
    IntegrationPillarListResponse,
    IntegrationPillarResponse,
    IntegrationPillarReference,
    IntegrationArticleSummaryResponse,
    IntegrationArticleVersionResponse,
    IntegrationGuideResponse,
    IntegrationIndexResponse,
)
from app.api.v1.content.constants import (
    CONTENT_ARTICLE_NOT_FOUND_DETAIL,
    CONTENT_ARTICLE_VERSION_NOT_FOUND_DETAIL,
)
from app.config import settings
from app.core.database import get_session
from app.core.redis import get_redis_client
from app.integrations.content_image_store import ContentImageStore
from app.models.content import ContentArticle, ContentArticleVersion
from app.models.content_pillar import ContentBriefPillarAssignment, ContentPillar
from app.models.generated_dtos import ContentArticlePatchDTO
from app.services.author_profiles import (
    enrich_modular_document_with_signed_author_image,
)
from app.services.internal_link_resolver import (
    resolve_deferred_internal_links_for_published_article,
)
from app.services.publication_webhook import (
    cancel_pending_publication_webhook_deliveries,
)

public_router = APIRouter()
protected_router = APIRouter(dependencies=[Depends(require_integration_api_key)])
logger = logging.getLogger(__name__)
INTEGRATION_ARTICLE_CACHE_PREFIX = "integration:article-version"
INTEGRATION_ARTICLE_CACHE_NAMESPACE = "v1"
INTEGRATION_ARTICLE_VERSION_CACHE_TTL_SECONDS = max(1, int(settings.cache_ttl_seconds))
INTEGRATION_ARTICLE_LATEST_CACHE_TTL_SECONDS = max(
    30,
    min(INTEGRATION_ARTICLE_VERSION_CACHE_TTL_SECONDS, 300),
)


def _resolve_integration_path(request: Request, suffix: str) -> str:
    root_path = str(request.scope.get("root_path") or "")
    return f"{root_path}{suffix}"


def _integration_article_cache_key(
    *,
    project_id: str,
    article_id: str,
    version_token: str,
) -> str:
    return (
        f"{INTEGRATION_ARTICLE_CACHE_PREFIX}:{INTEGRATION_ARTICLE_CACHE_NAMESPACE}:"
        f"{project_id}:{article_id}:{version_token}"
    )


async def _read_cached_integration_article_version(
    *,
    project_id: str,
    article_id: str,
    version_token: str,
) -> IntegrationArticleVersionResponse | None:
    cache_key = _integration_article_cache_key(
        project_id=project_id,
        article_id=article_id,
        version_token=version_token,
    )
    try:
        redis = get_redis_client()
        raw_payload = await redis.get(cache_key)
    except Exception:
        logger.exception(
            "Failed to read integration article cache payload",
            extra={"cache_key": cache_key},
        )
        return None
    if not isinstance(raw_payload, str) or not raw_payload:
        return None
    try:
        return IntegrationArticleVersionResponse.model_validate_json(raw_payload)
    except Exception:
        logger.warning(
            "Invalid integration article cache payload",
            extra={"cache_key": cache_key},
        )
        return None


async def _write_cached_integration_article_version(
    *,
    project_id: str,
    article_id: str,
    version_token: str,
    payload: IntegrationArticleVersionResponse,
    ttl_seconds: int,
) -> None:
    cache_key = _integration_article_cache_key(
        project_id=project_id,
        article_id=article_id,
        version_token=version_token,
    )
    try:
        redis = get_redis_client()
        await redis.set(
            cache_key,
            payload.model_dump_json(),
            ex=max(1, int(ttl_seconds)),
        )
    except Exception:
        logger.exception(
            "Failed to write integration article cache payload",
            extra={"cache_key": cache_key},
        )


async def _invalidate_cached_integration_article_versions(
    *,
    project_id: str,
    article_id: str,
) -> None:
    cache_pattern = _integration_article_cache_key(
        project_id=project_id,
        article_id=article_id,
        version_token="*",
    )
    try:
        redis = get_redis_client()
        keys = [key async for key in redis.scan_iter(match=cache_pattern)]
        if keys:
            await redis.delete(*keys)
    except Exception:
        logger.exception(
            "Failed to invalidate integration article cache payloads",
            extra={"cache_pattern": cache_pattern},
        )


async def _load_integration_pillar_payloads(
    *,
    session: AsyncSession,
    project_id: str,
    brief_ids: list[str],
) -> dict[str, dict]:
    if not brief_ids:
        return {}

    result = await session.execute(
        select(ContentBriefPillarAssignment, ContentPillar)
        .join(ContentPillar, ContentPillar.id == ContentBriefPillarAssignment.pillar_id)
        .where(
            ContentBriefPillarAssignment.project_id == project_id,
            ContentBriefPillarAssignment.brief_id.in_(brief_ids),
            ContentPillar.status == "active",
        )
        .order_by(
            ContentBriefPillarAssignment.brief_id.asc(),
            ContentBriefPillarAssignment.relationship_type.asc(),
            ContentBriefPillarAssignment.created_at.asc(),
        )
    )

    payload: dict[str, dict] = {}
    for assignment, pillar in result.all():
        brief_id = str(assignment.brief_id)
        entry = payload.setdefault(
            brief_id,
            {"primary": None, "secondary": [], "confidence": None, "secondary_ids": set()},
        )
        ref = IntegrationPillarReference(
            id=str(pillar.id),
            name=pillar.name,
            slug=pillar.slug,
        )
        relationship_type = str(assignment.relationship_type or "").strip().lower()
        if relationship_type == "primary" and entry["primary"] is None:
            entry["primary"] = ref
            entry["confidence"] = assignment.confidence_score
            continue
        if str(pillar.id) in entry["secondary_ids"]:
            continue
        entry["secondary"].append(ref)
        entry["secondary_ids"].add(str(pillar.id))

    for entry in payload.values():
        entry.pop("secondary_ids", None)
    return payload


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
    payload = dict(modular_document)
    featured_image = payload.get("featured_image")
    if isinstance(featured_image, dict):
        object_key = str(featured_image.get("object_key") or "").strip()
        if object_key:
            enriched_featured_image = dict(featured_image)
            try:
                store = ContentImageStore()
                enriched_featured_image["signed_url"] = store.create_signed_read_url(
                    object_key=object_key
                )
            except Exception as exc:
                logger.warning(
                    "Failed to enrich featured image with signed URL",
                    extra={"object_key": object_key, "error": str(exc)},
                )
            payload["featured_image"] = enriched_featured_image

    return enrich_modular_document_with_signed_author_image(payload)


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
        article_list_path_template=(
            "/articles?project_id={project_id}&page={page}&page_size={page_size}"
            "&pillar_slug={pillar_slug_optional}"
        ),
        pillar_list_path_template=(
            "/pillars?project_id={project_id}&include_archived={include_archived_optional}"
        ),
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
        modular_document_field_reference=MODULAR_DOCUMENT_FIELD_REFERENCE,
        block_type_reference=MODULAR_BLOCK_TYPE_REFERENCE,
        webhook_contract=PUBLICATION_WEBHOOK_CONTRACT,
        client_env_vars=INTEGRATION_CLIENT_ENV_VARS,
        client_env_template=INTEGRATION_CLIENT_ENV_TEMPLATE.strip(),
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
    "/articles",
    response_model=IntegrationArticleListResponse,
    summary="List project articles",
    description=(
        "Return a paginated, lightweight article list for a project. "
        "This endpoint excludes heavy content fields like modular_document and rendered_html."
    ),
)
async def list_project_articles(
    project_id: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    pillar_slug: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
) -> IntegrationArticleListResponse:
    """List articles with lightweight metadata for integration clients."""
    page_size = int(page_size)
    page = int(page)
    offset = (page - 1) * page_size

    article_query = select(ContentArticle).where(ContentArticle.project_id == project_id)
    if pillar_slug:
        article_query = (
            article_query.join(
                ContentBriefPillarAssignment,
                ContentBriefPillarAssignment.brief_id == ContentArticle.brief_id,
            )
            .join(ContentPillar, ContentPillar.id == ContentBriefPillarAssignment.pillar_id)
            .where(
                ContentBriefPillarAssignment.project_id == project_id,
                ContentPillar.slug == pillar_slug,
                ContentPillar.status == "active",
            )
            .distinct()
        )

    total_result = await session.execute(select(func.count()).select_from(article_query.subquery()))
    total = int(total_result.scalar_one() or 0)

    result = await session.execute(
        article_query.order_by(ContentArticle.updated_at.desc(), ContentArticle.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    articles = list(result.scalars().all())
    pillar_payloads = await _load_integration_pillar_payloads(
        session=session,
        project_id=project_id,
        brief_ids=[str(article.brief_id) for article in articles],
    )

    items = [
        IntegrationArticleSummaryResponse(
            id=str(article.id),
            project_id=str(article.project_id),
            brief_id=str(article.brief_id),
            title=article.title,
            slug=article.slug,
            primary_keyword=article.primary_keyword,
            current_version=article.current_version,
            status=article.status,
            publish_status=article.publish_status,
            published_at=article.published_at,
            published_url=article.published_url,
            primary_pillar=(
                pillar_payloads.get(str(article.brief_id), {}).get("primary")
                if isinstance(pillar_payloads.get(str(article.brief_id)), dict)
                else None
            ),
            secondary_pillars=(
                pillar_payloads.get(str(article.brief_id), {}).get("secondary", [])
                if isinstance(pillar_payloads.get(str(article.brief_id)), dict)
                else []
            ),
            pillar_assignment_confidence=(
                pillar_payloads.get(str(article.brief_id), {}).get("confidence")
                if isinstance(pillar_payloads.get(str(article.brief_id)), dict)
                else None
            ),
            created_at=article.created_at,
            updated_at=article.updated_at,
        )
        for article in articles
    ]

    return IntegrationArticleListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@protected_router.get(
    "/pillars",
    response_model=IntegrationPillarListResponse,
    summary="List project pillars",
    description=(
        "Return all project pillars with assignment/article counts. "
        "Use this for frontend category navigation."
    ),
)
async def list_project_pillars(
    project_id: str = Query(..., min_length=1),
    include_archived: bool = Query(False),
    session: AsyncSession = Depends(get_session),
) -> IntegrationPillarListResponse:
    """List project content pillars for integration clients."""
    query = select(ContentPillar).where(ContentPillar.project_id == project_id)
    if not include_archived:
        query = query.where(ContentPillar.status == "active")
    query = query.order_by(ContentPillar.name.asc(), ContentPillar.created_at.asc())

    result = await session.execute(query)
    pillars = list(result.scalars().all())
    if not pillars:
        return IntegrationPillarListResponse(items=[], total=0)

    pillar_ids = [str(pillar.id) for pillar in pillars]

    assignment_result = await session.execute(
        select(
            ContentBriefPillarAssignment.pillar_id,
            ContentBriefPillarAssignment.relationship_type,
            func.count(ContentBriefPillarAssignment.id),
        )
        .where(
            ContentBriefPillarAssignment.project_id == project_id,
            ContentBriefPillarAssignment.pillar_id.in_(pillar_ids),
        )
        .group_by(
            ContentBriefPillarAssignment.pillar_id,
            ContentBriefPillarAssignment.relationship_type,
        )
    )
    assignment_counts: dict[str, dict[str, int]] = {
        pillar_id: {"primary": 0, "secondary": 0} for pillar_id in pillar_ids
    }
    for pillar_id, relationship_type, count in assignment_result.all():
        role = str(relationship_type or "").strip().lower()
        if role not in {"primary", "secondary"}:
            continue
        assignment_counts[str(pillar_id)][role] = int(count or 0)

    article_result = await session.execute(
        select(
            ContentBriefPillarAssignment.pillar_id,
            func.count(ContentArticle.id),
            func.count(ContentArticle.id).filter(
                or_(
                    ContentArticle.publish_status == "published",
                    ContentArticle.published_at.is_not(None),
                )
            ),
        )
        .join(
            ContentArticle,
            and_(
                ContentArticle.project_id == ContentBriefPillarAssignment.project_id,
                ContentArticle.brief_id == ContentBriefPillarAssignment.brief_id,
            ),
        )
        .where(
            ContentBriefPillarAssignment.project_id == project_id,
            ContentBriefPillarAssignment.relationship_type == "primary",
            ContentBriefPillarAssignment.pillar_id.in_(pillar_ids),
        )
        .group_by(ContentBriefPillarAssignment.pillar_id)
    )
    article_counts: dict[str, tuple[int, int]] = {
        str(pillar_id): (int(article_count or 0), int(published_count or 0))
        for pillar_id, article_count, published_count in article_result.all()
    }

    items = []
    for pillar in pillars:
        pillar_id = str(pillar.id)
        role_counts = assignment_counts.get(pillar_id, {"primary": 0, "secondary": 0})
        primary_count = int(role_counts.get("primary", 0))
        secondary_count = int(role_counts.get("secondary", 0))
        article_count, published_count = article_counts.get(pillar_id, (0, 0))
        items.append(
            IntegrationPillarResponse(
                id=pillar_id,
                project_id=str(pillar.project_id),
                name=pillar.name,
                slug=pillar.slug,
                description=pillar.description,
                status=pillar.status,
                source=pillar.source,
                locked=bool(pillar.locked),
                primary_brief_count=primary_count,
                secondary_brief_count=secondary_count,
                total_brief_count=primary_count + secondary_count,
                primary_article_count=article_count,
                published_primary_article_count=published_count,
                created_at=pillar.created_at,
                updated_at=pillar.updated_at,
            )
        )

    return IntegrationPillarListResponse(
        items=items,
        total=len(items),
    )


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
    version_token = (
        f"version:{int(version_number)}" if version_number is not None else "latest"
    )
    cached_payload = await _read_cached_integration_article_version(
        project_id=project_id,
        article_id=article_id,
        version_token=version_token,
    )
    if cached_payload is not None:
        return cached_payload

    article, article_version = await _get_article_version(
        session=session,
        project_id=project_id,
        article_id=article_id,
        version_number=version_number,
    )
    payload = _serialize_article_version(article, article_version)

    if version_number is None:
        await _write_cached_integration_article_version(
            project_id=project_id,
            article_id=article_id,
            version_token="latest",
            payload=payload,
            ttl_seconds=INTEGRATION_ARTICLE_LATEST_CACHE_TTL_SECONDS,
        )
        await _write_cached_integration_article_version(
            project_id=project_id,
            article_id=article_id,
            version_token=f"version:{int(article_version.version_number)}",
            payload=payload,
            ttl_seconds=INTEGRATION_ARTICLE_VERSION_CACHE_TTL_SECONDS,
        )
    else:
        await _write_cached_integration_article_version(
            project_id=project_id,
            article_id=article_id,
            version_token=version_token,
            payload=payload,
            ttl_seconds=INTEGRATION_ARTICLE_VERSION_CACHE_TTL_SECONDS,
        )
    return payload


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
    version_token = f"version:{int(version_number)}"
    cached_payload = await _read_cached_integration_article_version(
        project_id=project_id,
        article_id=article_id,
        version_token=version_token,
    )
    if cached_payload is not None:
        return cached_payload

    article, article_version = await _get_article_version(
        session=session,
        project_id=project_id,
        article_id=article_id,
        version_number=version_number,
    )
    payload = _serialize_article_version(article, article_version)
    await _write_cached_integration_article_version(
        project_id=project_id,
        article_id=article_id,
        version_token=version_token,
        payload=payload,
        ttl_seconds=INTEGRATION_ARTICLE_VERSION_CACHE_TTL_SECONDS,
    )
    return payload


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
    await _invalidate_cached_integration_article_versions(
        project_id=str(article.project_id),
        article_id=str(article.id),
    )

    return IntegrationArticlePublicationResponse(
        article_id=str(article.id),
        project_id=str(article.project_id),
        publish_status=article.publish_status,
        published_at=article.published_at,
        published_url=article.published_url,
        updated_at=article.updated_at,
    )
