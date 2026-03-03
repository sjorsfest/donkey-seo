"""Content briefs API endpoints."""

import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select

from app.api.v1.content.constants import (
    CONTENT_ARTICLE_NOT_FOUND_DETAIL,
    CONTENT_ARTICLE_VERSION_NOT_FOUND_DETAIL,
    CONTENT_BRIEF_NOT_FOUND_DETAIL,
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)
from app.api.v1.dependencies import get_user_project
from app.dependencies import CurrentUser, DbSession
from app.models.author import Author
from app.models.brand import BrandProfile
from app.models.content import (
    ContentArticle,
    ContentArticleVersion,
    ContentBrief,
    ContentFeaturedImage,
    WriterInstructions,
)
from app.models.content_pillar import ContentBriefPillarAssignment, ContentPillar
from app.models.generated_dtos import (
    ContentArticlePatchDTO,
    ContentArticleVersionCreateDTO,
    ContentBriefCreateDTO,
    ContentBriefPatchDTO,
)
from app.models.style_guide import BriefDelta
from app.schemas.content import (
    ContentArticleDetailResponse,
    ContentArticleListResponse,
    ContentArticlePublishNowResponse,
    ContentArticleResponse,
    ContentArticleVersionResponse,
    ContentBriefCreate,
    ContentBriefDetailResponse,
    ContentBriefListResponse,
    ContentBriefResponse,
    ContentBriefUpdate,
    ContentCalendarItemResponse,
    ContentCalendarItemState,
    ContentCalendarResponse,
    ContentPillarListResponse,
    ContentPillarReference,
    ContentPillarResponse,
    RegenerateArticleRequest,
    WriterInstructionsResponse,
)
from app.services.article_generation import ArticleGenerationService
from app.services.author_profiles import (
    author_modular_document_payload,
    choose_random_author,
)
from app.services.content_keyword_tracking import (
    analyze_keyword_usage,
    persist_article_keyword_usages,
    sync_brief_keywords,
    with_keyword_coverage_report,
)
from app.services.content_renderer import render_modular_document
from app.services.featured_image_generation import (
    FeaturedImageGenerationService,
    generation_retry_settings,
    modular_featured_image_payload,
    retry_with_backoff,
)
from app.services.publication_webhook import (
    dispatch_publication_webhook_delivery,
    schedule_publication_webhook_for_article,
    reschedule_publication_webhook_for_brief,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _brief_payload(brief: ContentBrief, *, locked_title: str | None = None) -> dict:
    working_titles = brief.working_titles or []
    if locked_title:
        working_titles = [locked_title]
    return {
        "id": str(brief.id),
        "primary_keyword": brief.primary_keyword,
        "search_intent": brief.search_intent,
        "page_type": brief.page_type,
        "funnel_stage": brief.funnel_stage,
        "working_titles": working_titles,
        "locked_title": locked_title or "",
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


def _enforce_locked_title(document: dict, locked_title: str) -> None:
    seo_meta = document.get("seo_meta")
    if not isinstance(seo_meta, dict):
        seo_meta = {}
        document["seo_meta"] = seo_meta
    seo_meta["h1"] = locked_title

    blocks = document.get("blocks")
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("block_type") or "").strip().lower() != "hero":
            continue
        block["heading"] = locked_title
        break


def _writer_instructions_payload(instructions: WriterInstructions | None) -> dict:
    if not instructions:
        return {}
    return {
        "voice_tone_constraints": instructions.voice_tone_constraints or {},
        "forbidden_claims": instructions.forbidden_claims or [],
        "compliance_notes": instructions.compliance_notes or [],
        "formatting_requirements": instructions.formatting_requirements or {},
        "h1_h2_usage": instructions.h1_h2_usage or {},
        "internal_linking_minimums": instructions.internal_linking_minimums or {},
        "schema_guidance": instructions.schema_guidance or "",
        "qa_checklist": instructions.qa_checklist or [],
        "pass_fail_thresholds": instructions.pass_fail_thresholds or {},
        "common_failure_modes": instructions.common_failure_modes or [],
    }


def _brief_delta_payload(delta: BriefDelta | None) -> dict:
    if not delta:
        return {}
    return {
        "page_type_rules": delta.page_type_rules or {},
        "must_include_sections": delta.must_include_sections or [],
        "h1_h2_usage": delta.h1_h2_usage or {},
        "schema_type": delta.schema_type or "Article",
        "additional_qa_items": delta.additional_qa_items or [],
    }


def _build_brand_context(brand: BrandProfile | None) -> str:
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
            if product.get("name"):
                parts.append(f"Product: {product['name']}")
            if product.get("description"):
                parts.append(f"Description: {product['description']}")
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


async def _load_project_authors(session: DbSession, project_id: str) -> list[Author]:
    result = await session.execute(
        select(Author)
        .where(Author.project_id == project_id)
        .order_by(Author.created_at.asc())
    )
    return list(result.scalars())


async def _resolve_author_for_regeneration(
    *,
    session: DbSession,
    project_id: str,
    existing_author_id: str | None,
) -> Author | None:
    if existing_author_id:
        existing_result = await session.execute(
            select(Author).where(
                Author.id == existing_author_id,
                Author.project_id == project_id,
            )
        )
        existing_author = existing_result.scalar_one_or_none()
        if existing_author is not None:
            return existing_author

    project_authors = await _load_project_authors(session, project_id)
    return choose_random_author(project_authors)


def _resolve_calendar_state(
    *,
    has_writer_instructions: bool,
    article: ContentArticle | None,
    publication_date: date | None,
    today: date | None = None,
) -> ContentCalendarItemState:
    reference_today = today or date.today()
    if article and (
        article.publish_status == "published"
        or article.published_at is not None
        or article.status == "published"
    ):
        return "published"
    if article:
        if publication_date is not None and publication_date < reference_today:
            return "publish_pending"
        return "article_ready"
    if has_writer_instructions:
        return "writer_instructions_ready"
    return "brief_ready"


def _publish_now_date() -> date:
    """Resolve publication date for immediate publish requests."""
    return date.today()


async def _load_pillar_payloads_for_briefs(
    *,
    session: DbSession,
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
        ref = ContentPillarReference(
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


def _article_response(
    article: ContentArticle,
    *,
    pillar_payload: dict | None,
) -> ContentArticleResponse:
    payload = {
        "id": str(article.id),
        "project_id": str(article.project_id),
        "brief_id": str(article.brief_id),
        "author_id": str(article.author_id) if article.author_id is not None else None,
        "title": article.title,
        "slug": article.slug,
        "primary_keyword": article.primary_keyword,
        "status": article.status,
        "publish_status": article.publish_status,
        "published_at": article.published_at,
        "published_url": article.published_url,
        "current_version": article.current_version,
        "generation_model": article.generation_model,
        "generated_at": article.generated_at,
        "created_at": article.created_at,
        "updated_at": article.updated_at,
        "primary_pillar": pillar_payload.get("primary") if isinstance(pillar_payload, dict) else None,
        "secondary_pillars": (
            pillar_payload.get("secondary")
            if isinstance(pillar_payload, dict)
            else []
        ),
        "pillar_assignment_confidence": (
            pillar_payload.get("confidence")
            if isinstance(pillar_payload, dict)
            else None
        ),
    }
    return ContentArticleResponse.model_validate(payload)


def _article_detail_response(
    article: ContentArticle,
    *,
    pillar_payload: dict | None,
) -> ContentArticleDetailResponse:
    base = _article_response(article, pillar_payload=pillar_payload).model_dump()
    base.update(
        {
            "modular_document": article.modular_document,
            "rendered_html": article.rendered_html,
            "qa_report": article.qa_report,
        }
    )
    return ContentArticleDetailResponse.model_validate(base)


@router.get(
    "/{project_id}/briefs",
    response_model=ContentBriefListResponse,
    summary="List content briefs",
    description=(
        "Return paginated content briefs for a project with optional status filtering."
    ),
)
async def list_briefs(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(DEFAULT_PAGE, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    status_filter: str | None = Query(None, alias="status"),
) -> ContentBriefListResponse:
    """List content briefs for a project."""
    await get_user_project(project_id, current_user, session)

    query = select(ContentBrief).where(ContentBrief.project_id == project_id)

    if status_filter:
        query = query.where(ContentBrief.status == status_filter)

    count_query = select(func.count()).select_from(query.subquery())
    total = await session.scalar(count_query) or 0

    offset = (page - 1) * page_size
    query = query.order_by(ContentBrief.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    briefs = result.scalars().all()

    return ContentBriefListResponse(
        items=[ContentBriefResponse.model_validate(brief) for brief in briefs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{project_id}/calendar",
    response_model=ContentCalendarResponse,
    summary="Get content calendar",
    description=(
        "Return scheduled content items keyed by proposed publication date with "
        "current production/publish state."
    ),
)
async def get_content_calendar(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
    date_from: date | None = Query(None, description="Inclusive lower date bound (YYYY-MM-DD)."),
    date_to: date | None = Query(None, description="Inclusive upper date bound (YYYY-MM-DD)."),
) -> ContentCalendarResponse:
    """Return content calendar entries for scheduled briefs."""
    await get_user_project(project_id, current_user, session)

    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="date_from must be earlier than or equal to date_to",
        )

    query = (
        select(ContentBrief, WriterInstructions.id, ContentArticle)
        .outerjoin(WriterInstructions, WriterInstructions.brief_id == ContentBrief.id)
        .outerjoin(
            ContentArticle,
            and_(
                ContentArticle.brief_id == ContentBrief.id,
                ContentArticle.project_id == ContentBrief.project_id,
            ),
        )
        .where(
            ContentBrief.project_id == project_id,
            ContentBrief.proposed_publication_date.is_not(None),
        )
    )
    if date_from is not None:
        query = query.where(ContentBrief.proposed_publication_date >= date_from)
    if date_to is not None:
        query = query.where(ContentBrief.proposed_publication_date <= date_to)

    query = query.order_by(
        ContentBrief.proposed_publication_date.asc(),
        ContentBrief.created_at.asc(),
    )
    result = await session.execute(query)

    items: list[ContentCalendarItemResponse] = []
    for brief, writer_instruction_id, article in result.all():
        has_writer_instructions = writer_instruction_id is not None
        calendar_state = _resolve_calendar_state(
            has_writer_instructions=has_writer_instructions,
            article=article,
            publication_date=brief.proposed_publication_date,
        )
        working_title = (brief.working_titles or [None])[0]
        article_id = str(article.id) if article else None
        items.append(
            ContentCalendarItemResponse(
                date=brief.proposed_publication_date,
                brief_id=str(brief.id),
                topic_id=str(brief.topic_id),
                primary_keyword=brief.primary_keyword,
                working_title=str(working_title) if working_title is not None else None,
                brief_status=brief.status,
                has_writer_instructions=has_writer_instructions,
                article_id=article_id,
                article_title=article.title if article else None,
                article_slug=article.slug if article else None,
                article_status=article.status if article else None,
                article_current_version=article.current_version if article else None,
                publish_status=article.publish_status if article else None,
                published_at=article.published_at if article else None,
                published_url=article.published_url if article else None,
                calendar_state=calendar_state,
            )
        )

    return ContentCalendarResponse(items=items)


@router.get(
    "/{project_id}/briefs/{brief_id}",
    response_model=ContentBriefDetailResponse,
    summary="Get content brief",
    description="Return detailed information for a specific content brief.",
)
async def get_brief(
    project_id: str,
    brief_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentBrief:
    """Get detailed content brief."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(ContentBrief).where(
            ContentBrief.id == brief_id,
            ContentBrief.project_id == project_id,
        )
    )
    brief = result.scalar_one_or_none()

    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_BRIEF_NOT_FOUND_DETAIL,
        )

    return brief


@router.post(
    "/{project_id}/briefs",
    response_model=ContentBriefResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create content brief",
    description="Create a new content brief for a project topic and primary keyword.",
)
async def create_brief(
    project_id: str,
    brief_data: ContentBriefCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentBrief:
    """Create a new content brief."""
    await get_user_project(project_id, current_user, session)

    brief = ContentBrief.create(
        session,
        ContentBriefCreateDTO(
            project_id=str(project_id),
            topic_id=brief_data.topic_id,
            primary_keyword=brief_data.primary_keyword,
            working_titles=brief_data.working_titles,
            target_word_count_min=brief_data.target_word_count_min,
            target_word_count_max=brief_data.target_word_count_max,
            proposed_publication_date=brief_data.proposed_publication_date,
        ),
    )
    await sync_brief_keywords(
        session,
        brief=brief,
        primary_keyword=brief_data.primary_keyword,
        supporting_keywords=[],
    )
    await session.flush()
    await session.refresh(brief)

    logger.info(
        "Content brief created",
        extra={
            "project_id": str(project_id),
            "brief_id": str(brief.id),
            "keyword": brief.primary_keyword,
        },
    )

    return brief


@router.put(
    "/{project_id}/briefs/{brief_id}",
    response_model=ContentBriefResponse,
    summary="Update content brief",
    description="Apply partial updates to an existing content brief.",
)
async def update_brief(
    project_id: str,
    brief_id: str,
    brief_data: ContentBriefUpdate,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentBrief:
    """Update a content brief."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(ContentBrief).where(
            ContentBrief.id == brief_id,
            ContentBrief.project_id == project_id,
        )
    )
    brief = result.scalar_one_or_none()

    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_BRIEF_NOT_FOUND_DETAIL,
        )

    update_data = brief_data.model_dump(exclude_unset=True)
    if "primary_keyword" in update_data and not str(update_data.get("primary_keyword") or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="primary_keyword cannot be empty",
        )
    publication_date_updated = "proposed_publication_date" in update_data
    brief.patch(
        session,
        ContentBriefPatchDTO.from_partial(update_data),
    )
    await sync_brief_keywords(
        session,
        brief=brief,
        primary_keyword=str(update_data.get("primary_keyword") or brief.primary_keyword or ""),
        supporting_keywords=(
            update_data.get("supporting_keywords")
            if isinstance(update_data.get("supporting_keywords"), list)
            else ([] if "supporting_keywords" in update_data else (brief.supporting_keywords or []))
        ),
    )

    if publication_date_updated:
        await reschedule_publication_webhook_for_brief(
            session,
            project_id=project_id,
            brief_id=brief_id,
            proposed_publication_date=brief.proposed_publication_date,
        )

    await session.flush()
    await session.refresh(brief)

    return brief


@router.delete(
    "/{project_id}/briefs/{brief_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete content brief",
    description="Delete a content brief from the project.",
)
async def delete_brief(
    project_id: str,
    brief_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    """Delete a content brief."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(ContentBrief).where(
            ContentBrief.id == brief_id,
            ContentBrief.project_id == project_id,
        )
    )
    brief = result.scalar_one_or_none()

    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_BRIEF_NOT_FOUND_DETAIL,
        )

    await brief.delete(session)


@router.get(
    "/{project_id}/briefs/{brief_id}/instructions",
    response_model=WriterInstructionsResponse | None,
    summary="Get writer instructions",
    description=(
        "Return the generated writer instructions attached to a content brief, if available."
    ),
)
async def get_writer_instructions(
    project_id: str,
    brief_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> WriterInstructions | None:
    """Get writer instructions for a brief."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(ContentBrief).where(
            ContentBrief.id == brief_id,
            ContentBrief.project_id == project_id,
        )
    )
    brief = result.scalar_one_or_none()

    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_BRIEF_NOT_FOUND_DETAIL,
        )

    result = await session.execute(
        select(WriterInstructions).where(WriterInstructions.brief_id == brief_id)
    )

    return result.scalar_one_or_none()


@router.get(
    "/{project_id}/articles",
    response_model=ContentArticleListResponse,
    summary="List generated articles",
    description="Return paginated generated article artifacts for a project.",
)
async def list_articles(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(DEFAULT_PAGE, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    status_filter: str | None = Query(None, alias="status"),
    pillar_slug: str | None = Query(None),
) -> ContentArticleListResponse:
    """List generated content articles for a project."""
    await get_user_project(project_id, current_user, session)

    query = select(ContentArticle).where(ContentArticle.project_id == project_id)
    if status_filter:
        query = query.where(ContentArticle.status == status_filter)
    if pillar_slug:
        query = (
            query.join(
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

    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    offset = (page - 1) * page_size
    result = await session.execute(
        query.order_by(ContentArticle.created_at.desc()).offset(offset).limit(page_size)
    )
    items = result.scalars().all()
    pillar_payloads = await _load_pillar_payloads_for_briefs(
        session=session,
        project_id=project_id,
        brief_ids=[str(item.brief_id) for item in items],
    )

    return ContentArticleListResponse(
        items=[
            _article_response(item, pillar_payload=pillar_payloads.get(str(item.brief_id)))
            for item in items
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/{project_id}/articles/{article_id}/publish-now",
    response_model=ContentArticlePublishNowResponse,
    summary="Publish article now",
    description=(
        "Set the article's brief publication date to the current call-day and "
        "dispatch its publication webhook immediately."
    ),
)
async def publish_article_now(
    project_id: str,
    article_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentArticlePublishNowResponse:
    """Set publication date to today and trigger immediate webhook dispatch."""
    await get_user_project(project_id, current_user, session)

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

    brief_result = await session.execute(
        select(ContentBrief).where(
            ContentBrief.id == article.brief_id,
            ContentBrief.project_id == project_id,
        )
    )
    brief = brief_result.scalar_one_or_none()
    if brief is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_BRIEF_NOT_FOUND_DETAIL,
        )

    if (
        article.publish_status == "published"
        or article.published_at is not None
        or article.status == "published"
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Article is already published",
        )

    publication_date = _publish_now_date()
    if brief.proposed_publication_date != publication_date:
        brief.patch(
            session,
            ContentBriefPatchDTO.from_partial(
                {"proposed_publication_date": publication_date}
            ),
        )

    delivery = await schedule_publication_webhook_for_article(
        session,
        article=article,
        proposed_publication_date=publication_date,
    )
    await session.flush()

    webhook_dispatch_triggered = False
    if delivery is not None:
        webhook_dispatch_triggered = await dispatch_publication_webhook_delivery(
            session,
            delivery_id=str(delivery.id),
        )
        await session.refresh(delivery)

    return ContentArticlePublishNowResponse(
        article_id=str(article.id),
        brief_id=str(article.brief_id),
        project_id=str(article.project_id),
        proposed_publication_date=publication_date,
        webhook_delivery_id=str(delivery.id) if delivery is not None else None,
        webhook_dispatch_triggered=webhook_dispatch_triggered,
        webhook_delivery_status=delivery.status if delivery is not None else None,
        webhook_attempt_count=delivery.attempt_count if delivery is not None else None,
        webhook_last_http_status=delivery.last_http_status if delivery is not None else None,
        webhook_last_error=delivery.last_error if delivery is not None else None,
    )


@router.get(
    "/{project_id}/briefs/{brief_id}/article",
    response_model=ContentArticleDetailResponse,
    summary="Get article by brief",
    description="Return the canonical generated article for a brief.",
)
async def get_article_for_brief(
    project_id: str,
    brief_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentArticleDetailResponse:
    """Get canonical generated article for a brief."""
    await get_user_project(project_id, current_user, session)

    article_result = await session.execute(
        select(ContentArticle).where(
            ContentArticle.project_id == project_id,
            ContentArticle.brief_id == brief_id,
        )
    )
    article = article_result.scalar_one_or_none()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_ARTICLE_NOT_FOUND_DETAIL,
        )

    pillar_payloads = await _load_pillar_payloads_for_briefs(
        session=session,
        project_id=project_id,
        brief_ids=[str(article.brief_id)],
    )
    return _article_detail_response(
        article,
        pillar_payload=pillar_payloads.get(str(article.brief_id)),
    )


@router.post(
    "/{project_id}/briefs/{brief_id}/article/regenerate",
    response_model=ContentArticleDetailResponse,
    summary="Regenerate article",
    description="Regenerate article content for a brief and increment article version.",
)
async def regenerate_article(
    project_id: str,
    brief_id: str,
    payload: RegenerateArticleRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentArticleDetailResponse:
    """Regenerate canonical article and store an immutable version snapshot."""
    project = await get_user_project(project_id, current_user, session)

    brief_result = await session.execute(
        select(ContentBrief).where(
            ContentBrief.id == brief_id,
            ContentBrief.project_id == project_id,
        )
    )
    brief = brief_result.scalar_one_or_none()
    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_BRIEF_NOT_FOUND_DETAIL,
        )

    article_result = await session.execute(
        select(ContentArticle).where(
            ContentArticle.project_id == project_id,
            ContentArticle.brief_id == brief_id,
        )
    )
    article = article_result.scalar_one_or_none()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_ARTICLE_NOT_FOUND_DETAIL,
        )

    instructions_result = await session.execute(
        select(WriterInstructions).where(WriterInstructions.brief_id == brief_id)
    )
    instructions = instructions_result.scalar_one_or_none()

    if not instructions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Writer instructions are required to regenerate article content",
        )

    delta_result = await session.execute(select(BriefDelta).where(BriefDelta.brief_id == brief_id))
    delta = delta_result.scalar_one_or_none()

    brand_result = await session.execute(
        select(BrandProfile).where(BrandProfile.project_id == project_id)
    )
    brand = brand_result.scalar_one_or_none()

    featured_image_result = await session.execute(
        select(ContentFeaturedImage).where(ContentFeaturedImage.brief_id == brief_id)
    )
    existing_featured_image = featured_image_result.scalar_one_or_none()
    featured_image_generator = FeaturedImageGenerationService()
    attempts, backoff_ms = generation_retry_settings()
    featured_image = await retry_with_backoff(
        attempts=attempts,
        backoff_ms=backoff_ms,
        coro_factory=lambda: featured_image_generator.generate_for_brief(
            session=session,
            project_id=project_id,
            brief=brief,
            brand=brand,
            existing=existing_featured_image,
            force_regenerate=True,
        ),
    )
    locked_title = featured_image.title_text

    conversion_intents = [project.primary_goal] if project.primary_goal else []
    generator = ArticleGenerationService(project.domain)
    artifact = await generator.generate_with_repair(
        brief=_brief_payload(brief, locked_title=locked_title),
        writer_instructions=_writer_instructions_payload(instructions),
        brief_delta=_brief_delta_payload(delta),
        brand_context=_build_brand_context(brand),
        conversion_intents=conversion_intents,
    )
    _enforce_locked_title(artifact.modular_document, locked_title)
    artifact.title = locked_title
    assigned_author = await _resolve_author_for_regeneration(
        session=session,
        project_id=project_id,
        existing_author_id=article.author_id,
    )
    if assigned_author is not None:
        artifact.modular_document["author"] = author_modular_document_payload(assigned_author)
    else:
        artifact.modular_document.pop("author", None)
    artifact.modular_document["featured_image"] = modular_featured_image_payload(
        featured_image=featured_image
    )
    artifact.rendered_html = render_modular_document(artifact.modular_document)

    next_version = article.current_version + 1
    keyword_usages = []
    try:
        keyword_coverage_report, keyword_usages = await analyze_keyword_usage(
            session=session,
            brief=brief,
            document=artifact.modular_document,
            article_version_number=next_version,
        )
        artifact.qa_report = with_keyword_coverage_report(
            artifact.qa_report,
            keyword_coverage_report,
        )
    except Exception as exc:
        logger.warning(
            "Keyword coverage analysis failed during regeneration",
            extra={"brief_id": brief_id, "article_id": str(article.id), "error": str(exc)},
        )

    article.patch(
        session,
        ContentArticlePatchDTO.from_partial(
            {
                "title": artifact.title,
                "slug": artifact.slug,
                "primary_keyword": artifact.primary_keyword,
                "author_id": str(assigned_author.id) if assigned_author is not None else None,
                "modular_document": artifact.modular_document,
                "rendered_html": artifact.rendered_html,
                "qa_report": artifact.qa_report,
                "status": artifact.status,
                "current_version": next_version,
                "generation_model": artifact.generation_model,
                "generation_temperature": artifact.generation_temperature,
            }
        ),
    )

    ContentArticleVersion.create(
        session,
        ContentArticleVersionCreateDTO(
            article_id=str(article.id),
            version_number=next_version,
            title=artifact.title,
            slug=artifact.slug,
            primary_keyword=artifact.primary_keyword,
            modular_document=artifact.modular_document,
            rendered_html=artifact.rendered_html,
            qa_report=artifact.qa_report,
            status=artifact.status,
            change_reason=payload.reason or "manual_regeneration",
            generation_model=artifact.generation_model,
            generation_temperature=artifact.generation_temperature,
            created_by_regeneration=True,
        ),
    )
    await persist_article_keyword_usages(
        session=session,
        article_id=str(article.id),
        article_version_number=next_version,
        brief_id=str(brief.id),
        keyword_usages=keyword_usages,
    )

    await session.flush()
    await session.refresh(article)
    pillar_payloads = await _load_pillar_payloads_for_briefs(
        session=session,
        project_id=project_id,
        brief_ids=[str(article.brief_id)],
    )
    return _article_detail_response(
        article,
        pillar_payload=pillar_payloads.get(str(article.brief_id)),
    )


@router.get(
    "/{project_id}/articles/{article_id}/versions/{version_number}",
    response_model=ContentArticleVersionResponse,
    summary="Get article version",
    description="Fetch an immutable article version snapshot by version number.",
)
async def get_article_version(
    project_id: str,
    article_id: str,
    version_number: int,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentArticleVersion:
    """Get a specific article version snapshot."""
    await get_user_project(project_id, current_user, session)

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

    version_result = await session.execute(
        select(ContentArticleVersion).where(
            ContentArticleVersion.article_id == article_id,
            ContentArticleVersion.version_number == version_number,
        )
    )
    version = version_result.scalar_one_or_none()
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CONTENT_ARTICLE_VERSION_NOT_FOUND_DETAIL,
        )

    return version
