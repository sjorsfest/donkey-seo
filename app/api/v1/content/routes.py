"""Content briefs API endpoints."""

import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import and_, func, select

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
from app.models.brand import BrandProfile
from app.models.content import (
    ContentArticle,
    ContentArticleVersion,
    ContentBrief,
    WriterInstructions,
)
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
    RegenerateArticleRequest,
    WriterInstructionsResponse,
)
from app.services.article_generation import ArticleGenerationService

logger = logging.getLogger(__name__)

router = APIRouter()


def _brief_payload(brief: ContentBrief) -> dict:
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


def _resolve_calendar_state(
    *,
    has_writer_instructions: bool,
    article: ContentArticle | None,
) -> ContentCalendarItemState:
    if article and (article.publish_status == "published" or article.published_at is not None):
        return "published"
    if article and article.status == "needs_review":
        return "article_needs_review"
    if article:
        return "article_ready"
    if has_writer_instructions:
        return "writer_instructions_ready"
    return "brief_ready"


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
    brief.patch(
        session,
        ContentBriefPatchDTO.from_partial(update_data),
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
) -> ContentArticleListResponse:
    """List generated content articles for a project."""
    await get_user_project(project_id, current_user, session)

    query = select(ContentArticle).where(ContentArticle.project_id == project_id)
    if status_filter:
        query = query.where(ContentArticle.status == status_filter)

    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    offset = (page - 1) * page_size
    result = await session.execute(
        query.order_by(ContentArticle.created_at.desc()).offset(offset).limit(page_size)
    )
    items = result.scalars().all()

    return ContentArticleListResponse(
        items=[ContentArticleResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
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
) -> ContentArticle:
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

    return article


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
) -> ContentArticle:
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

    conversion_intents = [project.primary_goal] if project.primary_goal else []
    generator = ArticleGenerationService(project.domain)
    artifact = await generator.generate_with_repair(
        brief=_brief_payload(brief),
        writer_instructions=_writer_instructions_payload(instructions),
        brief_delta=_brief_delta_payload(delta),
        brand_context=_build_brand_context(brand),
        conversion_intents=conversion_intents,
    )

    next_version = article.current_version + 1
    article.patch(
        session,
        ContentArticlePatchDTO.from_partial(
            {
                "title": artifact.title,
                "slug": artifact.slug,
                "primary_keyword": artifact.primary_keyword,
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

    await session.flush()
    await session.refresh(article)
    return article


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
