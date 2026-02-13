"""Content briefs API endpoints."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.v1.content.constants import (
    CONTENT_BRIEF_NOT_FOUND_DETAIL,
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)
from app.api.v1.dependencies import get_user_project
from app.dependencies import CurrentUser, DbSession
from app.models.content import ContentBrief, WriterInstructions
from app.models.generated_dtos import ContentBriefCreateDTO, ContentBriefPatchDTO
from app.schemas.content import (
    ContentBriefCreate,
    ContentBriefDetailResponse,
    ContentBriefListResponse,
    ContentBriefResponse,
    ContentBriefUpdate,
    WriterInstructionsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{project_id}/briefs", response_model=ContentBriefListResponse)
async def list_briefs(
    project_id: uuid.UUID,
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


@router.get("/{project_id}/briefs/{brief_id}", response_model=ContentBriefDetailResponse)
async def get_brief(
    project_id: uuid.UUID,
    brief_id: uuid.UUID,
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
)
async def create_brief(
    project_id: uuid.UUID,
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


@router.put("/{project_id}/briefs/{brief_id}", response_model=ContentBriefResponse)
async def update_brief(
    project_id: uuid.UUID,
    brief_id: uuid.UUID,
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


@router.delete("/{project_id}/briefs/{brief_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_brief(
    project_id: uuid.UUID,
    brief_id: uuid.UUID,
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
)
async def get_writer_instructions(
    project_id: uuid.UUID,
    brief_id: uuid.UUID,
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
