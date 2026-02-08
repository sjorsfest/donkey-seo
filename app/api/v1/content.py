"""Content briefs API endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.dependencies import CurrentUser, DbSession
from app.models.content import ContentBrief, WriterInstructions
from app.models.project import Project
from app.schemas.content import (
    ContentBriefCreate,
    ContentBriefDetailResponse,
    ContentBriefListResponse,
    ContentBriefResponse,
    ContentBriefUpdate,
    WriterInstructionsResponse,
)

router = APIRouter()


async def get_user_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Helper to get and verify project ownership."""
    result = await session.execute(
        select(Project).where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return project


@router.get("/{project_id}/briefs", response_model=ContentBriefListResponse)
async def list_briefs(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: str | None = Query(None, alias="status"),
) -> ContentBriefListResponse:
    """List content briefs for a project."""
    await get_user_project(project_id, current_user, session)

    # Build query
    query = select(ContentBrief).where(ContentBrief.project_id == project_id)

    if status_filter:
        query = query.where(ContentBrief.status == status_filter)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = await session.scalar(count_query) or 0

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.order_by(ContentBrief.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    briefs = result.scalars().all()

    return ContentBriefListResponse(
        items=[ContentBriefResponse.model_validate(b) for b in briefs],
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
            detail="Content brief not found",
        )

    return brief


@router.post("/{project_id}/briefs", response_model=ContentBriefResponse, status_code=status.HTTP_201_CREATED)
async def create_brief(
    project_id: uuid.UUID,
    brief_data: ContentBriefCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> ContentBrief:
    """Create a new content brief."""
    await get_user_project(project_id, current_user, session)

    brief = ContentBrief(
        project_id=project_id,
        topic_id=uuid.UUID(brief_data.topic_id),
        primary_keyword=brief_data.primary_keyword,
        working_titles=brief_data.working_titles,
        target_word_count_min=brief_data.target_word_count_min,
        target_word_count_max=brief_data.target_word_count_max,
    )
    session.add(brief)
    await session.flush()
    await session.refresh(brief)

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
            detail="Content brief not found",
        )

    update_data = brief_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(brief, field, value)

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
            detail="Content brief not found",
        )

    await session.delete(brief)


@router.get("/{project_id}/briefs/{brief_id}/instructions", response_model=WriterInstructionsResponse | None)
async def get_writer_instructions(
    project_id: uuid.UUID,
    brief_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> WriterInstructions | None:
    """Get writer instructions for a brief."""
    await get_user_project(project_id, current_user, session)

    # Verify brief exists
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
            detail="Content brief not found",
        )

    # Get instructions
    result = await session.execute(
        select(WriterInstructions).where(WriterInstructions.brief_id == brief_id)
    )

    return result.scalar_one_or_none()
