"""Keywords API endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.dependencies import CurrentUser, DbSession
from app.models.keyword import Keyword
from app.models.project import Project
from app.schemas.keyword import (
    KeywordBulkUpdateRequest,
    KeywordCreate,
    KeywordDetailResponse,
    KeywordListResponse,
    KeywordResponse,
    KeywordUpdate,
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


@router.get("/{project_id}", response_model=KeywordListResponse)
async def list_keywords(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    intent: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    topic_id: uuid.UUID | None = Query(None),
    min_volume: int | None = Query(None),
    max_difficulty: float | None = Query(None),
    search: str | None = Query(None),
) -> KeywordListResponse:
    """List keywords for a project with filters."""
    await get_user_project(project_id, current_user, session)

    # Build query
    query = select(Keyword).where(Keyword.project_id == project_id)

    # Apply filters
    if intent:
        query = query.where(Keyword.intent == intent)
    if status_filter:
        query = query.where(Keyword.status == status_filter)
    if topic_id:
        query = query.where(Keyword.topic_id == topic_id)
    if min_volume is not None:
        query = query.where(Keyword.search_volume >= min_volume)
    if max_difficulty is not None:
        query = query.where(Keyword.difficulty <= max_difficulty)
    if search:
        query = query.where(Keyword.keyword.ilike(f"%{search}%"))

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = await session.scalar(count_query) or 0

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.order_by(Keyword.priority_score.desc().nulls_last()).offset(offset).limit(page_size)
    result = await session.execute(query)
    keywords = result.scalars().all()

    return KeywordListResponse(
        items=[KeywordResponse.model_validate(k) for k in keywords],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{project_id}/{keyword_id}", response_model=KeywordDetailResponse)
async def get_keyword(
    project_id: uuid.UUID,
    keyword_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> Keyword:
    """Get detailed keyword information."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Keyword).where(Keyword.id == keyword_id, Keyword.project_id == project_id)
    )
    keyword = result.scalar_one_or_none()

    if not keyword:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Keyword not found",
        )

    return keyword


@router.post("/{project_id}", response_model=KeywordResponse, status_code=status.HTTP_201_CREATED)
async def create_keyword(
    project_id: uuid.UUID,
    keyword_data: KeywordCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> Keyword:
    """Manually add a keyword to a project."""
    await get_user_project(project_id, current_user, session)

    keyword = Keyword(
        project_id=project_id,
        keyword=keyword_data.keyword,
        keyword_normalized=keyword_data.keyword.lower().strip(),
        language=keyword_data.language,
        locale=keyword_data.locale,
        source=keyword_data.source,
    )
    session.add(keyword)
    await session.flush()
    await session.refresh(keyword)

    return keyword


@router.put("/{project_id}/{keyword_id}", response_model=KeywordResponse)
async def update_keyword(
    project_id: uuid.UUID,
    keyword_id: uuid.UUID,
    keyword_data: KeywordUpdate,
    current_user: CurrentUser,
    session: DbSession,
) -> Keyword:
    """Update a keyword."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Keyword).where(Keyword.id == keyword_id, Keyword.project_id == project_id)
    )
    keyword = result.scalar_one_or_none()

    if not keyword:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Keyword not found",
        )

    update_data = keyword_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(keyword, field, value)

    await session.flush()
    await session.refresh(keyword)

    return keyword


@router.delete("/{project_id}/{keyword_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_keyword(
    project_id: uuid.UUID,
    keyword_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    """Delete (exclude) a keyword."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Keyword).where(Keyword.id == keyword_id, Keyword.project_id == project_id)
    )
    keyword = result.scalar_one_or_none()

    if not keyword:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Keyword not found",
        )

    # Soft delete by setting status
    keyword.status = "excluded"
    await session.flush()


@router.post("/{project_id}/bulk-update", response_model=dict[str, int])
async def bulk_update_keywords(
    project_id: uuid.UUID,
    request: KeywordBulkUpdateRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, int]:
    """Bulk update keywords."""
    await get_user_project(project_id, current_user, session)

    # Get keywords
    result = await session.execute(
        select(Keyword).where(
            Keyword.id.in_([uuid.UUID(k) for k in request.keyword_ids]),
            Keyword.project_id == project_id,
        )
    )
    keywords = result.scalars().all()

    updated = 0
    for keyword in keywords:
        if request.status:
            keyword.status = request.status
        if request.intent:
            keyword.intent = request.intent
        if request.topic_id:
            keyword.topic_id = uuid.UUID(request.topic_id)
        updated += 1

    await session.flush()

    return {"updated": updated}
