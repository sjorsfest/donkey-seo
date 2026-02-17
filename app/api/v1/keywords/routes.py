"""Keywords API endpoints."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.v1.dependencies import get_user_project
from app.api.v1.keywords.constants import (
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    KEYWORD_NOT_FOUND_DETAIL,
    MAX_PAGE_SIZE,
)
from app.dependencies import CurrentUser, DbSession
from app.models.generated_dtos import KeywordCreateDTO, KeywordPatchDTO
from app.models.keyword import Keyword
from app.schemas.keyword import (
    KeywordBulkUpdateRequest,
    KeywordCreate,
    KeywordDetailResponse,
    KeywordListResponse,
    KeywordResponse,
    KeywordUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{project_id}",
    response_model=KeywordListResponse,
    summary="List keywords",
    description=(
        "Return paginated keywords for a project with optional filters for intent, status, topic, "
        "volume, difficulty, and text search."
    ),
)
async def list_keywords(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(DEFAULT_PAGE, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    intent: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    topic_id: uuid.UUID | None = Query(None),
    min_volume: int | None = Query(None),
    max_difficulty: float | None = Query(None),
    search: str | None = Query(None),
) -> KeywordListResponse:
    """List keywords for a project with filters."""
    await get_user_project(project_id, current_user, session)

    query = select(Keyword).where(Keyword.project_id == project_id)

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

    count_query = select(func.count()).select_from(query.subquery())
    total = await session.scalar(count_query) or 0

    offset = (page - 1) * page_size
    query = (
        query.order_by(Keyword.priority_score.desc().nulls_last())
        .offset(offset)
        .limit(page_size)
    )
    result = await session.execute(query)
    keywords = result.scalars().all()

    return KeywordListResponse(
        items=[KeywordResponse.model_validate(keyword) for keyword in keywords],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{project_id}/{keyword_id}",
    response_model=KeywordDetailResponse,
    summary="Get keyword",
    description="Return detailed information for a single keyword in the project.",
)
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
            detail=KEYWORD_NOT_FOUND_DETAIL,
        )

    return keyword


@router.post(
    "/{project_id}",
    response_model=KeywordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create keyword",
    description="Add a new keyword manually to the project.",
)
async def create_keyword(
    project_id: uuid.UUID,
    keyword_data: KeywordCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> Keyword:
    """Manually add a keyword to a project."""
    await get_user_project(project_id, current_user, session)

    keyword = Keyword.create(
        session,
        KeywordCreateDTO(
            project_id=str(project_id),
            keyword=keyword_data.keyword,
            keyword_normalized=keyword_data.keyword.lower().strip(),
            language=keyword_data.language,
            locale=keyword_data.locale,
            source=keyword_data.source,
        ),
    )
    await session.flush()
    await session.refresh(keyword)

    return keyword


@router.put(
    "/{project_id}/{keyword_id}",
    response_model=KeywordResponse,
    summary="Update keyword",
    description="Apply partial updates to an existing keyword in the project.",
)
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
            detail=KEYWORD_NOT_FOUND_DETAIL,
        )

    update_data = keyword_data.model_dump(exclude_unset=True)
    keyword.patch(
        session,
        KeywordPatchDTO.from_partial(update_data),
    )

    await session.flush()
    await session.refresh(keyword)

    return keyword


@router.delete(
    "/{project_id}/{keyword_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Exclude keyword",
    description="Soft-delete a keyword by marking its status as excluded.",
)
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
            detail=KEYWORD_NOT_FOUND_DETAIL,
        )

    keyword.patch(
        session,
        KeywordPatchDTO.from_partial({"status": "excluded"}),
    )
    await session.flush()


@router.post(
    "/{project_id}/bulk-update",
    response_model=dict[str, int],
    summary="Bulk update keywords",
    description="Update status, intent, or topic assignment for multiple keywords in one request.",
)
async def bulk_update_keywords(
    project_id: uuid.UUID,
    request: KeywordBulkUpdateRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, int]:
    """Bulk update keywords."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Keyword).where(
            Keyword.id.in_([uuid.UUID(keyword_id) for keyword_id in request.keyword_ids]),
            Keyword.project_id == project_id,
        )
    )
    keywords = result.scalars().all()

    updated = 0
    for keyword in keywords:
        patch_data: dict[str, str] = {}
        if request.status:
            patch_data["status"] = request.status
        if request.intent:
            patch_data["intent"] = request.intent
        if request.topic_id:
            patch_data["topic_id"] = str(request.topic_id)
        if patch_data:
            keyword.patch(
                session,
                KeywordPatchDTO.from_partial(patch_data),
            )
        updated += 1

    await session.flush()

    logger.info(
        "Keywords bulk updated",
        extra={"project_id": str(project_id), "updated_count": updated},
    )

    return {"updated": updated}
