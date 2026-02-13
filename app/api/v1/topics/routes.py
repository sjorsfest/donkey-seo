"""Topics API endpoints."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from app.api.v1.dependencies import get_user_project
from app.api.v1.topics.constants import (
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    SOME_TOPICS_NOT_FOUND_DETAIL,
    TOPIC_NOT_FOUND_DETAIL,
)
from app.dependencies import CurrentUser, DbSession
from app.models.generated_dtos import TopicCreateDTO, TopicPatchDTO
from app.models.keyword import Keyword
from app.models.topic import Topic
from app.schemas.topic import (
    TopicCreate,
    TopicDetailResponse,
    TopicHierarchyResponse,
    TopicListResponse,
    TopicMergeRequest,
    TopicResponse,
    TopicUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{project_id}", response_model=TopicListResponse)
async def list_topics(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(DEFAULT_PAGE, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> TopicListResponse:
    """List topics for a project."""
    await get_user_project(project_id, current_user, session)

    count_query = select(func.count()).select_from(Topic).where(Topic.project_id == project_id)
    total = await session.scalar(count_query) or 0

    offset = (page - 1) * page_size
    query = (
        select(Topic)
        .where(Topic.project_id == project_id)
        .order_by(Topic.priority_rank.asc().nulls_last())
        .offset(offset)
        .limit(page_size)
    )
    result = await session.execute(query)
    topics = result.scalars().all()

    return TopicListResponse(
        items=[TopicResponse.model_validate(topic) for topic in topics],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{project_id}/ranked", response_model=list[TopicResponse])
async def get_ranked_topics(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> list[Topic]:
    """Get prioritized topic backlog."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Topic)
        .where(Topic.project_id == project_id, Topic.priority_rank.isnot(None))
        .order_by(Topic.priority_rank.asc())
        .limit(limit)
    )

    return list(result.scalars().all())


@router.get("/{project_id}/hierarchy", response_model=list[TopicHierarchyResponse])
async def get_topic_hierarchy(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> list[TopicHierarchyResponse]:
    """Get topic hierarchy tree."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Topic)
        .where(Topic.project_id == project_id, Topic.parent_topic_id.is_(None))
        .options(selectinload(Topic.children))
        .order_by(Topic.priority_rank.asc().nulls_last())
    )
    root_topics = result.scalars().all()

    def build_hierarchy(topic: Topic) -> TopicHierarchyResponse:
        return TopicHierarchyResponse(
            id=str(topic.id),
            name=topic.name,
            keyword_count=topic.keyword_count,
            priority_rank=topic.priority_rank,
            children=[build_hierarchy(child) for child in topic.children],
        )

    return [build_hierarchy(topic) for topic in root_topics]


@router.get("/{project_id}/{topic_id}", response_model=TopicDetailResponse)
async def get_topic(
    project_id: uuid.UUID,
    topic_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> Topic:
    """Get detailed topic information."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Topic).where(Topic.id == topic_id, Topic.project_id == project_id)
    )
    topic = result.scalar_one_or_none()

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=TOPIC_NOT_FOUND_DETAIL,
        )

    return topic


@router.post("/{project_id}", response_model=TopicResponse, status_code=status.HTTP_201_CREATED)
async def create_topic(
    project_id: uuid.UUID,
    topic_data: TopicCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> Topic:
    """Create a new topic."""
    await get_user_project(project_id, current_user, session)

    topic = Topic.create(
        session,
        TopicCreateDTO(
            project_id=str(project_id),
            name=topic_data.name,
            description=topic_data.description,
            parent_topic_id=topic_data.parent_topic_id,
        ),
    )
    await session.flush()
    await session.refresh(topic)

    return topic


@router.put("/{project_id}/{topic_id}", response_model=TopicResponse)
async def update_topic(
    project_id: uuid.UUID,
    topic_id: uuid.UUID,
    topic_data: TopicUpdate,
    current_user: CurrentUser,
    session: DbSession,
) -> Topic:
    """Update a topic."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Topic).where(Topic.id == topic_id, Topic.project_id == project_id)
    )
    topic = result.scalar_one_or_none()

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=TOPIC_NOT_FOUND_DETAIL,
        )

    update_data = topic_data.model_dump(exclude_unset=True)
    topic.patch(
        session,
        TopicPatchDTO.from_partial(update_data),
    )

    await session.flush()
    await session.refresh(topic)

    return topic


@router.delete("/{project_id}/{topic_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_topic(
    project_id: uuid.UUID,
    topic_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    """Delete a topic."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Topic).where(Topic.id == topic_id, Topic.project_id == project_id)
    )
    topic = result.scalar_one_or_none()

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=TOPIC_NOT_FOUND_DETAIL,
        )

    await topic.delete(session)


@router.post("/{project_id}/merge", response_model=TopicResponse)
async def merge_topics(
    project_id: uuid.UUID,
    request: TopicMergeRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> Topic:
    """Merge multiple topics into one."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Topic).where(
            Topic.id.in_([uuid.UUID(topic_id) for topic_id in request.source_topic_ids]),
            Topic.project_id == project_id,
        )
    )
    source_topics = list(result.scalars().all())

    if len(source_topics) != len(request.source_topic_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=SOME_TOPICS_NOT_FOUND_DETAIL,
        )

    merged_topic = Topic.create(
        session,
        TopicCreateDTO(
            project_id=str(project_id),
            name=request.target_name,
            description=request.target_description,
            keyword_count=sum(topic.keyword_count for topic in source_topics),
            total_volume=sum(topic.total_volume or 0 for topic in source_topics),
        ),
    )
    await session.flush()

    for source_topic in source_topics:
        await session.execute(
            update(Keyword)
            .where(Keyword.topic_id == source_topic.id)
            .values(topic_id=merged_topic.id)
        )

    for source_topic in source_topics:
        await source_topic.delete(session)

    await session.refresh(merged_topic)

    logger.info(
        "Topics merged",
        extra={
            "project_id": str(project_id),
            "source_count": len(request.source_topic_ids),
            "target_name": request.target_name,
        },
    )

    return merged_topic
