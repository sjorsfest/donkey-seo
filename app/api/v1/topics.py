"""Topics API endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.dependencies import CurrentUser, DbSession
from app.models.project import Project
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


@router.get("/{project_id}", response_model=TopicListResponse)
async def list_topics(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> TopicListResponse:
    """List topics for a project."""
    await get_user_project(project_id, current_user, session)

    # Count total
    count_query = select(func.count()).select_from(Topic).where(Topic.project_id == project_id)
    total = await session.scalar(count_query) or 0

    # Get paginated results
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
        items=[TopicResponse.model_validate(t) for t in topics],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{project_id}/ranked", response_model=list[TopicResponse])
async def get_ranked_topics(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    limit: int = Query(50, ge=1, le=200),
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

    # Get root topics (no parent)
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

    return [build_hierarchy(t) for t in root_topics]


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
            detail="Topic not found",
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

    topic = Topic(
        project_id=project_id,
        name=topic_data.name,
        description=topic_data.description,
        parent_topic_id=uuid.UUID(topic_data.parent_topic_id) if topic_data.parent_topic_id else None,
    )
    session.add(topic)
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
            detail="Topic not found",
        )

    update_data = topic_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == "parent_topic_id" and value:
            value = uuid.UUID(value)
        setattr(topic, field, value)

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
            detail="Topic not found",
        )

    await session.delete(topic)


@router.post("/{project_id}/merge", response_model=TopicResponse)
async def merge_topics(
    project_id: uuid.UUID,
    request: TopicMergeRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> Topic:
    """Merge multiple topics into one."""
    await get_user_project(project_id, current_user, session)

    # Get source topics
    result = await session.execute(
        select(Topic).where(
            Topic.id.in_([uuid.UUID(t) for t in request.source_topic_ids]),
            Topic.project_id == project_id,
        )
    )
    source_topics = list(result.scalars().all())

    if len(source_topics) != len(request.source_topic_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Some topics not found",
        )

    # Create new merged topic
    merged_topic = Topic(
        project_id=project_id,
        name=request.target_name,
        description=request.target_description,
        keyword_count=sum(t.keyword_count for t in source_topics),
        total_volume=sum(t.total_volume or 0 for t in source_topics),
    )
    session.add(merged_topic)
    await session.flush()

    # Move keywords from source topics to merged topic
    from app.models.keyword import Keyword

    for source_topic in source_topics:
        await session.execute(
            select(Keyword)
            .where(Keyword.topic_id == source_topic.id)
            .execution_options(synchronize_session=False)
        )
        # Update keywords
        from sqlalchemy import update

        await session.execute(
            update(Keyword)
            .where(Keyword.topic_id == source_topic.id)
            .values(topic_id=merged_topic.id)
        )

    # Delete source topics
    for source_topic in source_topics:
        await session.delete(source_topic)

    await session.refresh(merged_topic)

    return merged_topic
