"""Project API endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.dependencies import CurrentUser, DbSession
from app.models.project import Project
from app.schemas.project import (
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)

router = APIRouter()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Create a new keyword research project."""
    project = Project(
        user_id=current_user.id,
        name=project_data.name,
        domain=project_data.domain,
        description=project_data.description,
        primary_language=project_data.primary_language,
        primary_locale=project_data.primary_locale,
        secondary_locales=project_data.secondary_locales,
        primary_goal=project_data.goals.primary_objective if project_data.goals else None,
        secondary_goals=project_data.goals.secondary_goals if project_data.goals else None,
        skip_steps=project_data.settings.skip_steps if project_data.settings else None,
    )
    session.add(project)
    await session.flush()
    await session.refresh(project)

    return project


@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> ProjectListResponse:
    """List all projects for the current user."""
    # Count total
    count_query = select(func.count()).select_from(Project).where(Project.user_id == current_user.id)
    total = await session.scalar(count_query) or 0

    # Get paginated results
    offset = (page - 1) * page_size
    query = (
        select(Project)
        .where(Project.user_id == current_user.id)
        .order_by(Project.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await session.execute(query)
    projects = result.scalars().all()

    return ProjectListResponse(
        items=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Get a specific project by ID."""
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


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    project_data: ProjectUpdate,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Update a project."""
    result = await session.execute(
        select(Project).where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    # Update fields
    update_data = project_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(project, field):
            setattr(project, field, value)

    await session.flush()
    await session.refresh(project)

    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    """Delete a project."""
    result = await session.execute(
        select(Project).where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    await session.delete(project)
