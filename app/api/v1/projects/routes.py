"""Project API endpoints."""

import logging
import uuid

from fastapi import APIRouter, Query, status
from sqlalchemy import func, select

from app.api.v1.dependencies import get_user_project
from app.api.v1.projects.constants import DEFAULT_PAGE, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.dependencies import CurrentUser, DbSession
from app.models.generated_dtos import ProjectCreateDTO, ProjectPatchDTO
from app.models.project import Project
from app.schemas.project import (
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create project",
    description=(
        "Create a keyword research project with domain, locale, goals, and optional pipeline "
        "settings."
    ),
)
async def create_project(
    project_data: ProjectCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Create a new keyword research project."""
    project = Project.create(
        session,
        ProjectCreateDTO(
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
        ),
    )
    await session.flush()
    await session.refresh(project)

    logger.info(
        "Project created",
        extra={
            "project_id": str(project.id),
            "domain": project.domain,
            "user_id": str(current_user.id),
        },
    )

    return project


@router.get(
    "/",
    response_model=ProjectListResponse,
    summary="List projects",
    description="Return paginated projects for the authenticated user, ordered by most recent.",
)
async def list_projects(
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(DEFAULT_PAGE, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> ProjectListResponse:
    """List all projects for the current user."""
    count_query = (
        select(func.count())
        .select_from(Project)
        .where(Project.user_id == current_user.id)
    )
    total = await session.scalar(count_query) or 0

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


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Get project",
    description="Return a single project by ID when it belongs to the authenticated user.",
)
async def get_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Get a specific project by ID."""
    return await get_user_project(project_id, current_user, session)


@router.put(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Update project",
    description="Apply partial updates to editable fields on an existing project.",
)
async def update_project(
    project_id: uuid.UUID,
    project_data: ProjectUpdate,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Update a project."""
    project = await get_user_project(project_id, current_user, session)

    update_data = project_data.model_dump(exclude_unset=True)
    patch_data = {
        field: value
        for field, value in update_data.items()
        if hasattr(project, field)
    }
    project.patch(
        session,
        ProjectPatchDTO.from_partial(patch_data),
    )

    await session.flush()
    await session.refresh(project)

    return project


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete project",
    description="Delete a project owned by the authenticated user.",
)
async def delete_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    """Delete a project."""
    project = await get_user_project(project_id, current_user, session)

    logger.info(
        "Project deleted",
        extra={"project_id": str(project_id), "user_id": str(current_user.id)},
    )

    await project.delete(session)
