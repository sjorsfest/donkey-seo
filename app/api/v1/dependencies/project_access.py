"""Shared dependencies for project ownership access checks."""


from fastapi import HTTPException, status
from sqlalchemy import select

from app.dependencies import CurrentUser, DbSession
from app.models.project import Project

PROJECT_NOT_FOUND_DETAIL = "Project not found"


async def get_user_project(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Return a project only when it belongs to the current user."""
    result = await session.execute(
        select(Project).where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=PROJECT_NOT_FOUND_DETAIL,
        )

    return project
