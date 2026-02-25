"""Dependencies for the external integration API."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_session
from app.core.integration_keys import hash_integration_api_key
from app.models.project import Project

logger = logging.getLogger(__name__)

integration_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _is_valid_project_api_key(
    *,
    session: AsyncSession,
    project_id: str,
    api_key: str,
) -> bool:
    """Return True when key hash matches the target project's active key."""
    api_key_hash = hash_integration_api_key(api_key)
    result = await session.execute(
        select(Project.id).where(
            Project.id == project_id,
            Project.integration_api_key_hash == api_key_hash,
        )
    )
    return result.scalar_one_or_none() is not None


async def _has_any_project_api_key(session: AsyncSession) -> bool:
    """Whether any project has a persisted integration API key."""
    result = await session.execute(
        select(Project.id).where(Project.integration_api_key_hash.isnot(None)).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def require_integration_api_key(
    api_key: Annotated[str | None, Security(integration_api_key_header)],
    project_id: Annotated[str, Query(..., min_length=1)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Validate API key for external integration routes."""
    allowed_keys = settings.get_integration_api_keys()
    candidate = (api_key or "").strip()

    if candidate and candidate in allowed_keys:
        return

    if candidate and await _is_valid_project_api_key(
        session=session,
        project_id=project_id,
        api_key=candidate,
    ):
        return

    if not allowed_keys and not await _has_any_project_api_key(session):
        logger.error("Integration API key check failed: no keys configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Integration API keys are not configured",
        )

    logger.warning(
        "Integration API key check failed: invalid key",
        extra={"project_id": project_id},
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
