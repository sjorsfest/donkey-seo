"""Dependencies for the external integration API."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger(__name__)

integration_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_integration_api_key(
    api_key: Annotated[str | None, Security(integration_api_key_header)],
) -> str:
    """Validate API key for external integration routes."""
    allowed_keys = settings.get_integration_api_keys()
    if not allowed_keys:
        logger.error("Integration API key check failed: no keys configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Integration API keys are not configured",
        )

    if not api_key or api_key not in allowed_keys:
        logger.warning("Integration API key check failed: invalid key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return api_key
