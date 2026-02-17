"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.pipeline.openapi_docs import (
    OPENAPI_PIPELINE_GUIDE_JSON,
    OPENAPI_PIPELINE_GUIDE_MARKDOWN,
    PIPELINE_TAG_DESCRIPTION,
)
from app.api.v1.router import api_router
from app.config import settings
from app.core.database import close_db, init_db
from app.core.logging import setup_logging
from app.core.redis import close_redis

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    setup_logging()

    logger.info(
        "Starting DonkeySEO",
        extra={
            "environment": settings.environment,
            "version": settings.app_version,
            "model_reasoning": settings.get_model("reasoning"),
            "model_standard": settings.get_model("standard"),
            "model_fast": settings.get_model("fast"),
        },
    )

    if settings.environment == "development":
        await init_db()
        logger.info("Development database initialized")

    yield

    logger.info("Shutting down DonkeySEO")
    await close_redis()
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    base_description = (
        "Keyword research backend service with a 14-step pipeline for "
        "programmatic content planning and modular article generation"
    )
    full_description = (
        f"{base_description}\n\n"
        f"{OPENAPI_PIPELINE_GUIDE_MARKDOWN.strip()}\n\n"
        "```json\n"
        f"{OPENAPI_PIPELINE_GUIDE_JSON}\n"
        "```"
    )
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=full_description,
        openapi_tags=[
            {
                "name": "Pipeline",
                "description": PIPELINE_TAG_DESCRIPTION.strip(),
            }
        ],
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        cast(Any, CORSMiddleware),
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get(
        "/health",
        summary="Health check",
        description="Return service health status and backend version information.",
    )
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": settings.app_version}

    return app


app = create_app()
