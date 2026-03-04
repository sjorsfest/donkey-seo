"""FastAPI application entry point."""

import asyncio
import logging
from copy import deepcopy
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any, cast

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.integration import integration_app
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
from app.services.discovery_pipeline_halt import read_discovery_reconciliation_metrics
from app.services.pipeline_task_manager import (
    get_content_pipeline_task_manager,
    get_discovery_pipeline_task_manager,
    get_setup_pipeline_task_manager,
)
from app.services.publication_webhook import run_publication_webhook_nightly_scheduler

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

    publication_stop_event = asyncio.Event()
    publication_scheduler_task: asyncio.Task[None] | None = None
    if settings.publication_webhook_auto_start:
        publication_scheduler_task = asyncio.create_task(
            run_publication_webhook_nightly_scheduler(
                stop_event=publication_stop_event,
                batch_size=max(1, int(settings.publication_webhook_batch_size)),
            ),
            name="publication-webhook-nightly-scheduler",
        )
        logger.info(
            "Publication webhook nightly scheduler auto-started",
            extra={
                "batch_size": max(1, int(settings.publication_webhook_batch_size)),
                "run_time_utc": "00:00",
            },
        )

    try:
        yield
    finally:
        if publication_scheduler_task is not None:
            publication_stop_event.set()
            publication_scheduler_task.cancel()
            with suppress(asyncio.CancelledError):
                await publication_scheduler_task

        logger.info("Shutting down DonkeySEO")
        await close_redis()
        await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    base_description = (
        "Keyword research backend service with modular setup/discovery/content pipelines "
        "for programmatic content planning and article generation"
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

    # Include API routers
    app.include_router(api_router, prefix=settings.resolved_internal_api_prefix)
    app.mount(settings.versioned_integration_api_prefix, integration_app)

    @app.get("/documentation", include_in_schema=False)
    async def integration_documentation() -> Any:
        """Render integration API docs without the app's versioned API prefix."""
        return get_swagger_ui_html(
            openapi_url="/documentation/openapi.json",
            title=f"{settings.app_name} Integration API Documentation",
            swagger_ui_parameters={
                "docExpansion": "list",
                "defaultModelsExpandDepth": -1,
                "displayRequestDuration": True,
                "persistAuthorization": True,
            },
        )

    @app.get("/documentation/redoc", include_in_schema=False)
    async def integration_documentation_redoc() -> Any:
        """Render ReDoc for integration API docs without versioned API prefix."""
        return get_redoc_html(
            openapi_url="/documentation/openapi.json",
            title=f"{settings.app_name} Integration API Reference",
        )

    @app.get("/documentation/openapi.json", include_in_schema=False)
    async def integration_documentation_openapi() -> JSONResponse:
        """Expose integration-only OpenAPI schema for the standalone docs route."""
        schema = deepcopy(integration_app.openapi())
        schema["servers"] = [
            {
                "url": settings.versioned_integration_api_prefix,
                "description": "Integration API base path",
            }
        ]
        return JSONResponse(schema)

    @app.get(
        "/health",
        summary="Health check",
        description="Return service health status and backend version information.",
    )
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": settings.app_version}

    @app.get(
        "/health/queue",
        summary="Queue health check",
        description="Return Redis-backed queue lengths for setup/discovery/content modules.",
    )
    async def queue_health_check() -> dict[str, Any]:
        """Queue health endpoint."""
        setup_manager = get_setup_pipeline_task_manager()
        discovery_manager = get_discovery_pipeline_task_manager()
        content_manager = get_content_pipeline_task_manager()

        setup_count, discovery_count, content_count = await asyncio.gather(
            setup_manager.get_queue_size(),
            discovery_manager.get_queue_size(),
            content_manager.get_queue_size(),
        )
        reconciliation = await read_discovery_reconciliation_metrics()
        reconciliation_payload = reconciliation or {}

        return {
            "status": "healthy",
            "version": settings.app_version,
            "queues": {
                "setup": {
                    "queued": setup_count,
                    "limit": setup_manager.queue_size_limit,
                    "workers": setup_manager.worker_count,
                },
                "discovery": {
                    "queued": discovery_count,
                    "limit": discovery_manager.queue_size_limit,
                    "workers": discovery_manager.worker_count,
                },
                "content": {
                    "queued": content_count,
                    "limit": content_manager.queue_size_limit,
                    "workers": content_manager.worker_count,
                },
            },
            "total_queued": setup_count + discovery_count + content_count,
            "discovery_auto_halt_reconciliation": {
                "last_run_started_at": reconciliation_payload.get("started_at"),
                "last_run_finished_at": reconciliation_payload.get("finished_at"),
                "last_status": reconciliation_payload.get("status"),
                "last_resumed_runs": reconciliation_payload.get("resumed_runs"),
                "last_error_message": reconciliation_payload.get("error_message"),
            },
        }

    return app


app = create_app()
