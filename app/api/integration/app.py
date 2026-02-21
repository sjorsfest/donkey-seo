"""FastAPI app for external Donkey SEO integrations."""

from fastapi import FastAPI

from app.api.integration.docs import (
    INTEGRATION_API_DESCRIPTION,
    INTEGRATION_ARTICLES_TAG_DESCRIPTION,
    INTEGRATION_DOCS_TAG_DESCRIPTION,
)
from app.api.integration.routes import protected_router, public_router
from app.config import settings


def create_integration_app() -> FastAPI:
    """Create integration app with independent docs and OpenAPI schema."""
    app = FastAPI(
        title=f"{settings.app_name} Integration API",
        version=settings.app_version,
        description=INTEGRATION_API_DESCRIPTION.strip(),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {
                "name": "Integration Docs",
                "description": INTEGRATION_DOCS_TAG_DESCRIPTION.strip(),
            },
            {
                "name": "Integration Articles",
                "description": INTEGRATION_ARTICLES_TAG_DESCRIPTION.strip(),
            },
        ],
    )
    app.include_router(public_router, tags=["Integration Docs"])
    app.include_router(protected_router, tags=["Integration Articles"])
    return app


integration_app = create_integration_app()
