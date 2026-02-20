"""API v1 router aggregator."""

from fastapi import APIRouter

from app.api.v1 import auth, brand, content, keywords, pipeline, projects, tasks, topics

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
api_router.include_router(keywords.router, prefix="/keywords", tags=["Keywords"])
api_router.include_router(topics.router, prefix="/topics", tags=["Topics"])
api_router.include_router(content.router, prefix="/content", tags=["Content"])
api_router.include_router(brand.router, prefix="/brand", tags=["Brand"])
