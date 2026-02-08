"""Pipeline API endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.dependencies import CurrentUser, DbSession
from app.models.pipeline import PipelineRun, StepExecution
from app.models.project import Project
from app.schemas.pipeline import (
    PipelineProgressResponse,
    PipelineRunResponse,
    PipelineStartRequest,
    StepExecutionResponse,
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


@router.post("/{project_id}/start", response_model=PipelineRunResponse, status_code=status.HTTP_201_CREATED)
async def start_pipeline(
    project_id: uuid.UUID,
    request: PipelineStartRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> PipelineRun:
    """Start a new pipeline run for a project."""
    project = await get_user_project(project_id, current_user, session)

    # Check if pipeline is already running
    result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.project_id == project_id,
            PipelineRun.status == "running",
        )
    )
    existing_run = result.scalar_one_or_none()

    if existing_run:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pipeline is already running for this project",
        )

    # Create pipeline run
    pipeline_run = PipelineRun(
        project_id=project_id,
        status="pending",
        start_step=request.start_step,
        end_step=request.end_step or 13,
        skip_steps=request.skip_steps or project.skip_steps or [],
        steps_config={
            "start": request.start_step,
            "end": request.end_step or 13,
            "skip": request.skip_steps or [],
        },
    )
    session.add(pipeline_run)
    await session.flush()
    await session.refresh(pipeline_run)

    # TODO: Actually start the pipeline execution via background task
    # For now, just create the run record

    return pipeline_run


@router.get("/{project_id}/runs", response_model=list[PipelineRunResponse])
async def list_pipeline_runs(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    limit: int = Query(10, ge=1, le=50),
) -> list[PipelineRun]:
    """List pipeline runs for a project."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.project_id == project_id)
        .options(selectinload(PipelineRun.step_executions))
        .order_by(PipelineRun.created_at.desc())
        .limit(limit)
    )

    return list(result.scalars().all())


@router.get("/{project_id}/runs/{run_id}", response_model=PipelineRunResponse)
async def get_pipeline_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> PipelineRun:
    """Get a specific pipeline run."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.id == run_id, PipelineRun.project_id == project_id)
        .options(selectinload(PipelineRun.step_executions))
    )
    pipeline_run = result.scalar_one_or_none()

    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline run not found",
        )

    return pipeline_run


@router.get("/{project_id}/runs/{run_id}/progress", response_model=PipelineProgressResponse)
async def get_pipeline_progress(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> PipelineProgressResponse:
    """Get real-time progress for a pipeline run."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.id == run_id, PipelineRun.project_id == project_id)
        .options(selectinload(PipelineRun.step_executions))
    )
    pipeline_run = result.scalar_one_or_none()

    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline run not found",
        )

    # Calculate overall progress
    steps = pipeline_run.step_executions
    if steps:
        completed = sum(1 for s in steps if s.status == "completed")
        total = len(steps)
        overall_progress = (completed / total) * 100 if total > 0 else 0
        current_step = next((s for s in steps if s.status == "running"), None)
    else:
        overall_progress = 0
        current_step = None

    return PipelineProgressResponse(
        run_id=str(pipeline_run.id),
        status=pipeline_run.status,
        current_step=current_step.step_number if current_step else None,
        current_step_name=current_step.step_name if current_step else None,
        overall_progress=overall_progress,
        steps=[StepExecutionResponse.model_validate(s) for s in steps],
    )


@router.post("/{project_id}/pause", status_code=status.HTTP_200_OK)
async def pause_pipeline(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, str]:
    """Pause a running pipeline."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.project_id == project_id,
            PipelineRun.status == "running",
        )
    )
    pipeline_run = result.scalar_one_or_none()

    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No running pipeline found",
        )

    pipeline_run.status = "paused"
    await session.flush()

    return {"message": "Pipeline paused"}


@router.post("/{project_id}/resume", status_code=status.HTTP_200_OK)
async def resume_pipeline(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, str]:
    """Resume a paused pipeline."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.project_id == project_id,
            PipelineRun.status == "paused",
        )
    )
    pipeline_run = result.scalar_one_or_none()

    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No paused pipeline found",
        )

    pipeline_run.status = "running"
    await session.flush()

    # TODO: Actually resume execution

    return {"message": "Pipeline resumed"}
