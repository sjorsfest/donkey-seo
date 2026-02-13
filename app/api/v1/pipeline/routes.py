"""Pipeline API endpoints."""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.v1.dependencies import get_user_project
from app.api.v1.pipeline.constants import (
    DEFAULT_RUN_LIMIT,
    MAX_RUN_LIMIT,
    NO_PAUSED_PIPELINE_DETAIL,
    NO_RUNNING_PIPELINE_DETAIL,
    PIPELINE_ALREADY_RUNNING_DETAIL,
    PIPELINE_RUN_NOT_FOUND_DETAIL,
)
from app.api.v1.pipeline.utils import (
    resume_pipeline_background,
    run_pipeline_background,
)
from app.dependencies import CurrentUser, DbSession
from app.models.generated_dtos import PipelineRunCreateDTO
from app.models.pipeline import PipelineRun
from app.schemas.pipeline import (
    PipelineProgressResponse,
    PipelineRunResponse,
    PipelineStartRequest,
    StepExecutionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/{project_id}/start",
    response_model=PipelineRunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_pipeline(
    project_id: uuid.UUID,
    request: PipelineStartRequest,
    current_user: CurrentUser,
    session: DbSession,
    background_tasks: BackgroundTasks,
) -> PipelineRun:
    """Start a new pipeline run for a project."""
    project = await get_user_project(project_id, current_user, session)
    project_id_str = str(project_id)

    logger.info(
        "Pipeline start requested",
        extra={
            "project_id": project_id_str,
            "start_step": request.start_step,
            "end_step": request.end_step,
        },
    )

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
            detail=PIPELINE_ALREADY_RUNNING_DETAIL,
        )

    end_step = request.end_step or 13
    skip_steps = request.skip_steps or project.skip_steps or []

    pipeline_run = PipelineRun.create(
        session,
        PipelineRunCreateDTO(
            project_id=project_id_str,
            status="pending",
            start_step=request.start_step,
            end_step=end_step,
            skip_steps=skip_steps,
            steps_config={
                "start": request.start_step,
                "end": end_step,
                "skip": skip_steps,
            },
        ),
    )
    await session.flush()
    await session.refresh(pipeline_run, ["step_executions"])
    # Commit before launching long-running background work so the request
    # session does not hold an open transaction for the entire pipeline run.
    await session.commit()

    background_tasks.add_task(
        run_pipeline_background,
        project_id=project_id_str,
        run_id=str(pipeline_run.id),
        start_step=request.start_step,
        end_step=end_step,
        skip_steps=skip_steps,
    )

    return pipeline_run


@router.get("/{project_id}/runs", response_model=list[PipelineRunResponse])
async def list_pipeline_runs(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    limit: int = Query(DEFAULT_RUN_LIMIT, ge=1, le=MAX_RUN_LIMIT),
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
            detail=PIPELINE_RUN_NOT_FOUND_DETAIL,
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
            detail=PIPELINE_RUN_NOT_FOUND_DETAIL,
        )

    steps = pipeline_run.step_executions
    if steps:
        completed = sum(1 for step in steps if step.status == "completed")
        total = len(steps)
        overall_progress = (completed / total) * 100 if total > 0 else 0
        current_step = next((step for step in steps if step.status == "running"), None)
    else:
        overall_progress = 0
        current_step = None

    return PipelineProgressResponse(
        run_id=str(pipeline_run.id),
        status=pipeline_run.status,
        current_step=current_step.step_number if current_step else None,
        current_step_name=current_step.step_name if current_step else None,
        overall_progress=overall_progress,
        steps=[StepExecutionResponse.model_validate(step) for step in steps],
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
            detail=NO_RUNNING_PIPELINE_DETAIL,
        )

    pipeline_run.status = "paused"
    await session.flush()

    logger.info("Pipeline paused", extra={"project_id": str(project_id)})

    return {"message": "Pipeline paused"}


@router.post("/{project_id}/resume", status_code=status.HTTP_200_OK)
async def resume_pipeline(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Resume a paused pipeline."""
    await get_user_project(project_id, current_user, session)

    logger.info("Pipeline resume requested", extra={"project_id": str(project_id)})

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
            detail=NO_PAUSED_PIPELINE_DETAIL,
        )

    pipeline_run.status = "running"
    await session.flush()
    # Commit before launching long-running background work so the request
    # session does not hold an open transaction for the entire pipeline run.
    await session.commit()

    background_tasks.add_task(
        resume_pipeline_background,
        project_id=str(project_id),
        run_id=str(pipeline_run.id),
    )

    return {"message": "Pipeline resumed"}
