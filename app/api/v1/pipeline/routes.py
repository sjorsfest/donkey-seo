"""Pipeline API endpoints."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.v1.dependencies import get_user_project
from app.api.v1.pipeline.constants import (
    DEFAULT_RUN_LIMIT,
    MAX_RUN_LIMIT,
    NO_PAUSED_PIPELINE_DETAIL,
    NO_RUNNING_PIPELINE_DETAIL,
    PIPELINE_ALREADY_RUNNING_DETAIL,
    PIPELINE_QUEUE_FULL_DETAIL,
    PIPELINE_RUN_NOT_FOUND_DETAIL,
)
from app.api.v1.pipeline.openapi_docs import PIPELINE_START_OPENAPI_EXTRA
from app.dependencies import CurrentUser, DbSession
from app.models.discovery_snapshot import DiscoveryTopicSnapshot
from app.models.generated_dtos import PipelineRunCreateDTO
from app.models.pipeline import PipelineRun
from app.schemas.pipeline import (
    ContentPipelineConfig,
    DiscoveryTopicSnapshotResponse,
    PipelineProgressResponse,
    PipelineRunResponse,
    PipelineStartRequest,
    StepExecutionResponse,
)
from app.services.pipeline_task_manager import (
    PipelineQueueFullError,
    get_pipeline_task_manager,
)
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/{project_id}/start",
    response_model=PipelineRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start pipeline run",
    description=(
        "Create a new pipeline run for a project and queue background execution for the "
        "requested step range."
    ),
    openapi_extra=PIPELINE_START_OPENAPI_EXTRA,
)
async def start_pipeline(
    project_id: uuid.UUID,
    request: PipelineStartRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> PipelineRun:
    """Start a new pipeline run for a project."""
    project = await get_user_project(project_id, current_user, session)
    project_id_str = str(project_id)

    mode = request.mode
    if mode == "discovery_loop":
        start_step = 2
        end_step = 8
        skip_steps = []
    elif mode == "content_production":
        start_step = 12
        end_step = 14
        skip_steps = []
    else:
        start_step = request.start_step
        end_step = request.end_step or 14
        skip_steps = request.skip_steps or project.skip_steps or []

    logger.info(
        "Pipeline start requested",
        extra={
            "project_id": project_id_str,
            "mode": mode,
            "start_step": start_step,
            "end_step": end_step,
            "has_strategy": request.strategy is not None,
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

    strategy_payload = request.strategy.model_dump() if request.strategy else None
    discovery_payload = request.discovery.model_dump() if request.discovery else None
    content_config = request.content
    if content_config is None and mode in {"discovery_loop", "content_production"}:
        content_config = ContentPipelineConfig()
    content_payload = content_config.model_dump() if content_config else None
    step_inputs_payload: dict[str, dict[str, object]] = {}
    if content_payload:
        step_inputs_payload["12"] = {
            "max_briefs": content_payload.get("max_briefs", 20),
            "posts_per_week": content_payload.get("posts_per_week", 1),
            "preferred_weekdays": content_payload.get("preferred_weekdays", []),
            "min_lead_days": content_payload.get("min_lead_days", 7),
            "publication_start_date": content_payload.get("publication_start_date"),
            "use_llm_timing_hints": content_payload.get("use_llm_timing_hints", True),
            "llm_timing_flex_days": content_payload.get("llm_timing_flex_days", 14),
            "include_zero_data_topics": content_payload.get("include_zero_data_topics", True),
            "zero_data_topic_share": content_payload.get("zero_data_topic_share", 0.2),
            "zero_data_fit_score_min": content_payload.get("zero_data_fit_score_min", 0.65),
        }
    steps_config = {
        "start": start_step,
        "end": end_step,
        "skip": skip_steps,
        "strategy": strategy_payload,
        "primary_goal": project.primary_goal,
        "mode": mode,
        "discovery": discovery_payload,
        "content": content_payload,
        "iteration_index": 0,
        "selected_topic_ids": [],
        "step_inputs": step_inputs_payload,
    }

    pipeline_run = PipelineRun.create(
        session,
        PipelineRunCreateDTO(
            project_id=project_id_str,
            status="pending",
            start_step=start_step,
            end_step=end_step,
            skip_steps=skip_steps,
            steps_config=steps_config,
        ),
    )
    await session.flush()
    await session.refresh(pipeline_run, ["step_executions"])
    # Commit before launching long-running background work so the request
    # session does not hold an open transaction for the entire pipeline run.
    await session.commit()

    task_id = str(pipeline_run.id)
    total_steps = len([step for step in range(start_step, end_step + 1) if step not in skip_steps])
    task_manager = TaskManager()
    await task_manager.set_task_state(
        task_id=task_id,
        status="queued",
        stage="Queued pipeline execution",
        project_id=project_id_str,
        current_step=start_step,
        current_step_name=None,
        completed_steps=0,
        total_steps=total_steps,
        progress_percent=0.0,
        error_message=None,
    )

    queue = get_pipeline_task_manager()
    try:
        await queue.enqueue_start(
            project_id=project_id_str,
            run_id=task_id,
            start_step=start_step,
            end_step=end_step,
            skip_steps=skip_steps,
        )
    except PipelineQueueFullError as exc:
        pipeline_run.status = "paused"
        pipeline_run.error_message = PIPELINE_QUEUE_FULL_DETAIL
        await session.flush()
        await session.commit()
        await task_manager.set_task_state(
            task_id=task_id,
            status="paused",
            stage="Pipeline queue is full",
            error_message=PIPELINE_QUEUE_FULL_DETAIL,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PIPELINE_QUEUE_FULL_DETAIL,
        ) from exc

    return pipeline_run


@router.get(
    "/{project_id}/runs",
    response_model=list[PipelineRunResponse],
    summary="List pipeline runs",
    description=(
        "Return recent pipeline runs for a project, including step execution data, limited by "
        "the provided count."
    ),
)
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


@router.get(
    "/{project_id}/runs/{run_id}",
    response_model=PipelineRunResponse,
    summary="Get pipeline run",
    description="Return full details for a specific pipeline run in the project.",
)
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


@router.get(
    "/{project_id}/runs/{run_id}/progress",
    response_model=PipelineProgressResponse,
    summary="Get pipeline progress",
    description=(
        "Return real-time pipeline progress including run status, active step, and per-step "
        "execution details."
    ),
)
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


@router.get(
    "/{project_id}/runs/{run_id}/discovery-snapshots",
    response_model=list[DiscoveryTopicSnapshotResponse],
    summary="List discovery snapshots",
    description=(
        "Return per-iteration topic decision snapshots (accepted/rejected) "
        "for a discovery-loop run."
    ),
)
async def list_discovery_snapshots(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> list[DiscoveryTopicSnapshot]:
    """List discovery snapshots for a pipeline run."""
    await get_user_project(project_id, current_user, session)

    run_result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.id == run_id,
            PipelineRun.project_id == project_id,
        )
    )
    pipeline_run = run_result.scalar_one_or_none()
    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=PIPELINE_RUN_NOT_FOUND_DETAIL,
        )

    result = await session.execute(
        select(DiscoveryTopicSnapshot)
        .where(
            DiscoveryTopicSnapshot.project_id == project_id,
            DiscoveryTopicSnapshot.pipeline_run_id == run_id,
        )
        .order_by(
            DiscoveryTopicSnapshot.iteration_index.asc(),
            DiscoveryTopicSnapshot.created_at.asc(),
        )
    )
    return list(result.scalars().all())


@router.post(
    "/{project_id}/pause",
    status_code=status.HTTP_200_OK,
    summary="Pause running pipeline",
    description="Pause the currently running pipeline run for the project.",
)
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


@router.post(
    "/{project_id}/resume/{run_id}",
    status_code=status.HTTP_200_OK,
    summary="Resume paused pipeline",
    description="Queue background work to resume a paused pipeline run.",
)
async def resume_pipeline(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, str]:
    """Resume a paused pipeline run."""
    await get_user_project(project_id, current_user, session)

    logger.info(
        "Pipeline resume requested",
        extra={"project_id": str(project_id), "run_id": str(run_id)},
    )

    result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.id == run_id,
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

    # Don't change status here — the orchestrator handles the paused→running
    # transition in its own session to avoid a race condition.

    task_manager = TaskManager()
    await task_manager.set_task_state(
        task_id=str(pipeline_run.id),
        status="queued",
        stage="Queued pipeline resume",
        project_id=str(project_id),
        current_step=pipeline_run.paused_at_step,
        current_step_name=None,
        error_message=None,
    )

    queue = get_pipeline_task_manager()
    try:
        await queue.enqueue_resume(
            project_id=str(project_id),
            run_id=str(pipeline_run.id),
        )
    except PipelineQueueFullError as exc:
        await task_manager.set_task_state(
            task_id=str(pipeline_run.id),
            status="paused",
            stage="Pipeline queue is full",
            error_message=PIPELINE_QUEUE_FULL_DETAIL,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PIPELINE_QUEUE_FULL_DETAIL,
        ) from exc

    return {"message": "Pipeline resumed"}
