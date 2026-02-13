"""Utility helpers for pipeline routes."""

import logging

from app.core.database import get_session_context
from app.services.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


async def run_pipeline_background(
    project_id: str,
    run_id: str,
    start_step: int,
    end_step: int,
    skip_steps: list[int],
) -> None:
    """Execute the pipeline in the background with a fresh DB session."""
    try:
        async with get_session_context() as session:
            orchestrator = PipelineOrchestrator(session, project_id)
            await orchestrator.start_pipeline(
                run_id=run_id,
                start_step=start_step,
                end_step=end_step,
                skip_steps=skip_steps,
            )
    except Exception:
        logger.exception("Pipeline execution failed for project %s", project_id)


async def resume_pipeline_background(project_id: str, run_id: str) -> None:
    """Resume a pipeline in the background with a fresh DB session."""
    try:
        async with get_session_context() as session:
            orchestrator = PipelineOrchestrator(session, project_id)
            await orchestrator.resume_pipeline(run_id)
    except Exception:
        logger.exception("Pipeline resume failed for project %s", project_id)
