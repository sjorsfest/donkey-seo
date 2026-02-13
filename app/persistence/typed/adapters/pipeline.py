"""Pipeline model write adapters."""

from __future__ import annotations

from app.models.generated_dtos import (
    PipelineRunCreateDTO,
    PipelineRunPatchDTO,
    StepExecutionCreateDTO,
    StepExecutionPatchDTO,
)
from app.models.pipeline import PipelineRun, StepExecution
from app.persistence.typed.adapters._base import BaseWriteAdapter

PIPELINE_RUN_PATCH_ALLOWLIST = {
    "status",
    "error_message",
    "paused_at_step",
    "steps_config",
    "start_step",
    "end_step",
    "skip_steps",
    "started_at",
    "completed_at",
}

STEP_EXECUTION_PATCH_ALLOWLIST = {
    "step_name",
    "started_at",
    "completed_at",
    "status",
    "progress_percent",
    "progress_message",
    "items_processed",
    "items_total",
    "result_summary",
    "error_message",
    "error_traceback",
    "checkpoint_data",
}

_PIPELINE_RUN_ADAPTER = BaseWriteAdapter[
    PipelineRun,
    PipelineRunCreateDTO,
    PipelineRunPatchDTO,
](
    model_cls=PipelineRun,
    patch_allowlist=PIPELINE_RUN_PATCH_ALLOWLIST,
)

_STEP_EXECUTION_ADAPTER = BaseWriteAdapter[
    StepExecution,
    StepExecutionCreateDTO,
    StepExecutionPatchDTO,
](
    model_cls=StepExecution,
    patch_allowlist=STEP_EXECUTION_PATCH_ALLOWLIST,
)


def register() -> None:
    """Register pipeline adapters."""
    from app.persistence.typed.registry import register_adapter

    register_adapter(PipelineRun, _PIPELINE_RUN_ADAPTER)
    register_adapter(StepExecution, _STEP_EXECUTION_ADAPTER)
