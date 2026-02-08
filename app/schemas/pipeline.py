"""Pipeline schemas."""

from datetime import datetime

from pydantic import BaseModel


class PipelineStartRequest(BaseModel):
    """Schema for starting a pipeline run."""

    start_step: int = 0
    end_step: int | None = None
    skip_steps: list[int] | None = None


class StepExecutionResponse(BaseModel):
    """Schema for step execution response."""

    id: str
    step_number: int
    step_name: str
    status: str
    progress_percent: float
    progress_message: str | None
    items_processed: int
    items_total: int | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None

    model_config = {"from_attributes": True}


class PipelineRunResponse(BaseModel):
    """Schema for pipeline run response."""

    id: str
    project_id: str
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    start_step: int | None
    end_step: int | None
    skip_steps: list[int] | None
    step_executions: list[StepExecutionResponse]
    created_at: datetime

    model_config = {"from_attributes": True}


class PipelineProgressResponse(BaseModel):
    """Schema for real-time pipeline progress."""

    run_id: str
    status: str
    current_step: int | None
    current_step_name: str | None
    overall_progress: float
    steps: list[StepExecutionResponse]
