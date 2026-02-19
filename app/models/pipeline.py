"""Pipeline execution tracking models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin
from app.models.generated_dtos import (
    PipelineRunCreateDTO,
    PipelineRunPatchDTO,
    StepExecutionCreateDTO,
    StepExecutionPatchDTO,
)

if TYPE_CHECKING:
    from app.models.discovery_learning import DiscoveryIterationLearning
    from app.models.discovery_snapshot import DiscoveryTopicSnapshot
    from app.models.project import Project
    from app.models.topic import Topic


class PipelineRun(
    TypedModelMixin[PipelineRunCreateDTO, PipelineRunPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Track a pipeline execution run."""

    __tablename__ = "pipeline_runs"

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pipeline_module: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    parent_run_id: Mapped[str | None] = mapped_column(
        StringUUID(),
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    source_topic_id: Mapped[str | None] = mapped_column(
        StringUUID(),
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    paused_at_step: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Configuration snapshot
    steps_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    start_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    skip_steps: Mapped[list[int] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    project: Mapped[Project] = relationship("Project", back_populates="pipeline_runs")
    parent_run: Mapped[PipelineRun | None] = relationship(
        "PipelineRun",
        back_populates="child_runs",
        remote_side="PipelineRun.id",
        foreign_keys=[parent_run_id],
    )
    child_runs: Mapped[list[PipelineRun]] = relationship(
        "PipelineRun",
        back_populates="parent_run",
        cascade="all, delete-orphan",
        foreign_keys=[parent_run_id],
    )
    source_topic: Mapped[Topic | None] = relationship("Topic", foreign_keys=[source_topic_id])
    step_executions: Mapped[list[StepExecution]] = relationship(
        "StepExecution",
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
        order_by="StepExecution.step_number",
    )
    discovery_topic_snapshots: Mapped[list[DiscoveryTopicSnapshot]] = relationship(
        "DiscoveryTopicSnapshot",
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
    )
    discovery_iteration_learnings: Mapped[list[DiscoveryIterationLearning]] = relationship(
        "DiscoveryIterationLearning",
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<PipelineRun {self.id} ({self.status})>"


class StepExecution(
    TypedModelMixin[StepExecutionCreateDTO, StepExecutionPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Track individual step execution."""

    __tablename__ = "step_executions"

    pipeline_run_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Step identification
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    step_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)

    # Progress tracking
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    progress_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    items_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Results
    result_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Resumability
    checkpoint_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    pipeline_run: Mapped[PipelineRun] = relationship("PipelineRun", back_populates="step_executions")

    def __repr__(self) -> str:
        return f"<StepExecution Step {self.step_number}: {self.status}>"
