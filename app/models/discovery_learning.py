"""Discovery-loop iteration learnings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.generated_dtos import (
        DiscoveryIterationLearningCreateDTO,
        DiscoveryIterationLearningPatchDTO,
    )
else:
    try:
        from app.models.generated_dtos import (
            DiscoveryIterationLearningCreateDTO,
            DiscoveryIterationLearningPatchDTO,
        )
    except ImportError:
        DiscoveryIterationLearningCreateDTO = Any
        DiscoveryIterationLearningPatchDTO = Any


class DiscoveryIterationLearning(
    TypedModelMixin[DiscoveryIterationLearningCreateDTO, DiscoveryIterationLearningPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Persisted learning extracted at the end of each discovery iteration."""

    __tablename__ = "discovery_iteration_learnings"

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pipeline_run_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    iteration_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    source_capability: Mapped[str] = mapped_column(String(100), nullable=False)
    source_agent: Mapped[str | None] = mapped_column(String(100), nullable=True)

    learning_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    learning_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    polarity: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    novelty_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_metric: Mapped[float | None] = mapped_column(Float, nullable=True)

    applies_to_capabilities: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    applies_to_agents: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    project = relationship("Project", back_populates="discovery_iteration_learnings")
    pipeline_run = relationship("PipelineRun", back_populates="discovery_iteration_learnings")

    def __repr__(self) -> str:
        return (
            f"<DiscoveryIterationLearning run={self.pipeline_run_id} "
            f"iter={self.iteration_index} key={self.learning_key}>"
        )
