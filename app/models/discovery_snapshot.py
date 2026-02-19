"""Discovery-loop topic decision snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.generated_dtos import (
        DiscoveryTopicSnapshotCreateDTO,
        DiscoveryTopicSnapshotPatchDTO,
    )
else:
    try:
        from app.models.generated_dtos import (
            DiscoveryTopicSnapshotCreateDTO,
            DiscoveryTopicSnapshotPatchDTO,
        )
    except ImportError:
        DiscoveryTopicSnapshotCreateDTO = Any
        DiscoveryTopicSnapshotPatchDTO = Any


class DiscoveryTopicSnapshot(
    TypedModelMixin[DiscoveryTopicSnapshotCreateDTO, DiscoveryTopicSnapshotPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Per-iteration snapshot of topic acceptance/rejection decisions."""

    __tablename__ = "discovery_topic_snapshots"

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
    source_topic_id: Mapped[str | None] = mapped_column(StringUUID(), nullable=True)
    topic_name: Mapped[str] = mapped_column(String(255), nullable=False)
    fit_tier: Mapped[str | None] = mapped_column(String(20), nullable=True)
    fit_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    keyword_difficulty: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain_diversity: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated_intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    validated_page_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    top_domains: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    decision: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    rejection_reasons: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    project = relationship("Project", back_populates="discovery_topic_snapshots")
    pipeline_run = relationship("PipelineRun", back_populates="discovery_topic_snapshots")

    def __repr__(self) -> str:
        return (
            f"<DiscoveryTopicSnapshot run={self.pipeline_run_id} "
            f"iter={self.iteration_index} topic={self.topic_name}>"
        )
