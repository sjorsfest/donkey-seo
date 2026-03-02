"""Content pillar taxonomy models for article grouping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin

try:
    from app.models.generated_dtos import (
        ContentBriefPillarAssignmentCreateDTO,
        ContentBriefPillarAssignmentPatchDTO,
        ContentPillarCreateDTO,
        ContentPillarPatchDTO,
    )
except ImportError:  # pragma: no cover - bootstrap before DTO regeneration
    ContentPillarCreateDTO = Any  # type: ignore[assignment]
    ContentPillarPatchDTO = Any  # type: ignore[assignment]
    ContentBriefPillarAssignmentCreateDTO = Any  # type: ignore[assignment]
    ContentBriefPillarAssignmentPatchDTO = Any  # type: ignore[assignment]

if TYPE_CHECKING:
    from app.models.content import ContentBrief
    from app.models.project import Project


class ContentPillar(
    TypedModelMixin[ContentPillarCreateDTO, ContentPillarPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Curated project-level pillar taxonomy."""

    __tablename__ = "content_pillars"
    __table_args__ = (
        UniqueConstraint("project_id", "slug", name="uq_content_pillars_project_slug"),
        Index("ix_content_pillars_project_status", "project_id", "status"),
    )

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(20), default="auto", nullable=False)
    locked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    project: Mapped[Project] = relationship("Project", back_populates="content_pillars")
    assignments: Mapped[list[ContentBriefPillarAssignment]] = relationship(
        "ContentBriefPillarAssignment",
        back_populates="pillar",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ContentPillar {self.project_id}:{self.slug}>"


class ContentBriefPillarAssignment(
    TypedModelMixin[
        ContentBriefPillarAssignmentCreateDTO,
        ContentBriefPillarAssignmentPatchDTO,
    ],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Primary/secondary pillar assignment for content briefs."""

    __tablename__ = "content_brief_pillar_assignments"
    __table_args__ = (
        UniqueConstraint(
            "brief_id",
            "pillar_id",
            name="uq_content_brief_pillar_assignments_brief_pillar",
        ),
        Index(
            "ix_content_brief_pillar_assignments_project_pillar",
            "project_id",
            "pillar_id",
        ),
        Index(
            "ix_content_brief_pillar_assignments_project_brief",
            "project_id",
            "brief_id",
        ),
        Index(
            "ix_content_brief_pillar_assignments_one_primary_per_brief",
            "brief_id",
            unique=True,
            postgresql_where=text("relationship_type = 'primary'"),
        ),
    )

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    brief_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("content_briefs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pillar_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("content_pillars.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    assignment_method: Mapped[str] = mapped_column(String(20), default="auto", nullable=False)

    project: Mapped[Project] = relationship("Project", back_populates="content_pillar_assignments")
    brief: Mapped[ContentBrief] = relationship("ContentBrief", back_populates="pillar_assignments")
    pillar: Mapped[ContentPillar] = relationship("ContentPillar", back_populates="assignments")

    def __repr__(self) -> str:
        return (
            f"<ContentBriefPillarAssignment brief={self.brief_id} "
            f"pillar={self.pillar_id} role={self.relationship_type}>"
        )
