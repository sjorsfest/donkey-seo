"""Project model for keyword research projects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin
from app.models.generated_dtos import ProjectCreateDTO, ProjectPatchDTO

if TYPE_CHECKING:
    from app.models.brand import BrandProfile
    from app.models.discovery_learning import DiscoveryIterationLearning
    from app.models.content import ContentArticle, ContentBrief
    from app.models.discovery_snapshot import DiscoveryTopicSnapshot
    from app.models.keyword import Keyword, SeedTopic
    from app.models.pipeline import PipelineRun
    from app.models.topic import Topic
    from app.models.user import User


ProjectStatus = Literal["created", "running", "paused", "completed", "failed"]
SiteMaturity = Literal["new", "mid", "strong"]


class Project(TypedModelMixin[ProjectCreateDTO, ProjectPatchDTO], Base, UUIDMixin, TimestampMixin):
    """Keyword research project."""

    __tablename__ = "projects"

    # Owner
    user_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Targeting
    primary_language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    primary_locale: Mapped[str] = mapped_column(String(10), default="en-US", nullable=False)
    secondary_locales: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Site maturity
    site_maturity: Mapped[str | None] = mapped_column(String(20), nullable=True)
    maturity_signals: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Goals
    primary_goal: Mapped[str | None] = mapped_column(String(100), nullable=True)
    secondary_goals: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    primary_cta: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Constraints
    topic_boundaries: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    compliance_flags: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Operational settings
    api_budget_caps: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    caching_ttls: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    enabled_steps: Mapped[list[int] | None] = mapped_column(JSONB, nullable=True)
    skip_steps: Mapped[list[int] | None] = mapped_column(JSONB, nullable=True)

    # Status tracking
    current_step: Mapped[int] = mapped_column(default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="created", nullable=False)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="projects")
    brand_profile: Mapped[BrandProfile | None] = relationship(
        "BrandProfile",
        back_populates="project",
        uselist=False,
        cascade="all, delete-orphan",
    )
    seed_topics: Mapped[list[SeedTopic]] = relationship(
        "SeedTopic",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    keywords: Mapped[list[Keyword]] = relationship(
        "Keyword",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    topics: Mapped[list[Topic]] = relationship(
        "Topic",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    content_briefs: Mapped[list[ContentBrief]] = relationship(
        "ContentBrief",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    content_articles: Mapped[list[ContentArticle]] = relationship(
        "ContentArticle",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    pipeline_runs: Mapped[list[PipelineRun]] = relationship(
        "PipelineRun",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    discovery_topic_snapshots: Mapped[list[DiscoveryTopicSnapshot]] = relationship(
        "DiscoveryTopicSnapshot",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    discovery_iteration_learnings: Mapped[list[DiscoveryIterationLearning]] = relationship(
        "DiscoveryIterationLearning",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Project {self.name} ({self.domain})>"
