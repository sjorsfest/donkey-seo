"""Topic cluster model (Step 6 output)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin
from app.models.generated_dtos import TopicCreateDTO, TopicPatchDTO

if TYPE_CHECKING:
    from app.models.content import ContentBrief
    from app.models.keyword import Keyword
    from app.models.project import Project


class Topic(TypedModelMixin[TopicCreateDTO, TopicPatchDTO], Base, UUIDMixin, TimestampMixin):
    """Clustered topic (Step 6 output)."""

    __tablename__ = "topics"

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_topic_id: Mapped[str | None] = mapped_column(
        StringUUID(),
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
    )
    pillar_seed_topic_id: Mapped[str | None] = mapped_column(
        StringUUID(),
        ForeignKey("seed_topics.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Clustering data
    cluster_method: Mapped[str | None] = mapped_column(String(50), nullable=True)
    cluster_coherence: Mapped[float | None] = mapped_column(Float, nullable=True)
    primary_keyword_id: Mapped[str | None] = mapped_column(
        StringUUID(),
        nullable=True,
    )

    # Dominant characteristics
    dominant_intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    dominant_page_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    funnel_stage: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Aggregated metrics
    total_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    adjusted_volume_sum: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_difficulty: Mapped[float | None] = mapped_column(Float, nullable=True)
    keyword_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_demand: Mapped[int | None] = mapped_column(Integer, nullable=True)
    market_mode: Mapped[str | None] = mapped_column(String(50), nullable=True)
    demand_fragmentation_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    serp_servedness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    serp_competitor_density: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Step 7: Priority
    priority_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    priority_factors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    recommended_url_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    recommended_publish_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_money_pages: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    expected_role: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Step 10: Cannibalization
    cannibalization_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    overlapping_topic_ids: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Notes
    cluster_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped[Project] = relationship("Project", back_populates="topics")
    keywords: Mapped[list[Keyword]] = relationship(
        "Keyword",
        back_populates="topic",
        foreign_keys="Keyword.topic_id",
    )
    children: Mapped[list[Topic]] = relationship(
        "Topic",
        back_populates="parent",
        foreign_keys=[parent_topic_id],
    )
    parent: Mapped[Topic | None] = relationship(
        "Topic",
        back_populates="children",
        remote_side="Topic.id",
        foreign_keys=[parent_topic_id],
    )
    content_briefs: Mapped[list[ContentBrief]] = relationship(
        "ContentBrief",
        back_populates="topic",
    )

    def __repr__(self) -> str:
        return f"<Topic {self.name}>"

    @property
    def fit_score(self) -> float | None:
        """Convenience accessor for fit score stored in priority_factors."""
        if not self.priority_factors:
            return None
        value = self.priority_factors.get("fit_score")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def fit_tier(self) -> str | None:
        """Convenience accessor for fit tier stored in priority_factors."""
        if not self.priority_factors:
            return None
        value = self.priority_factors.get("fit_tier")
        return value if isinstance(value, str) else None

    @property
    def fit_reasons(self) -> list[str] | None:
        """Convenience accessor for fit reasons stored in priority_factors."""
        if not self.priority_factors:
            return None
        value = self.priority_factors.get("fit_reasons")
        if isinstance(value, list):
            return [str(v) for v in value]
        return None

    @property
    def fit_threshold_used(self) -> float | None:
        """Convenience accessor for the threshold applied during fit gating."""
        if not self.priority_factors:
            return None
        value = self.priority_factors.get("fit_threshold_used")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
