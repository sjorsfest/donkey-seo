"""Keyword and seed topic models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.topic import Topic


KeywordSource = Literal["seed", "expansion", "related", "questions", "serp", "gsc"]
KeywordIntent = Literal["informational", "navigational", "commercial", "transactional"]
PageType = Literal["blog", "guide", "comparison", "alternatives", "tool", "glossary", "landing", "product", "category"]
FunnelStage = Literal["tofu", "mofu", "bofu"]
KeywordStatus = Literal["active", "excluded", "merged"]


class SeedTopic(Base, UUIDMixin, TimestampMixin):
    """Seed topic/pillar generated in Step 2."""

    __tablename__ = "seed_topics"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    pillar_type: Mapped[str] = mapped_column(String(50), default="main_pillar", nullable=False)

    # ICP + product tie-in
    icp_relevance: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_tie_in: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Content types
    intended_content_types: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    coverage_intent: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Scoring
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    project: Mapped[Project] = relationship("Project", back_populates="seed_topics")
    keywords: Mapped[list[Keyword]] = relationship(
        "Keyword",
        back_populates="seed_topic",
        foreign_keys="Keyword.seed_topic_id",
    )

    def __repr__(self) -> str:
        return f"<SeedTopic {self.name}>"


class Keyword(Base, UUIDMixin, TimestampMixin):
    """Keyword with metrics, intent, and priority data."""

    __tablename__ = "keywords"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seed_topic_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("seed_topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    topic_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    parent_keyword_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("keywords.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Core data
    keyword: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    keyword_normalized: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    locale: Mapped[str] = mapped_column(String(10), default="en-US", nullable=False)

    # Step 3: Source
    source: Mapped[str] = mapped_column(String(50), default="seed", nullable=False)
    source_method: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_variants: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    exclusion_flags: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Step 4: Metrics
    search_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_volume_period: Mapped[str | None] = mapped_column(String(50), nullable=True)
    cpc: Mapped[float | None] = mapped_column(Float, nullable=True)
    competition: Mapped[float | None] = mapped_column(Float, nullable=True)
    difficulty: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend_data: Mapped[list[int] | None] = mapped_column(JSONB, nullable=True)
    metrics_data_source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metrics_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Step 5: Intent
    intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    intent_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_page_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    page_type_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    funnel_stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    risk_flags: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Step 7: Prioritization
    priority_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    priority_factors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Step 8: SERP validation
    serp_top_results: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    serp_features: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    validated_intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    validated_page_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    format_requirements: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    serp_mismatch_flags: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    exclusion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped[Project] = relationship("Project", back_populates="keywords")
    seed_topic: Mapped[SeedTopic | None] = relationship(
        "SeedTopic",
        back_populates="keywords",
        foreign_keys=[seed_topic_id],
    )
    topic: Mapped[Topic | None] = relationship(
        "Topic",
        back_populates="keywords",
        foreign_keys=[topic_id],
    )
    parent_keyword: Mapped[Keyword | None] = relationship(
        "Keyword",
        remote_side="Keyword.id",
        foreign_keys=[parent_keyword_id],
    )

    def __repr__(self) -> str:
        return f"<Keyword {self.keyword}>"
