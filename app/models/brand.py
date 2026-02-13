"""Brand profile model for extracted brand information."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.project import Project


class BrandProfile(Base, UUIDMixin, TimestampMixin):
    """Extracted brand profile from website scraping (Step 1 output)."""

    __tablename__ = "brand_profiles"

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Raw scraped content
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_pages: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Offering map
    products_services: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    money_pages: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    # Positioning
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tagline: Mapped[str | None] = mapped_column(Text, nullable=True)
    unique_value_props: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    differentiators: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    competitor_positioning: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    # Audience (ICP-lite)
    target_roles: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    target_industries: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    company_sizes: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    primary_pains: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    desired_outcomes: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    objections: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Voice & claims
    tone_attributes: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    allowed_claims: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    restricted_claims: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Topic boundaries
    in_scope_topics: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    out_of_scope_topics: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Extraction metadata
    extraction_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    project: Mapped[Project] = relationship("Project", back_populates="brand_profile")

    def __repr__(self) -> str:
        return f"<BrandProfile for project {self.project_id}>"
