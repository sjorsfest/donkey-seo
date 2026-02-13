"""Project style guide and brief delta models (Step 13)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin
from app.models.generated_dtos import (
    BriefDeltaCreateDTO,
    BriefDeltaPatchDTO,
    ProjectStyleGuideCreateDTO,
    ProjectStyleGuidePatchDTO,
)

if TYPE_CHECKING:
    from app.models.content import ContentBrief
    from app.models.project import Project


class ProjectStyleGuide(
    TypedModelMixin[ProjectStyleGuideCreateDTO, ProjectStyleGuidePatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Project-level style guide (generated ONCE per project).

    Contains brand voice, compliance rules, and base QA checklist
    that apply to ALL content briefs in this project.

    This approach:
    - Reduces storage by ~80% (no duplication across briefs)
    - Reduces LLM costs (generated once)
    - Ensures consistency across all briefs
    """

    __tablename__ = "project_style_guides"

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        unique=True,  # Only one per project
        nullable=False,
        index=True,
    )

    # Voice and Tone
    voice_tone_constraints: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Voice/tone rules: dos, donts, examples",
    )
    tone_examples: Mapped[list[dict] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Good/bad examples of tone",
    )

    # Compliance and Restrictions
    forbidden_claims: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Claims that should never be made",
    )
    compliance_notes: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Industry-specific compliance notes",
    )
    legal_disclaimers: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Required legal disclaimers",
    )

    # Formatting Standards
    formatting_requirements: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Heading styles, list formatting, etc.",
    )
    citation_style: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Citation format (APA, Chicago, inline, etc.)",
    )

    # Base QA Checklist
    base_qa_checklist: Mapped[list[dict] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Base QA items that apply to all content",
    )
    common_failure_modes: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Common mistakes to watch for",
    )

    # SEO Defaults
    default_internal_linking_min: Mapped[int | None] = mapped_column(
        nullable=True,
        default=3,
        comment="Minimum internal links per article",
    )
    default_external_linking_min: Mapped[int | None] = mapped_column(
        nullable=True,
        default=2,
        comment="Minimum external links per article",
    )

    # Relationships
    project: Mapped[Project] = relationship("Project")
    brief_deltas: Mapped[list[BriefDelta]] = relationship(
        "BriefDelta",
        back_populates="style_guide",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ProjectStyleGuide for project {self.project_id}>"


class BriefDelta(
    TypedModelMixin[BriefDeltaCreateDTO, BriefDeltaPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Per-brief delta from the project style guide.

    Contains ONLY the page-type specific additions that differ
    from the base ProjectStyleGuide. Much smaller than storing
    full instructions per brief.
    """

    __tablename__ = "brief_deltas"

    style_guide_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("project_style_guides.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    brief_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("content_briefs.id", ondelete="CASCADE"),
        unique=True,  # One delta per brief
        nullable=False,
        index=True,
    )

    # Page-type specific rules
    page_type_rules: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Rules specific to this page type",
    )
    must_include_sections: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Required sections for this content type",
    )

    # SEO specifics
    h1_h2_usage: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Heading usage rules for this page type",
    )
    internal_linking_minimums: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Override linking minimums for this brief",
    )

    # Schema and structured data
    schema_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Schema.org type (HowTo, FAQ, Article, etc.)",
    )
    schema_block_template: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="JSON-LD template for this content type",
    )

    # Tooling hooks
    link_placeholders: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Placeholder markers for link insertion",
    )

    # QA additions
    additional_qa_items: Mapped[list[dict] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Extra QA items specific to this brief",
    )

    # Relationships
    style_guide: Mapped[ProjectStyleGuide] = relationship(
        "ProjectStyleGuide",
        back_populates="brief_deltas",
    )
    brief: Mapped[ContentBrief] = relationship("ContentBrief")

    def __repr__(self) -> str:
        return f"<BriefDelta for brief {self.brief_id}>"
