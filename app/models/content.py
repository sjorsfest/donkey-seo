"""Content brief and writer instructions models."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.keyword import Keyword
    from app.models.project import Project
    from app.models.topic import Topic


class ContentBrief(Base, UUIDMixin, TimestampMixin):
    """Content brief for a topic (Step 12 output)."""

    __tablename__ = "content_briefs"

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_keyword_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("keywords.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Page definition
    primary_keyword: Mapped[str] = mapped_column(String(500), nullable=False)
    search_intent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    page_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    funnel_stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    working_titles: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    target_audience: Mapped[str | None] = mapped_column(Text, nullable=True)
    reader_job_to_be_done: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Content requirements
    outline: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    supporting_keywords: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    supporting_keywords_map: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    examples_required: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    faq_questions: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    recommended_schema_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # SEO + UX requirements
    internal_links_out: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    internal_links_in: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    money_page_links: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    external_sources_required: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    meta_title_guidelines: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_description_guidelines: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Acceptance criteria
    target_word_count_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_word_count_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    must_include_sections: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="draft", nullable=False)

    # Relationships
    project: Mapped[Project] = relationship("Project", back_populates="content_briefs")
    topic: Mapped[Topic] = relationship("Topic", back_populates="content_briefs")
    target_keyword: Mapped[Keyword | None] = relationship("Keyword")
    writer_instructions: Mapped[WriterInstructions | None] = relationship(
        "WriterInstructions",
        back_populates="brief",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ContentBrief {self.primary_keyword}>"


class WriterInstructions(Base, UUIDMixin, TimestampMixin):
    """Writer instructions and QA gates for a brief (Step 13 output)."""

    __tablename__ = "writer_instructions"

    brief_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("content_briefs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Writing rules
    voice_tone_constraints: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    forbidden_claims: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    compliance_notes: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    formatting_requirements: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # SEO rules
    h1_h2_usage: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    internal_linking_minimums: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    schema_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    snippet_ctr_guidelines: Mapped[str | None] = mapped_column(Text, nullable=True)

    # QA rubric
    qa_checklist: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    pass_fail_thresholds: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    common_failure_modes: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    # Tooling hooks
    link_placeholders: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    schema_block_templates: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    citation_templates: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    brief: Mapped[ContentBrief] = relationship("ContentBrief", back_populates="writer_instructions")

    def __repr__(self) -> str:
        return f"<WriterInstructions for brief {self.brief_id}>"
