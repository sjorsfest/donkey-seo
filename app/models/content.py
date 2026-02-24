"""Content brief and writer instructions models."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin
from app.models.generated_dtos import (
    ContentArticleCreateDTO,
    ContentArticlePatchDTO,
    ContentArticleVersionCreateDTO,
    ContentArticleVersionPatchDTO,
    ContentBriefCreateDTO,
    ContentBriefPatchDTO,
    WriterInstructionsCreateDTO,
    WriterInstructionsPatchDTO,
)

try:
    from app.models.generated_dtos import (
        ContentFeaturedImageCreateDTO,
        ContentFeaturedImagePatchDTO,
        PublicationWebhookDeliveryCreateDTO,
        PublicationWebhookDeliveryPatchDTO,
    )
except ImportError:  # pragma: no cover - bootstrap before DTO regeneration
    ContentFeaturedImageCreateDTO = Any  # type: ignore[assignment]
    ContentFeaturedImagePatchDTO = Any  # type: ignore[assignment]
    PublicationWebhookDeliveryCreateDTO = Any  # type: ignore[assignment]
    PublicationWebhookDeliveryPatchDTO = Any  # type: ignore[assignment]

if TYPE_CHECKING:
    from app.models.keyword import Keyword
    from app.models.project import Project
    from app.models.topic import Topic


class ContentBrief(
    TypedModelMixin[ContentBriefCreateDTO, ContentBriefPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Content brief for a topic (Step 12 output)."""

    __tablename__ = "content_briefs"
    __table_args__ = (
        Index(
            "ix_content_briefs_project_publication_date",
            "project_id",
            "proposed_publication_date",
        ),
    )

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
    proposed_publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)

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
    content_article: Mapped[ContentArticle | None] = relationship(
        "ContentArticle",
        back_populates="brief",
        uselist=False,
        cascade="all, delete-orphan",
    )
    featured_image: Mapped[ContentFeaturedImage | None] = relationship(
        "ContentFeaturedImage",
        back_populates="brief",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ContentBrief {self.primary_keyword}>"


class WriterInstructions(
    TypedModelMixin[WriterInstructionsCreateDTO, WriterInstructionsPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
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


class ContentFeaturedImage(
    TypedModelMixin[ContentFeaturedImageCreateDTO, ContentFeaturedImagePatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Rendered featured image artifact for a content brief."""

    __tablename__ = "content_featured_images"
    __table_args__ = (
        UniqueConstraint("brief_id", name="uq_content_featured_images_brief_id"),
        Index(
            "ix_content_featured_images_project_status",
            "project_id",
            "status",
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

    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False, index=True)
    title_text: Mapped[str] = mapped_column(String(500), nullable=False)
    style_variant_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    template_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    template_spec: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    byte_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)

    generation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship("Project", back_populates="content_featured_images")
    brief: Mapped[ContentBrief] = relationship("ContentBrief", back_populates="featured_image")

    def __repr__(self) -> str:
        return f"<ContentFeaturedImage brief={self.brief_id} status={self.status}>"


class ContentArticle(
    TypedModelMixin[ContentArticleCreateDTO, ContentArticlePatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Canonical article artifact generated from a content brief."""

    __tablename__ = "content_articles"

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
        unique=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    slug: Mapped[str] = mapped_column(String(500), nullable=False)
    primary_keyword: Mapped[str] = mapped_column(String(500), nullable=False)

    modular_document: Mapped[dict] = mapped_column(JSONB, nullable=False)
    rendered_html: Mapped[str] = mapped_column(Text, nullable=False)
    qa_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    status: Mapped[str] = mapped_column(String(30), default="draft", nullable=False, index=True)
    publish_status: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    published_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    current_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    generation_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    generation_temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship("Project", back_populates="content_articles")
    brief: Mapped[ContentBrief] = relationship("ContentBrief", back_populates="content_article")
    versions: Mapped[list[ContentArticleVersion]] = relationship(
        "ContentArticleVersion",
        back_populates="article",
        cascade="all, delete-orphan",
        order_by="ContentArticleVersion.version_number",
    )
    publication_webhook_deliveries: Mapped[list[PublicationWebhookDelivery]] = relationship(
        "PublicationWebhookDelivery",
        back_populates="article",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ContentArticle brief={self.brief_id} version={self.current_version}>"


class ContentArticleVersion(
    TypedModelMixin[ContentArticleVersionCreateDTO, ContentArticleVersionPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Immutable version snapshots for content articles."""

    __tablename__ = "content_article_versions"
    __table_args__ = (
        UniqueConstraint(
            "article_id",
            "version_number",
            name="uq_content_article_versions_article_version",
        ),
    )

    article_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("content_articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    slug: Mapped[str] = mapped_column(String(500), nullable=False)
    primary_keyword: Mapped[str] = mapped_column(String(500), nullable=False)

    modular_document: Mapped[dict] = mapped_column(JSONB, nullable=False)
    rendered_html: Mapped[str] = mapped_column(Text, nullable=False)
    qa_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False)

    change_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    generation_temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_by_regeneration: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    article: Mapped[ContentArticle] = relationship("ContentArticle", back_populates="versions")

    def __repr__(self) -> str:
        return (
            f"<ContentArticleVersion article={self.article_id} "
            f"version={self.version_number}>"
        )


class PublicationWebhookDelivery(
    TypedModelMixin["PublicationWebhookDeliveryCreateDTO", "PublicationWebhookDeliveryPatchDTO"],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Persistent outbound publication webhook delivery state."""

    __tablename__ = "publication_webhook_deliveries"
    __table_args__ = (
        UniqueConstraint(
            "article_id",
            "event_type",
            name="uq_publication_webhook_deliveries_article_event",
        ),
        Index(
            "ix_publication_webhook_deliveries_status_next_attempt",
            "status",
            "next_attempt_at",
        ),
    )

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    article_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("content_articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped[Project] = relationship("Project")
    article: Mapped[ContentArticle] = relationship(
        "ContentArticle",
        back_populates="publication_webhook_deliveries",
    )

    def __repr__(self) -> str:
        return (
            f"<PublicationWebhookDelivery article={self.article_id} "
            f"event={self.event_type} status={self.status}>"
        )
