"""Content brief schemas."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class OutlineSection(BaseModel):
    """Content outline section."""

    heading: str
    level: int  # H2=2, H3=3, etc.
    purpose: str | None = None
    key_points: list[str] | None = None
    supporting_keywords: list[str] | None = None


class InternalLink(BaseModel):
    """Internal link suggestion."""

    url: str
    anchor_text: str
    context: str | None = None
    priority: int = 1


class ContentBriefCreate(BaseModel):
    """Schema for creating a content brief."""

    topic_id: str
    primary_keyword: str
    working_titles: list[str] | None = None
    target_word_count_min: int | None = None
    target_word_count_max: int | None = None
    proposed_publication_date: date | None = None


class ContentBriefUpdate(BaseModel):
    """Schema for updating a content brief."""

    working_titles: list[str] | None = None
    outline: list[OutlineSection] | None = None
    supporting_keywords: list[str] | None = None
    faq_questions: list[str] | None = None
    target_word_count_min: int | None = None
    target_word_count_max: int | None = None
    proposed_publication_date: date | None = None
    status: str | None = None


class ContentBriefResponse(BaseModel):
    """Schema for content brief response."""

    id: str
    project_id: str
    topic_id: str
    primary_keyword: str
    search_intent: str | None
    page_type: str | None
    funnel_stage: str | None
    working_titles: list[str] | None
    target_audience: str | None
    proposed_publication_date: date | None
    target_word_count_min: int | None
    target_word_count_max: int | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ContentBriefDetailResponse(ContentBriefResponse):
    """Detailed content brief response."""

    reader_job_to_be_done: str | None
    outline: list[dict] | None
    supporting_keywords: list[str] | None
    supporting_keywords_map: dict | None
    examples_required: list[str] | None
    faq_questions: list[str] | None
    recommended_schema_type: str | None
    internal_links_out: list[dict] | None
    internal_links_in: list[dict] | None
    money_page_links: list[dict] | None
    external_sources_required: list[str] | None
    meta_title_guidelines: str | None
    meta_description_guidelines: str | None
    must_include_sections: list[str] | None


class ContentBriefListResponse(BaseModel):
    """Schema for content brief list response."""

    items: list[ContentBriefResponse]
    total: int
    page: int
    page_size: int


class WriterInstructionsResponse(BaseModel):
    """Schema for writer instructions response."""

    id: str
    brief_id: str
    voice_tone_constraints: dict | None
    forbidden_claims: list[str] | None
    compliance_notes: list[str] | None
    formatting_requirements: dict | None
    h1_h2_usage: dict | None
    internal_linking_minimums: dict | None
    schema_guidance: str | None
    qa_checklist: list[dict] | None
    pass_fail_thresholds: dict | None

    model_config = {"from_attributes": True}


ArticleBlockType = Literal[
    "hero",
    "summary",
    "section",
    "list",
    "comparison_table",
    "steps",
    "faq",
    "cta",
    "conclusion",
    "sources",
]
ArticleBlockSemanticTag = Literal["header", "section", "aside", "footer", "table"]


class ArticleLink(BaseModel):
    """Link metadata for block-level link placement."""

    anchor: str
    href: str


class ArticleFAQItem(BaseModel):
    """FAQ item for FAQ blocks."""

    question: str
    answer: str


class ArticleCTA(BaseModel):
    """CTA payload for CTA blocks."""

    label: str
    href: str


class ArticleBlock(BaseModel):
    """CMS-agnostic semantic block."""

    block_type: ArticleBlockType
    semantic_tag: ArticleBlockSemanticTag
    heading: str | None = None
    level: int | None = Field(default=None, ge=2, le=4)
    body: str | None = None
    items: list[str] = Field(default_factory=list)
    ordered: bool = False
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    faq_items: list[ArticleFAQItem] = Field(default_factory=list)
    cta: ArticleCTA | None = None
    links: list[ArticleLink] = Field(default_factory=list)


class RegenerateArticleRequest(BaseModel):
    """Request payload for article regeneration."""

    reason: str | None = None


class ContentArticleResponse(BaseModel):
    """Canonical content article summary."""

    id: str
    project_id: str
    brief_id: str
    title: str
    slug: str
    primary_keyword: str
    status: str
    current_version: int
    generation_model: str | None
    generated_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ContentArticleDetailResponse(ContentArticleResponse):
    """Full content article contract."""

    modular_document: dict
    rendered_html: str
    qa_report: dict | None


class ContentArticleVersionResponse(BaseModel):
    """Immutable article version response."""

    id: str
    article_id: str
    version_number: int
    title: str
    slug: str
    primary_keyword: str
    modular_document: dict
    rendered_html: str
    qa_report: dict | None
    status: str
    change_reason: str | None
    generation_model: str | None
    generation_temperature: float | None
    created_by_regeneration: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ContentArticleListResponse(BaseModel):
    """Paginated article list."""

    items: list[ContentArticleResponse]
    total: int
    page: int
    page_size: int
