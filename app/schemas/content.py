"""Content brief schemas."""

from datetime import datetime

from pydantic import BaseModel


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


class ContentBriefUpdate(BaseModel):
    """Schema for updating a content brief."""

    working_titles: list[str] | None = None
    outline: list[OutlineSection] | None = None
    supporting_keywords: list[str] | None = None
    faq_questions: list[str] | None = None
    target_word_count_min: int | None = None
    target_word_count_max: int | None = None
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
