"""Content-related write adapters."""

from __future__ import annotations

from app.models.content import (
    ContentArticle,
    ContentArticleVersion,
    ContentBrief,
    WriterInstructions,
)
from app.models.generated_dtos import (
    BriefDeltaCreateDTO,
    BriefDeltaPatchDTO,
    ContentArticleCreateDTO,
    ContentArticlePatchDTO,
    ContentArticleVersionCreateDTO,
    ContentArticleVersionPatchDTO,
    ContentBriefCreateDTO,
    ContentBriefPatchDTO,
    ProjectStyleGuideCreateDTO,
    ProjectStyleGuidePatchDTO,
    WriterInstructionsCreateDTO,
    WriterInstructionsPatchDTO,
)
from app.models.style_guide import BriefDelta, ProjectStyleGuide
from app.persistence.typed.adapters._base import BaseWriteAdapter

CONTENT_BRIEF_PATCH_ALLOWLIST = {
    "target_keyword_id",
    "primary_keyword",
    "search_intent",
    "page_type",
    "funnel_stage",
    "working_titles",
    "target_audience",
    "reader_job_to_be_done",
    "outline",
    "supporting_keywords",
    "supporting_keywords_map",
    "examples_required",
    "faq_questions",
    "recommended_schema_type",
    "internal_links_out",
    "internal_links_in",
    "money_page_links",
    "external_sources_required",
    "meta_title_guidelines",
    "meta_description_guidelines",
    "target_word_count_min",
    "target_word_count_max",
    "must_include_sections",
    "proposed_publication_date",
    "status",
}

WRITER_INSTRUCTIONS_PATCH_ALLOWLIST = {
    "voice_tone_constraints",
    "forbidden_claims",
    "compliance_notes",
    "formatting_requirements",
    "h1_h2_usage",
    "internal_linking_minimums",
    "schema_guidance",
    "snippet_ctr_guidelines",
    "qa_checklist",
    "pass_fail_thresholds",
    "common_failure_modes",
    "link_placeholders",
    "schema_block_templates",
    "citation_templates",
}

PROJECT_STYLE_GUIDE_PATCH_ALLOWLIST = {
    "voice_tone_constraints",
    "tone_examples",
    "forbidden_claims",
    "compliance_notes",
    "legal_disclaimers",
    "formatting_requirements",
    "citation_style",
    "base_qa_checklist",
    "common_failure_modes",
    "default_internal_linking_min",
    "default_external_linking_min",
}

BRIEF_DELTA_PATCH_ALLOWLIST = {
    "page_type_rules",
    "must_include_sections",
    "h1_h2_usage",
    "internal_linking_minimums",
    "schema_type",
    "schema_block_template",
    "link_placeholders",
    "additional_qa_items",
}

CONTENT_ARTICLE_PATCH_ALLOWLIST = {
    "title",
    "slug",
    "primary_keyword",
    "modular_document",
    "rendered_html",
    "qa_report",
    "status",
    "current_version",
    "generation_model",
    "generation_temperature",
    "generated_at",
}

CONTENT_ARTICLE_VERSION_PATCH_ALLOWLIST = {
    "title",
    "slug",
    "primary_keyword",
    "modular_document",
    "rendered_html",
    "qa_report",
    "status",
    "change_reason",
    "generation_model",
    "generation_temperature",
    "created_by_regeneration",
}

_CONTENT_BRIEF_ADAPTER = BaseWriteAdapter[
    ContentBrief,
    ContentBriefCreateDTO,
    ContentBriefPatchDTO,
](
    model_cls=ContentBrief,
    patch_allowlist=CONTENT_BRIEF_PATCH_ALLOWLIST,
)

_WRITER_INSTRUCTIONS_ADAPTER = BaseWriteAdapter[
    WriterInstructions,
    WriterInstructionsCreateDTO,
    WriterInstructionsPatchDTO,
](
    model_cls=WriterInstructions,
    patch_allowlist=WRITER_INSTRUCTIONS_PATCH_ALLOWLIST,
)

_PROJECT_STYLE_GUIDE_ADAPTER = BaseWriteAdapter[
    ProjectStyleGuide,
    ProjectStyleGuideCreateDTO,
    ProjectStyleGuidePatchDTO,
](
    model_cls=ProjectStyleGuide,
    patch_allowlist=PROJECT_STYLE_GUIDE_PATCH_ALLOWLIST,
)

_BRIEF_DELTA_ADAPTER = BaseWriteAdapter[
    BriefDelta,
    BriefDeltaCreateDTO,
    BriefDeltaPatchDTO,
](
    model_cls=BriefDelta,
    patch_allowlist=BRIEF_DELTA_PATCH_ALLOWLIST,
)

_CONTENT_ARTICLE_ADAPTER = BaseWriteAdapter[
    ContentArticle,
    ContentArticleCreateDTO,
    ContentArticlePatchDTO,
](
    model_cls=ContentArticle,
    patch_allowlist=CONTENT_ARTICLE_PATCH_ALLOWLIST,
)

_CONTENT_ARTICLE_VERSION_ADAPTER = BaseWriteAdapter[
    ContentArticleVersion,
    ContentArticleVersionCreateDTO,
    ContentArticleVersionPatchDTO,
](
    model_cls=ContentArticleVersion,
    patch_allowlist=CONTENT_ARTICLE_VERSION_PATCH_ALLOWLIST,
)


def register() -> None:
    """Register content adapters."""
    from app.persistence.typed.registry import register_adapter

    register_adapter(ContentBrief, _CONTENT_BRIEF_ADAPTER)
    register_adapter(WriterInstructions, _WRITER_INSTRUCTIONS_ADAPTER)
    register_adapter(ProjectStyleGuide, _PROJECT_STYLE_GUIDE_ADAPTER)
    register_adapter(BriefDelta, _BRIEF_DELTA_ADAPTER)
    register_adapter(ContentArticle, _CONTENT_ARTICLE_ADAPTER)
    register_adapter(ContentArticleVersion, _CONTENT_ARTICLE_VERSION_ADAPTER)
