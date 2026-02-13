"""Keyword-related write adapters."""

from __future__ import annotations

from app.models.generated_dtos import (
    KeywordCreateDTO,
    KeywordPatchDTO,
    SeedTopicCreateDTO,
    SeedTopicPatchDTO,
)
from app.models.keyword import Keyword, SeedTopic
from app.persistence.typed.adapters._base import BaseWriteAdapter

KEYWORD_PATCH_ALLOWLIST = {
    "seed_topic_id",
    "topic_id",
    "parent_keyword_id",
    "keyword",
    "keyword_normalized",
    "language",
    "locale",
    "source",
    "source_method",
    "raw_variants",
    "exclusion_flags",
    "search_volume",
    "search_volume_period",
    "cpc",
    "competition",
    "difficulty",
    "trend_data",
    "metrics_data_source",
    "metrics_updated_at",
    "metrics_confidence",
    "intent",
    "intent_confidence",
    "recommended_page_type",
    "page_type_rationale",
    "funnel_stage",
    "risk_flags",
    "priority_score",
    "priority_factors",
    "serp_top_results",
    "serp_features",
    "validated_intent",
    "validated_page_type",
    "format_requirements",
    "serp_mismatch_flags",
    "status",
    "exclusion_reason",
}

SEED_TOPIC_PATCH_ALLOWLIST = {
    "name",
    "description",
    "pillar_type",
    "icp_relevance",
    "product_tie_in",
    "intended_content_types",
    "coverage_intent",
    "relevance_score",
}

_KEYWORD_ADAPTER = BaseWriteAdapter[Keyword, KeywordCreateDTO, KeywordPatchDTO](
    model_cls=Keyword,
    patch_allowlist=KEYWORD_PATCH_ALLOWLIST,
)

_SEED_TOPIC_ADAPTER = BaseWriteAdapter[SeedTopic, SeedTopicCreateDTO, SeedTopicPatchDTO](
    model_cls=SeedTopic,
    patch_allowlist=SEED_TOPIC_PATCH_ALLOWLIST,
)


def register() -> None:
    """Register keyword adapters."""
    from app.persistence.typed.registry import register_adapter

    register_adapter(Keyword, _KEYWORD_ADAPTER)
    register_adapter(SeedTopic, _SEED_TOPIC_ADAPTER)
