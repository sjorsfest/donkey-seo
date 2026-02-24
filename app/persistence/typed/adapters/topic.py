"""Topic write adapters."""

from __future__ import annotations

from app.models.generated_dtos import TopicCreateDTO, TopicPatchDTO
from app.models.topic import Topic
from app.persistence.typed.adapters._base import BaseWriteAdapter

TOPIC_PATCH_ALLOWLIST = {
    "name",
    "description",
    "parent_topic_id",
    "pillar_seed_topic_id",
    "cluster_method",
    "cluster_coherence",
    "primary_keyword_id",
    "dominant_intent",
    "dominant_page_type",
    "funnel_stage",
    "total_volume",
    "adjusted_volume_sum",
    "avg_difficulty",
    "keyword_count",
    "estimated_demand",
    "market_mode",
    "demand_fragmentation_index",
    "serp_servedness_score",
    "serp_competitor_density",
    "priority_rank",
    "priority_score",
    "fit_tier",
    "fit_score",
    "brand_fit_score",
    "opportunity_score",
    "dynamic_fit_score",
    "dynamic_opportunity_score",
    "deterministic_priority_score",
    "final_priority_score",
    "llm_rerank_delta",
    "llm_fit_adjustment",
    "llm_tier_recommendation",
    "fit_threshold_primary",
    "fit_threshold_secondary",
    "hard_exclusion_reason",
    "final_cut_reason_code",
    "serp_intent_confidence",
    "serp_evidence_keyword_id",
    "serp_evidence_source",
    "serp_evidence_keyword_count",
    "prioritization_diagnostics",
    "recommended_url_type",
    "recommended_publish_order",
    "target_money_pages",
    "expected_role",
    "cannibalization_risk",
    "overlapping_topic_ids",
    "cluster_notes",
}

_TOPIC_ADAPTER = BaseWriteAdapter[Topic, TopicCreateDTO, TopicPatchDTO](
    model_cls=Topic,
    patch_allowlist=TOPIC_PATCH_ALLOWLIST,
)


def register() -> None:
    """Register topic adapter."""
    from app.persistence.typed.registry import register_adapter

    register_adapter(Topic, _TOPIC_ADAPTER)
