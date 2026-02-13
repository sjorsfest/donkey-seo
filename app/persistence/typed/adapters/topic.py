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
    "avg_difficulty",
    "keyword_count",
    "estimated_demand",
    "priority_rank",
    "priority_score",
    "priority_factors",
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
