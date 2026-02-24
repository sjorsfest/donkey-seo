"""Schema-level tests for topics API V2 typed prioritization fields."""

from datetime import UTC, datetime
from types import SimpleNamespace

from app.schemas.topic import TopicDetailResponse, TopicResponse


def test_topic_response_exposes_typed_prioritization_fields() -> None:
    fields = TopicResponse.model_fields

    assert "fit_tier" in fields
    assert "fit_score" in fields
    assert "brand_fit_score" in fields
    assert "opportunity_score" in fields
    assert "dynamic_fit_score" in fields
    assert "dynamic_opportunity_score" in fields
    assert "deterministic_priority_score" in fields
    assert "final_priority_score" in fields
    assert "llm_rerank_delta" in fields
    assert "llm_fit_adjustment" in fields
    assert "llm_tier_recommendation" in fields
    assert "hard_exclusion_reason" in fields
    assert "final_cut_reason_code" in fields
    assert "serp_intent_confidence" in fields
    assert "serp_evidence_keyword_id" in fields
    assert "serp_evidence_source" in fields
    assert "serp_evidence_keyword_count" in fields
    assert "priority_factors" not in fields


def test_topic_detail_response_uses_prioritization_diagnostics_not_priority_factors() -> None:
    fields = TopicDetailResponse.model_fields

    assert "prioritization_diagnostics" in fields
    assert "priority_factors" not in fields


def test_topic_response_model_validate_accepts_typed_values() -> None:
    now = datetime.now(UTC)
    topic = SimpleNamespace(
        id="topic-1",
        name="Helpdesk Alternatives",
        description="Commercial comparison topic",
        parent_topic_id=None,
        cluster_method="llm_refined",
        cluster_coherence=0.81,
        primary_keyword_id="kw-1",
        dominant_intent="commercial",
        dominant_page_type="comparison",
        funnel_stage="mofu",
        total_volume=4200,
        adjusted_volume_sum=3900,
        avg_difficulty=26.0,
        keyword_count=12,
        market_mode="established_category",
        demand_fragmentation_index=0.06,
        serp_servedness_score=0.42,
        serp_competitor_density=0.35,
        priority_rank=1,
        priority_score=78.6,
        deterministic_priority_score=76.8,
        final_priority_score=78.6,
        dynamic_fit_score=0.74,
        dynamic_opportunity_score=0.65,
        expected_role="revenue_driver",
        fit_score=0.79,
        brand_fit_score=0.79,
        opportunity_score=0.64,
        fit_tier="primary",
        fit_threshold_primary=0.70,
        fit_threshold_secondary=0.56,
        llm_rerank_delta=1.2,
        llm_fit_adjustment=0.08,
        llm_tier_recommendation="primary",
        hard_exclusion_reason=None,
        final_cut_reason_code="llm_primary",
        serp_intent_confidence=0.92,
        serp_evidence_keyword_id="kw-1",
        serp_evidence_source="primary",
        serp_evidence_keyword_count=3,
        created_at=now,
        updated_at=now,
    )

    response = TopicResponse.model_validate(topic)

    assert response.fit_tier == "primary"
    assert response.final_priority_score == 78.6
    assert response.serp_evidence_source == "primary"
