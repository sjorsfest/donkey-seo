"""Topic schemas."""

from datetime import datetime

from pydantic import BaseModel


class TopicCreate(BaseModel):
    """Schema for creating a topic."""

    name: str
    description: str | None = None
    parent_topic_id: str | None = None


class TopicUpdate(BaseModel):
    """Schema for updating a topic."""

    name: str | None = None
    description: str | None = None
    parent_topic_id: str | None = None
    priority_rank: int | None = None


class TopicResponse(BaseModel):
    """Schema for topic response."""

    id: str
    name: str
    description: str | None
    parent_topic_id: str | None

    # Clustering data
    cluster_method: str | None
    cluster_coherence: float | None
    primary_keyword_id: str | None

    # Characteristics
    dominant_intent: str | None
    dominant_page_type: str | None
    funnel_stage: str | None

    # Aggregated metrics
    total_volume: int | None
    adjusted_volume_sum: int | None
    avg_difficulty: float | None
    keyword_count: int
    market_mode: str | None
    demand_fragmentation_index: float | None
    serp_servedness_score: float | None
    serp_competitor_density: float | None

    # Priority
    priority_rank: int | None
    priority_score: float | None
    deterministic_priority_score: float | None
    final_priority_score: float | None
    dynamic_fit_score: float | None
    dynamic_opportunity_score: float | None
    expected_role: str | None
    fit_score: float | None
    brand_fit_score: float | None
    opportunity_score: float | None
    fit_tier: str | None
    fit_threshold_primary: float | None
    fit_threshold_secondary: float | None
    llm_rerank_delta: float | None
    llm_fit_adjustment: float | None
    llm_tier_recommendation: str | None
    hard_exclusion_reason: str | None
    final_cut_reason_code: str | None
    serp_intent_confidence: float | None
    serp_evidence_keyword_id: str | None
    serp_evidence_source: str | None
    serp_evidence_keyword_count: int | None

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TopicDetailResponse(TopicResponse):
    """Detailed topic response."""

    prioritization_diagnostics: dict | None
    recommended_url_type: str | None
    recommended_publish_order: int | None
    target_money_pages: list[str] | None
    cannibalization_risk: float | None
    overlapping_topic_ids: list[str] | None
    cluster_notes: str | None


class TopicListResponse(BaseModel):
    """Schema for topic list response."""

    items: list[TopicResponse]
    total: int
    page: int
    page_size: int


class TopicHierarchyResponse(BaseModel):
    """Schema for topic hierarchy response."""

    id: str
    name: str
    keyword_count: int
    priority_rank: int | None
    children: list["TopicHierarchyResponse"]

    model_config = {"from_attributes": True}


class TopicMergeRequest(BaseModel):
    """Schema for merging topics."""

    source_topic_ids: list[str]
    target_name: str
    target_description: str | None = None


class TopicSplitRequest(BaseModel):
    """Schema for splitting a topic."""

    keyword_groups: list[list[str]]
    new_topic_names: list[str]
