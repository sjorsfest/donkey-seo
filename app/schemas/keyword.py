"""Keyword schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class KeywordBase(BaseModel):
    """Base keyword schema."""

    keyword: str
    language: str = "en"
    locale: str = "en-US"


class KeywordCreate(KeywordBase):
    """Schema for creating a keyword."""

    source: str = "manual"


class KeywordUpdate(BaseModel):
    """Schema for updating a keyword."""

    intent: str | None = None
    recommended_page_type: str | None = None
    status: str | None = None
    exclusion_reason: str | None = None


class KeywordMetrics(BaseModel):
    """Keyword metrics data."""

    search_volume: int | None
    adjusted_volume: int | None
    cpc: float | None
    competition: float | None
    difficulty: float | None
    trend_data: list[int] | None
    metrics_updated_at: datetime | None


class KeywordIntent(BaseModel):
    """Keyword intent data."""

    intent: str | None
    intent_layer: str | None
    intent_score: float | None
    intent_confidence: float | None
    recommended_page_type: str | None
    page_type_rationale: str | None
    funnel_stage: str | None
    risk_flags: list[str] | None


class KeywordResponse(BaseModel):
    """Schema for keyword response."""

    id: str
    keyword: str
    keyword_normalized: str
    language: str
    locale: str
    source: str
    status: str

    # Metrics (Step 4)
    search_volume: int | None
    adjusted_volume: int | None
    cpc: float | None
    difficulty: float | None

    # Intent (Step 5)
    intent: str | None
    intent_layer: str | None
    intent_score: float | None
    recommended_page_type: str | None
    funnel_stage: str | None

    # Priority (Step 7)
    priority_score: float | None

    # Cluster
    topic_id: str | None

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KeywordDetailResponse(KeywordResponse):
    """Detailed keyword response with all fields."""

    source_method: str | None
    raw_variants: list[str] | None
    exclusion_flags: list[str] | None

    # Full metrics
    competition: float | None
    trend_data: list[int] | None
    metrics_data_source: str | None
    metrics_updated_at: datetime | None
    metrics_confidence: float | None

    # Full intent
    intent_confidence: float | None
    page_type_rationale: str | None
    risk_flags: list[str] | None

    # Priority
    priority_factors: dict | None
    discovery_signals: dict | None

    # SERP validation
    serp_top_results: list[dict] | None
    serp_features: list[str] | None
    validated_intent: str | None
    validated_page_type: str | None

    exclusion_reason: str | None


class KeywordListResponse(BaseModel):
    """Schema for keyword list response."""

    items: list[KeywordResponse]
    total: int
    page: int
    page_size: int


class KeywordBulkUpdateRequest(BaseModel):
    """Schema for bulk keyword update."""

    keyword_ids: list[str]
    status: str | None = None
    intent: str | None = None
    topic_id: str | None = None
