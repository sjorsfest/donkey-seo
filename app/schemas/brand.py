"""Brand visual context schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BrandAssetMetadata(BaseModel):
    """Stored private brand asset metadata (no signed URLs)."""

    asset_id: str
    object_key: str
    sha256: str
    mime_type: str
    byte_size: int
    width: int | None = None
    height: int | None = None
    role: str
    role_confidence: float
    source_url: str
    origin: str
    ingested_at: str


class BrandProductServiceMetadata(BaseModel):
    """Structured product/service metadata extracted in Step 1."""

    name: str
    description: str | None = None
    category: str | None = None
    target_audience: str | None = None
    core_benefits: list[str] = Field(default_factory=list)


class BrandSuggestedICPNiche(BaseModel):
    """Suggested ICP niche returned from Step 1 recommender."""

    niche_name: str
    target_roles: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    company_sizes: list[str] = Field(default_factory=list)
    primary_pains: list[str] = Field(default_factory=list)
    desired_outcomes: list[str] = Field(default_factory=list)
    likely_objections: list[str] = Field(default_factory=list)
    why_good_fit: str | None = None


class BrandVisualContextResponse(BaseModel):
    """Brand visual context used for image generation."""

    project_id: str
    company_name: str | None = None
    tagline: str | None = None
    products_services: list[BrandProductServiceMetadata] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    suggested_icp_niches: list[BrandSuggestedICPNiche] = Field(default_factory=list)
    extraction_confidence: float | None = None
    brand_assets: list[BrandAssetMetadata] = Field(default_factory=list)
    visual_style_guide: dict[str, Any] = Field(default_factory=dict)
    visual_prompt_contract: dict[str, Any] = Field(default_factory=dict)
    visual_extraction_confidence: float | None = None
    visual_last_synced_at: datetime | None = None


class BrandAssetIngestRequest(BaseModel):
    """Manual URL-based asset ingestion request."""

    source_urls: list[str] = Field(min_length=1)
    role: str = "reference"


class BrandAssetIngestResponse(BaseModel):
    """Manual ingestion response payload."""

    ingested_count: int
    total_assets: int
    brand_assets: list[BrandAssetMetadata]


class BrandAssetSignedReadUrlResponse(BaseModel):
    """Signed private object read URL response."""

    asset_id: str
    object_key: str
    expires_in_seconds: int
    signed_url: str


class BrandVisualStylePatchRequest(BaseModel):
    """Partial visual style/prompt contract override payload."""

    visual_style_guide: dict[str, Any] | None = None
    visual_prompt_contract: dict[str, Any] | None = None
