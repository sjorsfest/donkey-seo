"""Brand visual context schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


def _normalize_http_url(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("URL is required")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be a valid http/https URL")
    return raw


class BrandAssetMetadata(BaseModel):
    """Stored private brand asset metadata (no signed URLs)."""

    asset_id: str
    object_key: str
    sha256: str
    mime_type: str
    byte_size: int
    width: int | None = None
    height: int | None = None
    dominant_colors: list[str] = Field(default_factory=list)
    average_luminance: float | None = None
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

    @field_validator("source_urls")
    @classmethod
    def validate_source_urls(cls, value: list[str]) -> list[str]:
        return [_normalize_http_url(item) for item in value]


class BrandAssetAddRequest(BaseModel):
    """Metadata attach request after direct client upload."""

    asset_id: str = Field(min_length=1, max_length=100)
    object_key: str = Field(min_length=1, max_length=1024)
    content_type: str = Field(min_length=1, max_length=100)
    byte_size: int = Field(ge=1)
    sha256: str = Field(min_length=8, max_length=128)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    dominant_colors: list[str] = Field(default_factory=list)
    average_luminance: float | None = Field(default=None, ge=0.0, le=1.0)
    role: str = "reference"
    role_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str) -> str:
        normalized = str(value).split(";")[0].strip().lower()
        if not normalized.startswith("image/"):
            raise ValueError("content_type must be an image MIME type")
        return normalized

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("sha256 is required")
        return normalized


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


class BrandAssetSignedUploadRequest(BaseModel):
    """Request payload to mint a signed brand-asset upload URL."""

    content_type: str = Field(min_length=1, max_length=100)

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str) -> str:
        normalized = str(value).split(";")[0].strip().lower()
        if not normalized.startswith("image/"):
            raise ValueError("content_type must be an image MIME type")
        return normalized


class BrandAssetSignedUploadResponse(BaseModel):
    """Signed upload URL details for direct brand asset uploads."""

    asset_id: str
    object_key: str
    upload_method: str = "PUT"
    upload_url: str
    expires_in_seconds: int
    required_headers: dict[str, str] = Field(default_factory=dict)


class BrandVisualStylePatchRequest(BaseModel):
    """Partial visual style/prompt contract override payload."""

    visual_style_guide: dict[str, Any] | None = None
    visual_prompt_contract: dict[str, Any] | None = None


class BrandAssetRemoveResponse(BaseModel):
    """Manual remove response payload."""

    removed_asset_id: str
    total_assets: int
    brand_assets: list[BrandAssetMetadata]
