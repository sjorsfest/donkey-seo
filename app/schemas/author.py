"""Author API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


def _normalize_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("URL is required")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be a valid http/https URL")
    return raw


class AuthorCreate(BaseModel):
    """Schema for creating an author profile."""

    name: str = Field(min_length=1, max_length=255)
    bio: str | None = None
    social_urls: dict[str, str] | None = None
    basic_info: dict[str, Any] | None = None
    profile_image_source_url: str | None = None

    @field_validator("social_urls")
    @classmethod
    def validate_social_urls(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        normalized: dict[str, str] = {}
        for key, url in value.items():
            platform = str(key).strip().lower()
            if not platform:
                continue
            normalized[platform] = _normalize_url(str(url))
        return normalized or None

    @field_validator("profile_image_source_url")
    @classmethod
    def validate_profile_image_source_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_url(value)


class AuthorUpdate(BaseModel):
    """Schema for patching an author profile."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    bio: str | None = None
    social_urls: dict[str, str] | None = None
    basic_info: dict[str, Any] | None = None
    profile_image_source_url: str | None = None

    @field_validator("social_urls")
    @classmethod
    def validate_social_urls(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        normalized: dict[str, str] = {}
        for key, url in value.items():
            platform = str(key).strip().lower()
            if not platform:
                continue
            normalized[platform] = _normalize_url(str(url))
        return normalized or None

    @field_validator("profile_image_source_url")
    @classmethod
    def validate_profile_image_source_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_url(value)


class AuthorResponse(BaseModel):
    """Schema for author profile response."""

    id: str
    project_id: str
    name: str
    bio: str | None
    social_urls: dict[str, str] | None
    basic_info: dict[str, Any] | None
    profile_image_source_url: str | None
    profile_image_object_key: str | None
    profile_image_mime_type: str | None
    profile_image_width: int | None
    profile_image_height: int | None
    profile_image_byte_size: int | None
    profile_image_sha256: str | None
    profile_image_signed_url: str | None
    created_at: datetime
    updated_at: datetime


class AuthorListResponse(BaseModel):
    """Schema for author list response."""

    items: list[AuthorResponse]
    total: int
