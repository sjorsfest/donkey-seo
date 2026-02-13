"""Project-related write adapters."""

from __future__ import annotations

from app.models.brand import BrandProfile
from app.models.generated_dtos import (
    BrandProfileCreateDTO,
    BrandProfilePatchDTO,
    ProjectCreateDTO,
    ProjectPatchDTO,
)
from app.models.project import Project
from app.persistence.typed.adapters._base import BaseWriteAdapter

PROJECT_PATCH_ALLOWLIST = {
    "name",
    "domain",
    "description",
    "primary_language",
    "primary_locale",
    "secondary_locales",
    "site_maturity",
    "maturity_signals",
    "primary_goal",
    "secondary_goals",
    "primary_cta",
    "topic_boundaries",
    "compliance_flags",
    "api_budget_caps",
    "caching_ttls",
    "enabled_steps",
    "skip_steps",
    "current_step",
    "status",
}

BRAND_PROFILE_PATCH_ALLOWLIST = {
    "raw_content",
    "source_pages",
    "products_services",
    "money_pages",
    "company_name",
    "tagline",
    "unique_value_props",
    "differentiators",
    "competitor_positioning",
    "target_roles",
    "target_industries",
    "company_sizes",
    "primary_pains",
    "desired_outcomes",
    "objections",
    "tone_attributes",
    "allowed_claims",
    "restricted_claims",
    "in_scope_topics",
    "out_of_scope_topics",
    "extraction_model",
    "extraction_confidence",
}

_PROJECT_ADAPTER = BaseWriteAdapter[Project, ProjectCreateDTO, ProjectPatchDTO](
    model_cls=Project,
    patch_allowlist=PROJECT_PATCH_ALLOWLIST,
)

_BRAND_PROFILE_ADAPTER = BaseWriteAdapter[
    BrandProfile,
    BrandProfileCreateDTO,
    BrandProfilePatchDTO,
](
    model_cls=BrandProfile,
    patch_allowlist=BRAND_PROFILE_PATCH_ALLOWLIST,
)


def register() -> None:
    """Register project adapters."""
    from app.persistence.typed.registry import register_adapter

    register_adapter(Project, _PROJECT_ADAPTER)
    register_adapter(BrandProfile, _BRAND_PROFILE_ADAPTER)
