"""Shared helpers for setup brand steps."""

from __future__ import annotations

from typing import Any

TARGET_AUDIENCE_KEYS = (
    "target_roles",
    "target_industries",
    "company_sizes",
    "primary_pains",
    "desired_outcomes",
    "objections",
)

PROMPT_CONTRACT_REQUIRED_VARIABLES = [
    "article_topic",
    "audience",
    "intent",
    "visual_goal",
    "brand_voice",
    "asset_refs",
]


def normalize_string_list(items: list[str]) -> list[str]:
    """Trim, deduplicate, and preserve order for text list fields."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in items:
        value = " ".join(str(item).strip().split())
        if not value:
            continue

        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)

    return normalized


def build_extracted_target_audience(raw: Any) -> dict[str, list[str]]:
    """Convert nested target-audience model payload into primitive dict lists."""
    return {
        "target_roles": [str(item) for item in list(getattr(raw, "target_roles", []) or [])],
        "target_industries": [
            str(item)
            for item in list(getattr(raw, "target_industries", []) or [])
        ],
        "company_sizes": [str(item) for item in list(getattr(raw, "company_sizes", []) or [])],
        "primary_pains": [str(item) for item in list(getattr(raw, "primary_pains", []) or [])],
        "desired_outcomes": [
            str(item)
            for item in list(getattr(raw, "desired_outcomes", []) or [])
        ],
        "objections": [
            str(item) for item in list(getattr(raw, "common_objections", []) or [])
        ],
    }


def extract_niche_audience_lists(
    suggested_icp_niches: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Flatten list-style fields from niche recommendations."""
    extracted: dict[str, list[str]] = {
        key: []
        for key in TARGET_AUDIENCE_KEYS
    }
    objections_aliases = ("objections", "likely_objections", "common_objections")

    for niche in suggested_icp_niches:
        for key in (
            "target_roles",
            "target_industries",
            "company_sizes",
            "primary_pains",
            "desired_outcomes",
        ):
            values = niche.get(key, [])
            if isinstance(values, list):
                extracted[key].extend(str(value) for value in values)

        for alias in objections_aliases:
            if alias not in niche:
                continue
            objections = niche.get(alias, [])
            if isinstance(objections, list):
                extracted["objections"].extend(str(value) for value in objections)
                break

    return extracted


def merge_target_audience(
    *,
    extracted_target_audience: dict[str, list[str]],
    suggested_icp_niches: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Merge website ICP extraction with LLM-suggested ICP niches."""
    merged: dict[str, list[str]] = {
        key: normalize_string_list(extracted_target_audience.get(key, []))
        for key in TARGET_AUDIENCE_KEYS
    }
    suggested_lists = extract_niche_audience_lists(suggested_icp_niches)
    max_items = {
        "target_roles": 20,
        "target_industries": 20,
        "company_sizes": 12,
        "primary_pains": 20,
        "desired_outcomes": 20,
        "objections": 20,
    }

    for key in TARGET_AUDIENCE_KEYS:
        combined = [*merged.get(key, []), *suggested_lists.get(key, [])]
        merged[key] = normalize_string_list(combined)[: max_items[key]]

    return merged


def is_low_quality_icon_candidate(candidate: dict[str, Any]) -> bool:
    """Skip icon assets that are typically too small for visual guide generation."""
    role = str(candidate.get("role") or "").strip().lower()
    origin = str(candidate.get("origin") or "").strip().lower()
    source_url = str(candidate.get("url") or "").strip().lower()

    if role == "icon" or origin == "link_icon":
        return True

    low_quality_tokens = ("favicon", "apple-touch-icon", "apple-touch")
    return any(token in source_url for token in low_quality_tokens)


def default_visual_prompt_contract() -> dict[str, Any]:
    return {
        "template": (
            "Create an on-brand image for {article_topic} aimed at {audience}. "
            "Intent: {intent}. Visual goal: {visual_goal}. "
            "Brand voice: {brand_voice}. Asset references: {asset_refs}."
        ),
        "required_variables": list(PROMPT_CONTRACT_REQUIRED_VARIABLES),
        "forbidden_terms": [
            "photoreal celebrity likeness",
            "unverified performance claims",
        ],
        "fallback_rules": [
            "If no logo asset is available, use neutral composition with brand colors only.",
            "If brand palette is missing, keep palette restrained and high-contrast.",
        ],
        "render_targets": [
            "blog_hero",
            "social_preview",
            "feature_callout",
        ],
    }


def normalize_prompt_contract(prompt_contract: dict[str, Any]) -> dict[str, Any]:
    """Ensure prompt contract has deterministic required fields."""
    required = normalize_string_list(
        [
            *PROMPT_CONTRACT_REQUIRED_VARIABLES,
            *[
                str(item)
                for item in (prompt_contract.get("required_variables") or [])
            ],
        ]
    )
    prompt_contract["required_variables"] = required

    template = str(prompt_contract.get("template") or "").strip()
    if not template:
        template = default_visual_prompt_contract()["template"]

    for variable in PROMPT_CONTRACT_REQUIRED_VARIABLES:
        placeholder = "{" + variable + "}"
        if placeholder not in template:
            template = f"{template} {placeholder}".strip()
    prompt_contract["template"] = " ".join(template.split())

    for key in ("forbidden_terms", "fallback_rules", "render_targets"):
        prompt_contract[key] = normalize_string_list(
            [str(value) for value in (prompt_contract.get(key) or [])]
        )

    return prompt_contract


def fallback_visual_confidence(
    *,
    extraction_confidence: float,
    has_assets: bool,
) -> float:
    baseline = min(max(extraction_confidence * 0.5, 0.0), 0.6)
    if has_assets:
        baseline = max(baseline, 0.2)
    return round(baseline, 3)


def default_visual_style_guide(
    *,
    tone_attributes: list[str],
    differentiators: list[str],
) -> dict[str, Any]:
    tone_text = ", ".join(tone_attributes[:4]) if tone_attributes else "professional"
    differentiators_text = (
        ", ".join(differentiators[:3])
        if differentiators
        else "clear positioning and practical value"
    )
    return {
        "brand_palette": {},
        "contrast_rules": [
            "Maintain high contrast between text and background elements.",
        ],
        "composition_rules": [
            "Prefer clean layouts with one dominant focal point.",
            "Align imagery with a tone that is " + tone_text + ".",
        ],
        "subject_rules": [
            "Center visuals around realistic business scenarios.",
            "Reinforce differentiators: " + differentiators_text + ".",
        ],
        "camera_lighting_rules": [
            "Use natural lighting and avoid harsh color casts.",
        ],
        "logo_usage_rules": [
            "Place logos in clear space and avoid distortion.",
        ],
        "negative_rules": [
            "Avoid exaggerated claims, misleading visuals, and generic stock look.",
        ],
        "accessibility_rules": [
            "Ensure readable overlays and avoid low-contrast color combinations.",
        ],
    }
