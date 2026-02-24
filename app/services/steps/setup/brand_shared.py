"""Shared helpers for setup brand steps."""

from __future__ import annotations

import re
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
        "component_template": (
            "Render an on-brand UI component scene for {article_topic} aimed at {audience}. "
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
        "render_modes": [
            "llm_image_generation",
            "component_render",
        ],
        "component_render_targets": [
            "hero_card",
            "feature_card",
            "cta_button",
            "announcement_badge",
        ],
        "component_fallback_rules": [
            "If recipe variants are missing, render hero_card with default token values.",
            "If component render fails, fallback to LLM image generation template.",
        ],
    }


def normalize_prompt_contract(prompt_contract: dict[str, Any]) -> dict[str, Any]:
    """Ensure prompt contract has deterministic required fields."""
    default_contract = default_visual_prompt_contract()
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
        template = str(default_contract["template"])

    for variable in PROMPT_CONTRACT_REQUIRED_VARIABLES:
        placeholder = "{" + variable + "}"
        if placeholder not in template:
            template = f"{template} {placeholder}".strip()
    prompt_contract["template"] = " ".join(template.split())

    component_template = str(prompt_contract.get("component_template") or "").strip()
    if not component_template:
        component_template = str(default_contract["component_template"])
    for variable in PROMPT_CONTRACT_REQUIRED_VARIABLES:
        placeholder = "{" + variable + "}"
        if placeholder not in component_template:
            component_template = f"{component_template} {placeholder}".strip()
    prompt_contract["component_template"] = " ".join(component_template.split())

    for key in (
        "forbidden_terms",
        "fallback_rules",
        "render_targets",
        "render_modes",
        "component_render_targets",
        "component_fallback_rules",
    ):
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
    homepage_visual_signals: dict[str, Any] | None = None,
    site_visual_signals: dict[str, Any] | None = None,
    brand_assets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    homepage_signals = (
        homepage_visual_signals
        if isinstance(homepage_visual_signals, dict)
        else {}
    )
    site_signals = site_visual_signals if isinstance(site_visual_signals, dict) else {}
    assets = brand_assets if isinstance(brand_assets, list) else []

    tone_text = ", ".join(tone_attributes[:4]) if tone_attributes else "professional"
    differentiators_text = (
        ", ".join(differentiators[:3])
        if differentiators
        else "clear positioning and practical value"
    )

    observed_colors = _collect_observed_palette(
        homepage_signals=homepage_signals,
        site_signals=site_signals,
        brand_assets=assets,
    )
    palette = _build_palette_from_colors(observed_colors)

    shape_cues = _collect_visual_signal_values(
        homepage_signals,
        site_signals,
        key="shape_cues",
        max_items=3,
    )
    surface_cues = _collect_visual_signal_values(
        homepage_signals,
        site_signals,
        key="surface_cues",
        max_items=3,
    )
    cta_labels = _collect_visual_signal_values(
        homepage_signals,
        site_signals,
        key="cta_labels",
        max_items=3,
    )
    imagery_cues = _collect_visual_signal_values(
        homepage_signals,
        site_signals,
        key="imagery_cues",
        max_items=3,
    )
    font_families = _collect_visual_signal_values(
        homepage_signals,
        site_signals,
        key="observed_font_families",
        max_items=3,
    )
    hero_headlines = _collect_visual_signal_values(
        homepage_signals,
        site_signals,
        key="hero_headlines",
        max_items=2,
    )

    composition_rules = [
        "Preserve landing-page visual rhythm and avoid generic stock composition.",
        "Align imagery with a tone that is " + tone_text + ".",
    ]
    if shape_cues:
        composition_rules.append(
            "Reflect observed UI shape cues: " + ", ".join(shape_cues) + "."
        )
    if surface_cues:
        composition_rules.append(
            "Carry forward surface styling cues: " + ", ".join(surface_cues) + "."
        )
    if cta_labels:
        composition_rules.append(
            "Match CTA tone seen in site actions like: " + ", ".join(cta_labels) + "."
        )

    subject_rules = [
        "Reinforce differentiators: " + differentiators_text + ".",
    ]
    if imagery_cues:
        subject_rules.insert(0, "Match observed imagery style: " + ", ".join(imagery_cues) + ".")
    else:
        subject_rules.insert(
            0,
            "Use brand-relevant product context and avoid disconnected abstract metaphors.",
        )

    illustration_first = any(
        "illustrative" in cue.casefold() or "ui/screenshot" in cue.casefold()
        for cue in imagery_cues
    )
    if illustration_first:
        camera_lighting_rules = [
            "Illustration/UI-first treatment with clean edges and consistent stroke weight.",
            "Use soft shading and subtle depth; avoid photoreal camera depth-of-field effects.",
        ]
    else:
        camera_lighting_rules = [
            "Use soft, natural lighting with moderate contrast and readable foreground details.",
        ]

    contrast_rules = [
        "Maintain WCAG AA contrast for text overlays and key callouts.",
    ]
    neutral_dark = str(palette.get("neutral_dark") or "").strip()
    neutral_light = str(palette.get("neutral_light") or "").strip()
    if neutral_dark and neutral_light:
        contrast_rules.append(
            f"Prefer {neutral_dark} on {neutral_light} and reverse for emphasis."
        )

    design_tokens = _build_design_tokens(
        palette=palette,
        font_families=font_families,
        shape_cues=shape_cues,
        surface_cues=surface_cues,
    )
    component_recipes = _build_component_recipes(
        cta_labels=cta_labels,
        shape_cues=shape_cues,
        surface_cues=surface_cues,
        hero_headlines=hero_headlines,
    )
    component_style_rules = [
        "Use design_tokens as the source of truth for colors, spacing, radii, border width, and shadows.",
        "Apply consistent stroke thickness across cards and controls for a coherent UI family.",
        "Use high-contrast button labeling and maintain visible focus outlines for keyboard navigation.",
    ]
    if shape_cues:
        component_style_rules.append(
            "Preserve observed geometry cues: " + ", ".join(shape_cues) + "."
        )

    component_layout_rules = [
        "Desktop hero components should support split layout with text and visual panel.",
        "Mobile layout should stack content vertically and preserve tap targets >= 44px.",
        "Keep component padding and spacing aligned to an 8px scale.",
    ]
    if cta_labels:
        component_layout_rules.append(
            "Primary CTA text should mirror site wording such as: " + ", ".join(cta_labels) + "."
        )

    return {
        "brand_palette": palette,
        "contrast_rules": normalize_string_list(contrast_rules),
        "composition_rules": normalize_string_list(composition_rules),
        "subject_rules": normalize_string_list(subject_rules),
        "camera_lighting_rules": normalize_string_list(camera_lighting_rules),
        "logo_usage_rules": [
            "Place logos in clear space and avoid distortion.",
        ],
        "negative_rules": [
            "Avoid exaggerated claims, misleading visuals, and mismatched generic stock aesthetics.",
            "Do not introduce unrelated platform branding or off-brand palettes.",
        ],
        "accessibility_rules": [
            "Ensure readable overlays and avoid low-contrast color combinations.",
            "Do not rely on color alone for emphasis; pair with shape or text cues.",
        ],
        "design_tokens": design_tokens,
        "component_style_rules": normalize_string_list(component_style_rules),
        "component_layout_rules": normalize_string_list(component_layout_rules),
        "component_recipes": component_recipes,
        "component_negative_rules": [
            "Do not use glassmorphism, neon glow, or gradients unless observed in source styles.",
            "Avoid default framework styling that ignores tokenized radii, stroke, and spacing rules.",
            "Do not switch to photoreal stock-photo composition in component-render mode.",
        ],
    }


def _collect_observed_palette(
    *,
    homepage_signals: dict[str, Any],
    site_signals: dict[str, Any],
    brand_assets: list[dict[str, Any]],
) -> list[str]:
    colors: list[str] = []

    for signal_set in (homepage_signals, site_signals):
        signal_colors = signal_set.get("observed_hex_colors")
        if isinstance(signal_colors, list):
            for color in signal_colors:
                normalized = _normalize_hex_color(str(color))
                if normalized:
                    colors.append(normalized)

    for asset in brand_assets:
        if not isinstance(asset, dict):
            continue
        dominant_colors = asset.get("dominant_colors")
        if not isinstance(dominant_colors, list):
            continue
        for color in dominant_colors:
            normalized = _normalize_hex_color(str(color))
            if normalized:
                colors.append(normalized)

    if colors:
        return normalize_string_list(colors)[:8]

    hinted_palette = {
        "pink": "#EC4899",
        "yellow": "#FACC15",
        "blue": "#0EA5E9",
        "indigo": "#4F46E5",
        "purple": "#9333EA",
        "green": "#10B981",
        "emerald": "#10B981",
        "teal": "#14B8A6",
        "orange": "#F97316",
        "red": "#EF4444",
        "gray": "#6B7280",
        "slate": "#334155",
        "black": "#111827",
        "white": "#F9FAFB",
        "cream": "#F4E8C1",
        "beige": "#EAD7B0",
    }
    hints = _collect_visual_signal_values(
        homepage_signals,
        site_signals,
        key="color_word_hints",
        max_items=8,
    )
    for hint in hints:
        mapped = hinted_palette.get(hint.casefold())
        if mapped:
            colors.append(mapped)

    return normalize_string_list(colors)[:8]


def _build_palette_from_colors(colors: list[str]) -> dict[str, str]:
    if not colors:
        return {}

    normalized = normalize_string_list(
        [color for color in colors if _normalize_hex_color(color)]
    )
    if not normalized:
        return {}

    palette: dict[str, str] = {
        "primary": normalized[0],
    }
    if len(normalized) > 1:
        palette["secondary"] = normalized[1]
    if len(normalized) > 2:
        palette["accent"] = normalized[2]

    sorted_by_luminance = sorted(normalized, key=_hex_to_luminance)
    darkest = sorted_by_luminance[0]
    lightest = sorted_by_luminance[-1]
    if _hex_to_luminance(darkest) < 0.35:
        palette["neutral_dark"] = darkest
    if _hex_to_luminance(lightest) > 0.75:
        palette["neutral_light"] = lightest

    return palette


def _build_design_tokens(
    *,
    palette: dict[str, str],
    font_families: list[str],
    shape_cues: list[str],
    surface_cues: list[str],
) -> dict[str, Any]:
    rounded_ui = any("rounded" in cue.casefold() or "pill" in cue.casefold() for cue in shape_cues)
    outlined_ui = any("outline" in cue.casefold() or "stroke" in cue.casefold() for cue in shape_cues)
    strong_shadow = any("shadow" in cue.casefold() for cue in surface_cues)

    colors = {
        "primary": palette.get("primary", "#2563EB"),
        "secondary": palette.get("secondary", "#1E293B"),
        "accent": palette.get("accent", "#F59E0B"),
        "surface": palette.get("neutral_light", "#FFFFFF"),
        "text_primary": palette.get("neutral_dark", "#0F172A"),
    }

    if rounded_ui:
        radii = {"sm": "10px", "md": "16px", "lg": "24px", "pill": "999px"}
    else:
        radii = {"sm": "6px", "md": "10px", "lg": "16px", "pill": "999px"}

    border_width = "2px" if outlined_ui else "1px"
    shadow_card = "0 8px 0 rgba(15, 23, 42, 0.22)" if strong_shadow else "0 4px 14px rgba(15, 23, 42, 0.12)"
    shadow_button = "0 4px 0 rgba(15, 23, 42, 0.3)" if strong_shadow else "0 2px 8px rgba(15, 23, 42, 0.2)"

    display_font = font_families[0] if font_families else "inherit"
    body_font = font_families[1] if len(font_families) > 1 else display_font

    return {
        "colors": colors,
        "typography": {
            "display_font": display_font,
            "body_font": body_font,
            "heading_weight": "700",
            "body_weight": "500",
            "heading_case": "sentence",
        },
        "spacing": {"xs": "8px", "sm": "12px", "md": "16px", "lg": "24px", "xl": "32px"},
        "radii": radii,
        "borders": {"width": border_width, "style": "solid"},
        "shadows": {"card": shadow_card, "button": shadow_button},
    }


def _build_component_recipes(
    *,
    cta_labels: list[str],
    shape_cues: list[str],
    surface_cues: list[str],
    hero_headlines: list[str],
) -> list[dict[str, Any]]:
    primary_cta = cta_labels[0] if cta_labels else "Get Started"
    secondary_cta = cta_labels[1] if len(cta_labels) > 1 else "See How It Works"
    headline_hint = hero_headlines[0] if hero_headlines else "Clear product value headline"
    shape_hint = ", ".join(shape_cues) if shape_cues else "Consistent rounded card geometry"
    surface_hint = ", ".join(surface_cues) if surface_cues else "Subtle elevation and clean layering"

    return [
        {
            "name": "hero_card",
            "purpose": "Primary above-the-fold visual for landing-page style imagery",
            "structure": [
                "announcement_badge",
                "headline",
                "supporting_copy",
                "primary_cta",
                "secondary_cta",
                "visual_panel_or_ui_mock",
            ],
            "style_rules": [
                f"Use headline tone similar to: {headline_hint}.",
                f"Preserve shape language: {shape_hint}.",
                f"Preserve surface treatment: {surface_hint}.",
            ],
            "variants": ["split", "stacked", "compact"],
        },
        {
            "name": "cta_button",
            "purpose": "Reusable call-to-action component for rendered scenes",
            "structure": ["label", "optional_icon"],
            "style_rules": [
                f"Primary CTA label reference: {primary_cta}.",
                f"Secondary CTA label reference: {secondary_cta}.",
                "Use tokenized padding, radius, border, and shadow values only.",
            ],
            "variants": ["primary", "secondary", "ghost"],
        },
        {
            "name": "feature_card",
            "purpose": "Supporting value/benefit card component",
            "structure": ["icon_or_avatar", "title", "description", "meta_or_tag"],
            "style_rules": [
                "Keep title short and benefit-driven.",
                "Prefer one clear border treatment and one elevation style per card.",
                "Maintain consistent spacing rhythm from design tokens.",
            ],
            "variants": ["single", "grid_item", "highlighted"],
        },
    ]


def _collect_visual_signal_values(
    *signal_sets: dict[str, Any],
    key: str,
    max_items: int,
) -> list[str]:
    values: list[str] = []
    for signal_set in signal_sets:
        raw_values = signal_set.get(key)
        if not isinstance(raw_values, list):
            continue
        values.extend(str(item) for item in raw_values)
    return normalize_string_list(values)[:max_items]


def _normalize_hex_color(value: str) -> str | None:
    raw = value.strip()
    if not raw.startswith("#"):
        return None
    match = re.fullmatch(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", raw)
    if not match:
        return None
    hex_part = match.group(1)
    if len(hex_part) == 3:
        hex_part = "".join(char * 2 for char in hex_part)
    return f"#{hex_part.upper()}"


def _hex_to_luminance(color: str) -> float:
    normalized = _normalize_hex_color(color)
    if not normalized:
        return 1.0
    red = int(normalized[1:3], 16) / 255.0
    green = int(normalized[3:5], 16) / 255.0
    blue = int(normalized[5:7], 16) / 255.0
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
