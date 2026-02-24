"""Agent for generating strict visual style guides for image generation."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

_REQUIRED_PROMPT_VARIABLES = [
    "article_topic",
    "audience",
    "intent",
    "visual_goal",
    "brand_voice",
    "asset_refs",
]


class VisualStyleGuide(BaseModel):
    """Structured visual style rules for a brand."""

    brand_palette: dict[str, str] = Field(default_factory=dict)
    contrast_rules: list[str] = Field(default_factory=list)
    composition_rules: list[str] = Field(default_factory=list)
    subject_rules: list[str] = Field(default_factory=list)
    camera_lighting_rules: list[str] = Field(default_factory=list)
    logo_usage_rules: list[str] = Field(default_factory=list)
    negative_rules: list[str] = Field(default_factory=list)
    accessibility_rules: list[str] = Field(default_factory=list)
    design_tokens: dict[str, Any] = Field(default_factory=dict)
    component_style_rules: list[str] = Field(default_factory=list)
    component_layout_rules: list[str] = Field(default_factory=list)
    component_recipes: list[dict[str, Any]] = Field(default_factory=list)
    component_negative_rules: list[str] = Field(default_factory=list)


class VisualPromptContract(BaseModel):
    """Deterministic prompt contract consumed by image prompt builder."""

    template: str
    component_template: str | None = None
    required_variables: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    fallback_rules: list[str] = Field(default_factory=list)
    render_targets: list[str] = Field(default_factory=list)
    render_modes: list[str] = Field(default_factory=list)
    component_render_targets: list[str] = Field(default_factory=list)
    component_fallback_rules: list[str] = Field(default_factory=list)


class BrandVisualGuideInput(BaseModel):
    """Input payload for visual guide generation."""

    company_name: str
    tagline: str | None = None
    tone_attributes: list[str] = Field(default_factory=list)
    unique_value_props: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    brand_assets: list[dict] = Field(default_factory=list)
    homepage_visual_signals: dict[str, list[str]] = Field(default_factory=dict)
    site_visual_signals: dict[str, list[str]] = Field(default_factory=dict)


class BrandVisualGuideOutput(BaseModel):
    """Output payload for visual style guide generation."""

    visual_style_guide: VisualStyleGuide
    visual_prompt_contract: VisualPromptContract
    extraction_confidence: float = Field(ge=0, le=1)


class BrandVisualGuideAgent(BaseAgent[BrandVisualGuideInput, BrandVisualGuideOutput]):
    """Generate strict visual style and prompt contract from brand context."""

    model_tier = "reasoning"
    temperature = 0.2

    @property
    def system_prompt(self) -> str:
        return """You are a brand visual director focused on faithful style extraction.

Return a strict JSON visual style guide and deterministic prompt contract for both:
- LLM image generation
- Component-render image generation (rendering styled HTML/CSS components into images)

Rules:
1. Keep rules explicit and testable.
2. Prioritize visual evidence from homepage/site visual signals and known assets.
3. Include required placeholder variables for prompt templates.
4. Avoid vague language (e.g. "nice", "good", "beautiful").
5. Include clear negative rules to avoid off-brand imagery.
6. Ensure accessibility (contrast, readability, inclusive visuals).
7. Do not invent colors, mascots, lighting setups, or platform context that is not present in evidence.
8. If evidence indicates an illustration/UI-first brand, avoid photographic camera rules.

Template constraints:
- The prompt contract template MUST include every required variable.
- Required variables: article_topic, audience, intent, visual_goal, brand_voice, asset_refs.
- The contract must be deterministic and reusable.
- Include both template and component_template.
- Include render_modes and component_render_targets for component pipelines.

Output priorities:
- Capture look-and-feel fidelity over generic design advice.
- Prefer concrete cues (shape language, line weight, shadow style, CTA tone, spacing rhythm).
- Use observed hex colors when available for brand_palette.
- Keep each rule list concise and practical for image generation.
- Include component-ready style fields:
  design_tokens, component_style_rules, component_layout_rules, component_recipes, component_negative_rules.
"""

    @property
    def output_type(self) -> type[BrandVisualGuideOutput]:
        return BrandVisualGuideOutput

    def _build_prompt(self, input_data: BrandVisualGuideInput) -> str:
        logger.info(
            "Building brand visual guide prompt",
            extra={
                "company_name": input_data.company_name,
                "assets_count": len(input_data.brand_assets),
            },
        )
        tone_text = (
            ", ".join(input_data.tone_attributes)
            if input_data.tone_attributes
            else "Professional"
        )
        uvp_text = (
            ", ".join(input_data.unique_value_props)
            if input_data.unique_value_props
            else "Not provided"
        )
        differentiator_text = (
            ", ".join(input_data.differentiators)
            if input_data.differentiators
            else "Not provided"
        )
        roles_text = (
            ", ".join(input_data.target_roles)
            if input_data.target_roles
            else "Not provided"
        )
        industries_text = (
            ", ".join(input_data.target_industries)
            if input_data.target_industries
            else "Not provided"
        )
        asset_evidence = self._summarize_assets(input_data.brand_assets)
        homepage_signals = input_data.homepage_visual_signals or {}
        site_signals = input_data.site_visual_signals or {}
        return f"""Generate a strict visual style guide and prompt contract for this brand.

## Brand
- Company: {input_data.company_name}
- Tagline: {input_data.tagline or 'Not provided'}
- Tone: {tone_text}
- UVPs: {uvp_text}
- Differentiators: {differentiator_text}
- Target roles: {roles_text}
- Target industries: {industries_text}

## Known Brand Assets
{json.dumps(asset_evidence, indent=2)}

## Homepage Visual Signals (observed)
{json.dumps(homepage_signals, indent=2)}

## Site-Wide Visual Signals (observed)
{json.dumps(site_signals, indent=2)}

## Mandatory required_variables
{_REQUIRED_PROMPT_VARIABLES}

## Non-negotiable fidelity requirements
- Reflect the actual landing-page vibe and interaction style inferred from evidence.
- Avoid boilerplate defaults like "rule of thirds" unless evidence clearly supports it.
- If visual evidence is sparse, return conservative neutral rules and lower extraction confidence.
- Mention specific visual mechanics when evidence exists: rounded/pill controls, heavy outlines,
  playful vs formal typography, pastel vs high-contrast palette, dense vs airy layout.

## Component-render compatibility requirements
- Provide design_tokens for colors, typography, spacing, radii, borders, and shadows.
- Provide component_recipes for at least hero card, CTA button, and feature/info card.
- Every component recipe should include name, purpose, structure, style_rules, and variants.
- component_layout_rules must include desktop+mobile behavior.
- component_negative_rules must explicitly block off-brand UI patterns.

Return valid JSON matching the output schema exactly.
"""

    @staticmethod
    def _summarize_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for raw_asset in assets:
            if not isinstance(raw_asset, dict):
                continue
            role = str(raw_asset.get("role") or "").strip()
            source_url = str(raw_asset.get("source_url") or raw_asset.get("url") or "").strip()
            object_key = str(raw_asset.get("object_key") or "").strip()
            if not role and not source_url and not object_key:
                continue

            dominant_colors = [
                str(color).strip()
                for color in raw_asset.get("dominant_colors", [])
                if str(color).strip()
            ]
            summary.append(
                {
                    "role": role or "reference",
                    "role_confidence": float(raw_asset.get("role_confidence") or 0.0),
                    "source_url": source_url or None,
                    "object_key": object_key or None,
                    "width": raw_asset.get("width"),
                    "height": raw_asset.get("height"),
                    "dominant_colors": dominant_colors[:6],
                    "average_luminance": raw_asset.get("average_luminance"),
                    "origin": str(raw_asset.get("origin") or "").strip() or None,
                }
            )
        return summary[:10]
