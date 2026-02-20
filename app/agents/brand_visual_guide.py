"""Agent for generating strict visual style guides for image generation."""

from __future__ import annotations

import logging

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


class VisualPromptContract(BaseModel):
    """Deterministic prompt contract consumed by image prompt builder."""

    template: str
    required_variables: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    fallback_rules: list[str] = Field(default_factory=list)
    render_targets: list[str] = Field(default_factory=list)


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
        return """You are a brand visual director.

Return a strict JSON visual style guide and deterministic prompt contract for image generation.

Rules:
1. Keep rules explicit and testable.
2. Align with brand context and known assets.
3. Include required placeholder variables for prompt templates.
4. Avoid vague language (e.g. "nice", "good", "beautiful").
5. Include clear negative rules to avoid off-brand imagery.
6. Ensure accessibility (contrast, readability, inclusive visuals).

Template constraints:
- The prompt contract template MUST include every required variable.
- Required variables: article_topic, audience, intent, visual_goal, brand_voice, asset_refs.
- The contract must be deterministic and reusable.
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
{input_data.brand_assets}

## Mandatory required_variables
{_REQUIRED_PROMPT_VARIABLES}

Return valid JSON matching the output schema exactly.
"""
