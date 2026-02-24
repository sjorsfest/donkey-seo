"""Agent for generating structured featured-image template specs."""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class GradientStop(BaseModel):
    """Single gradient stop for the canvas background."""

    color: str
    position: int = Field(ge=0, le=100)


class GradientSpec(BaseModel):
    """Linear gradient definition."""

    angle: int = Field(default=135, ge=0, le=360)
    stops: list[GradientStop] = Field(default_factory=list)


class ShapeSpec(BaseModel):
    """Abstract decorative shape constrained for deterministic rendering."""

    shape_type: Literal["circle", "rounded_rect", "blob", "line"]
    color: str
    opacity: float = Field(default=0.12, ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.02, le=1.0)
    height: float = Field(ge=0.02, le=1.0)
    blur_px: int = Field(default=0, ge=0, le=120)
    rotation_deg: int = Field(default=0, ge=-180, le=180)
    border_radius_px: int = Field(default=24, ge=0, le=240)


class TypographySpec(BaseModel):
    """Title typography settings."""

    font_family: str = "system-ui, -apple-system, Segoe UI, sans-serif"
    font_size_px: int = Field(default=72, ge=36, le=110)
    font_weight: int = Field(default=700, ge=400, le=900)
    line_height: float = Field(default=1.1, ge=0.9, le=1.5)
    letter_spacing_em: float = Field(default=0.0, ge=-0.05, le=0.1)
    color: str = "#111111"
    max_lines: int = Field(default=3, ge=2, le=4)
    align: Literal["left", "center"] = "left"


class TitleZoneSpec(BaseModel):
    """Layout constraints for placing the blog title text."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.1, le=1.0)
    height: float = Field(ge=0.1, le=1.0)
    padding_px: int = Field(default=36, ge=16, le=120)
    typography: TypographySpec = Field(default_factory=TypographySpec)


class LogoZoneSpec(BaseModel):
    """Optional logo placement constraints."""

    enabled: bool = True
    include_if_logo_available: bool = True
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.05, le=0.35)
    height: float = Field(ge=0.04, le=0.25)
    opacity: float = Field(default=1.0, ge=0.1, le=1.0)
    anchor: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "bottom-right"


class FeaturedImageTemplateSpec(BaseModel):
    """Structured template specification for deterministic image rendering."""

    template_version: str = "1.0"
    style_variant_id: str
    safe_margin_px: int = Field(default=36, ge=16, le=96)
    background_color: str
    gradient: GradientSpec | None = None
    shapes: list[ShapeSpec] = Field(default_factory=list, max_length=8)
    title_zone: TitleZoneSpec
    logo_zone: LogoZoneSpec


class FeaturedImageTemplateInput(BaseModel):
    """Input payload for featured image template generation."""

    article_title: str
    primary_keyword: str
    audience: str
    search_intent: str
    page_type: str
    brand_name: str
    brand_tagline: str | None = None
    visual_style_guide: dict = Field(default_factory=dict)
    brand_assets: list[dict] = Field(default_factory=list)


class FeaturedImageTemplateOutput(BaseModel):
    """Output payload for featured image template generation."""

    template: FeaturedImageTemplateSpec


class FeaturedImageTemplateAgent(
    BaseAgent[FeaturedImageTemplateInput, FeaturedImageTemplateOutput]
):
    """Generate high-variation template specs constrained by brand guidance."""

    model_tier = "reasoning"
    temperature = 0.8

    @property
    def system_prompt(self) -> str:
        return """You are a visual director producing STRICT JSON template specs for deterministic rendering.

Goal:
Design a creative but minimal featured image template for a blog post card.

Hard constraints:
1. Canvas is exactly 1200x630.
2. The only text that will be rendered is the provided article title.
3. Do not include subtitles, badges, labels, UI chrome, or decorative text.
4. Keep composition minimal: one clear focal text zone and restrained decorative shapes.
5. Obey brand style and palette guidance; never introduce clashing neon palettes unless evidence supports it.
6. If logo is available, reserve a clear logo zone with safe spacing.
7. Output must be valid JSON matching schema exactly.

Creativity mode:
- High experimentation is allowed for composition, gradients, shape language, and spacing.
- Maintain readability and brand alignment at all times.
"""

    @property
    def output_type(self) -> type[FeaturedImageTemplateOutput]:
        return FeaturedImageTemplateOutput

    def _build_prompt(self, input_data: FeaturedImageTemplateInput) -> str:
        logger.info(
            "Building featured image template prompt",
            extra={
                "primary_keyword": input_data.primary_keyword,
                "brand_name": input_data.brand_name,
                "assets_count": len(input_data.brand_assets),
            },
        )
        return (
            "Generate one featured image template specification for deterministic rendering.\n\n"
            "## Article Context\n"
            f"{json.dumps({
                'article_title': input_data.article_title,
                'primary_keyword': input_data.primary_keyword,
                'audience': input_data.audience,
                'search_intent': input_data.search_intent,
                'page_type': input_data.page_type,
            }, indent=2, ensure_ascii=True)}\n\n"
            "## Brand Context\n"
            f"{json.dumps({
                'brand_name': input_data.brand_name,
                'brand_tagline': input_data.brand_tagline,
                'visual_style_guide': input_data.visual_style_guide,
                'brand_assets': input_data.brand_assets,
            }, indent=2, ensure_ascii=True)}\n\n"
            "## Rendering Guardrails\n"
            "- Keep the layout uncluttered and intentionally minimal.\n"
            "- Use high contrast for title readability.\n"
            "- Keep title_zone and logo_zone separated with safe spacing.\n"
            "- Use at most 8 decorative shapes.\n"
            "- Do not request any additional text overlays.\n"
            "\nReturn JSON now."
        )
