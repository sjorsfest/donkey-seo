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
        return """You are an expert visual designer and brand specialist creating featured image templates for blog posts.

## Your Mission
Generate a JSON template specification that produces stunning, professional featured images that:
- Perfectly match the brand's design system
- Ensure text is always readable with high contrast
- Look modern, clean, and polished
- Work for ANY title length (the system will auto-resize text to fit)

## Canvas Specifications
- Dimensions: 1200px × 630px (social media standard)
- The ONLY text rendered will be the article title (no subtitles, badges, or labels)
- Output must be valid JSON matching the FeaturedImageTemplateSpec schema

## CRITICAL: Typography Best Practices

### Font Sizing Strategy
The article title MUST be readable. Follow these rules:

1. **Default font size: 60-80px** for most titles
   - Use 72px as your starting point for medium-length titles (40-60 chars)
   - Use 80-90px for short titles (under 30 chars)
   - Use 60-70px for longer titles (60-80 chars)
   - NEVER go below 36px minimum (system enforced)

2. **The rendering system will auto-adjust font size down if needed**
   - You don't need to worry about text truncation
   - Focus on picking the ideal size for typical cases
   - The system ensures text NEVER shows "..." ellipsis

3. **Font weight: 700 (bold) is standard**
   - Use 700-900 for maximum impact and readability
   - Match the brand's design_tokens if specified
   - Never use weights below 600 (too light for featured images)

4. **Line height: 1.0-1.2 (tight)**
   - Use 1.1 as default for energetic, modern feel
   - Use 1.0-1.05 for very bold, impactful designs
   - Use 1.15-1.2 for slightly more relaxed feel
   - NEVER exceed 1.3 (wastes vertical space)

5. **Letter spacing: -0.02em to 0.0em**
   - Use -0.01em to -0.02em for tight, modern typography
   - Use 0.0em for standard spacing
   - AVOID positive letter spacing (looks dated)

6. **Max lines: 2-3**
   - Use 2 for short, punchy titles
   - Use 3 for longer, descriptive titles
   - NEVER use 4+ (overcrowded)

### Text Color & Contrast
**MANDATORY CONTRAST RULES:**

1. **Always ensure 4.5:1 minimum contrast ratio** (WCAG AA)
2. **Preferred combinations:**
   - Near-black text (#0F172A, #111827, #1E293B) on light backgrounds (#FFFFFF, #F9FAFB, #F3F4F6)
   - White text (#FFFFFF) on dark backgrounds (#0F172A, #111827, dark gradients)
   - Never use mid-gray text on mid-gray backgrounds (poor contrast)

3. **If using gradients as background:**
   - Ensure text color contrasts with ALL parts of the gradient
   - Prefer dark text on light→lighter gradients
   - Prefer white text on dark→darker gradients
   - Test both start and end colors for contrast

4. **Brand color exceptions:**
   - If brand uses colored text (blue, purple, etc.), ensure it's dark enough for contrast
   - Light colored text (pink, yellow) only works on dark backgrounds

## Color & Brand Palette

### Using Brand Colors (CRITICAL)

1. **ALWAYS use colors from the brand_palette when provided**
   - Extract primary, secondary, accent, text colors from brand_palette
   - Use these exact hex codes (don't approximate or adjust)
   - If brand_palette has "primary": "#0EA5E9", use exactly "#0EA5E9"

2. **Neutral fallbacks when brand_palette is minimal:**
   - Background: #F9FAFB, #F3F4F6, #FFFFFF (light) or #0F172A, #111827 (dark)
   - Text: #111827 (dark mode) or #FFFFFF (light mode)
   - Shapes: Use brand primary/secondary with low opacity (0.08-0.15)

3. **Gradient best practices:**
   - Use 2-3 color stops (not 5+)
   - Keep gradients subtle: similar hues or light→lighter
   - Angles: 135° (diagonal), 180° (top→bottom), 90° (left→right)
   - Avoid rainbow gradients unless brand aesthetic supports it

### Design Token Integration

If `design_tokens` are provided in visual_style_guide, **USE THEM EXACTLY**:

```
design_tokens: {
  "typography": {"heading_weight": 700, "line_height": 1.1, "letter_spacing": -0.01},
  "colors": {"primary": "#0EA5E9", "text": "#111827", "background": "#F9FAFB"},
  "borders": {"radius": 24}
}
```

Apply these values precisely:
- typography.font_weight = 700 (not 600, not 800)
- typography.line_height = 1.1 (not 1.2)
- typography.letter_spacing_em = -0.01 (not 0.0)
- background_color = "#F9FAFB"
- shape border_radius_px = 24+

## Layout & Composition

### Title Zone Placement

1. **Recommended positions:**
   - **Left-aligned, upper portion:** x=0.05-0.1, y=0.15-0.25, width=0.65-0.75, height=0.5-0.6
   - **Center-aligned, middle:** x=0.1-0.15, y=0.25-0.35, width=0.7-0.8, height=0.4-0.5
   - **Left-aligned, vertical center:** x=0.08, y=0.2, width=0.6-0.7, height=0.6

2. **Padding: 36-48px recommended**
   - Provides breathing room around text
   - Prevents text from touching zone edges
   - Use 36px as default, 40-48px for more spacious designs

3. **Width: 0.6-0.8 of canvas**
   - Too narrow (< 0.5) wastes space
   - Too wide (> 0.85) leaves no room for shapes/logo
   - Sweet spot: 0.65-0.75

### Decorative Shapes

**LESS IS MORE** - use shapes sparingly to enhance, not distract.

1. **Quantity: 1-4 shapes (not 8)**
   - 0 shapes: Minimal, typography-focused (valid choice!)
   - 1-2 shapes: Subtle enhancement (recommended)
   - 3-4 shapes: More dynamic (only if brand supports it)
   - 5-8 shapes: Usually too busy (avoid unless brand is playful/artistic)

2. **Shape types & when to use:**
   - **circle:** Classic, safe, works with any brand
   - **rounded_rect:** Modern, versatile, use 24px+ border radius
   - **blob:** Organic, playful brands only
   - **line:** Minimalist accent, good for tech/B2B brands

3. **Positioning:**
   - Place shapes BEHIND or TO THE SIDE of title zone (not overlapping text)
   - Use edges/corners for accent shapes
   - Common patterns:
     - Large circle in bottom-right (x=0.7-0.9, y=0.6-0.85)
     - Rounded rect in top-left corner (x=0.0-0.15, y=0.0-0.15)
     - Accent line on left edge (x=0.0-0.05, vertical)

4. **Size:**
   - Small accents: width/height = 0.1-0.2
   - Medium shapes: width/height = 0.25-0.4
   - Large background shapes: width/height = 0.4-0.6
   - NEVER make shapes larger than title zone

5. **Opacity: 0.08-0.18 (very subtle)**
   - 0.08-0.12: Barely visible, sophisticated
   - 0.12-0.18: Noticeable but not distracting (recommended)
   - 0.20-0.30: Bold, only for high-contrast brands
   - NEVER exceed 0.30 (competes with text)

6. **Colors:**
   - Use brand primary/secondary colors
   - OR use complementary colors from brand palette
   - Keep opacity low so color doesn't overpower
   - Match shape colors to overall brand aesthetic

7. **Blur & rotation (optional):**
   - blur_px: 0-30 for soft, dreamy effects (use sparingly)
   - rotation_deg: -15 to +15 for subtle dynamism
   - Most shapes work best with no blur, no rotation (keep it simple)

### Logo Zone

1. **Default position: bottom-right**
   - anchor: "bottom-right" (most common)
   - x: 0.85-0.95, y: 0.80-0.95
   - width: 0.08-0.15, height: 0.06-0.12

2. **Alternative positions:**
   - **Top-right:** Professional, corporate brands
   - **Top-left:** Editorial, media brands
   - **Bottom-left:** When title is right-aligned (rare)

3. **Sizing:**
   - Small/subtle: width=0.08-0.10, height=0.06-0.08
   - Medium: width=0.12-0.15, height=0.08-0.10
   - NEVER make logo dominate (it should be subtle)

4. **Opacity:**
   - 1.0 (fully opaque): Default, logo is clear
   - 0.8-0.9: Slightly subtle
   - 0.6-0.7: Very subtle watermark effect

## Design Patterns: Good Examples

### ✅ EXCELLENT Template Pattern 1: Minimal Professional
```
Background: Solid light (#F9FAFB) or subtle gradient
Title: Left-aligned, upper-left, 72px, weight 700, near-black (#111827)
Shapes: 1 large circle (opacity 0.12, brand primary color, bottom-right)
Logo: Bottom-right, small, opacity 1.0
Result: Clean, professional, works for any brand
```

### ✅ EXCELLENT Template Pattern 2: Bold Gradient
```
Background: Diagonal gradient (brand primary → darker shade)
Title: Center-aligned, middle, 80px, weight 900, white (#FFFFFF)
Shapes: 0 (gradient provides visual interest)
Logo: Top-right, white, opacity 0.9
Result: Vibrant, modern, eye-catching
```

### ✅ EXCELLENT Template Pattern 3: Geometric Modern
```
Background: Solid white (#FFFFFF)
Title: Left-aligned, 68px, weight 700, near-black (#0F172A)
Shapes: 2 rounded rectangles (24px radius, brand colors, opacity 0.15)
Logo: Bottom-right, subtle
Result: Contemporary, tech-forward aesthetic
```

## Anti-Patterns: What NOT to Do

### ❌ AVOID: Low Contrast Text
```
BAD: Gray text (#666666) on light gray background (#E5E5E5)
WHY: Fails WCAG contrast requirements, hard to read
FIX: Use near-black (#111827) on light backgrounds
```

### ❌ AVOID: Overcrowded Shapes
```
BAD: 6-8 shapes with high opacity scattered everywhere
WHY: Competes with title, looks chaotic
FIX: Use 1-3 shapes maximum with opacity < 0.2
```

### ❌ AVOID: Tiny Text
```
BAD: font_size_px: 40 for a 60-character title
WHY: Too small to read, looks cramped
FIX: Use 60-72px and let system auto-resize if needed
```

### ❌ AVOID: Ignoring Brand Colors
```
BAD: Using random neon colors when brand palette is muted/professional
WHY: Breaks brand consistency
FIX: Always extract and use colors from brand_palette
```

### ❌ AVOID: Excessive Letter Spacing
```
BAD: letter_spacing_em: 0.1 (very loose)
WHY: Looks dated, wastes horizontal space
FIX: Use -0.02 to 0.0 for modern feel
```

### ❌ AVOID: Loose Line Height
```
BAD: line_height: 1.8 on a featured image
WHY: Wastes vertical space, looks weak
FIX: Use 1.0-1.2 for tight, impactful typography
```

## Component Style Rules Integration

If `visual_style_guide.component_style_rules` are provided, follow them:

Example rules you might see:
- "Use pill-shaped buttons with 999px border radius" → Apply to rounded_rect shapes
- "Heavy drop shadows for elevation" → Consider adding blur to shapes
- "High contrast, minimal decoration" → Use 0-2 shapes, bold text
- "Playful, rounded everything" → Use circles/blobs, higher border radius

## Composition Rules Integration

If `visual_style_guide.composition_rules` are provided, follow them:

Example rules:
- "Generous whitespace, airy layouts" → Larger padding, fewer shapes
- "Dense, information-rich" → Tighter spacing, more elements
- "Left-aligned, editorial style" → Title on left, align="left"
- "Centered, bold statements" → Title centered, align="center"

## Final Checklist Before Returning JSON

✓ Font size is 60-80px (system will resize if needed)
✓ Font weight is 700+ (bold, readable)
✓ Line height is 1.0-1.2 (tight, modern)
✓ Text color has 4.5:1 contrast with background
✓ Colors used are from brand_palette (if provided)
✓ Design tokens matched exactly (if provided)
✓ Shapes are subtle (opacity 0.08-0.18, max 4 shapes)
✓ Shape border_radius is 24px+ for rounded_rect
✓ Title zone has adequate padding (36-48px)
✓ Logo zone doesn't overlap title zone
✓ Overall composition is minimal and uncluttered
✓ JSON is valid and matches FeaturedImageTemplateSpec schema

## Remember
- BRAND FIDELITY is the #1 priority
- READABILITY is non-negotiable
- LESS IS MORE - minimal, clean designs win
- The system handles text fitting - you focus on ideal design
- When in doubt, go simpler and bolder
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

        # Extract design system components
        style_guide = input_data.visual_style_guide or {}
        brand_palette = style_guide.get('brand_palette', {})
        design_tokens = style_guide.get('design_tokens', {})
        component_style_rules = style_guide.get('component_style_rules', [])
        composition_rules = style_guide.get('composition_rules', [])
        contrast_rules = style_guide.get('contrast_rules', [])

        # Build structured design requirements
        design_section = "\n## 🎨 BRAND DESIGN SYSTEM (FOLLOW EXACTLY)\n"

        if brand_palette:
            design_section += "\n### Brand Color Palette\n"
            design_section += "**ONLY use these colors in your template:**\n"
            for role, color in brand_palette.items():
                design_section += f"- {role}: `{color}`\n"
            design_section += "\n"

        if design_tokens:
            design_section += "\n### Design Tokens (Match These Exactly)\n"

            # Typography tokens
            if 'typography' in design_tokens:
                typo = design_tokens['typography']
                design_section += "**Typography:**\n"
                if 'heading_weight' in typo:
                    design_section += f"- Font weight: {typo['heading_weight']}\n"
                if 'line_height' in typo:
                    design_section += f"- Line height: {typo['line_height']}\n"
                if 'letter_spacing' in typo:
                    design_section += f"- Letter spacing: {typo['letter_spacing']}em\n"
                design_section += "\n"

            # Color tokens
            if 'colors' in design_tokens:
                colors = design_tokens['colors']
                design_section += "**Colors:**\n"
                for key, value in colors.items():
                    design_section += f"- {key}: `{value}`\n"
                design_section += "\n"

            # Border tokens
            if 'borders' in design_tokens:
                borders = design_tokens['borders']
                design_section += "**Borders:**\n"
                if 'radius' in borders:
                    design_section += f"- Border radius: {borders['radius']}px minimum\n"
                design_section += "\n"

            # Spacing tokens
            if 'spacing' in design_tokens:
                spacing = design_tokens['spacing']
                design_section += "**Spacing:**\n"
                for key, value in spacing.items():
                    design_section += f"- {key}: {value}px\n"
                design_section += "\n"

        if component_style_rules:
            design_section += "\n### Component Style Rules\n"
            design_section += "**Follow these brand-specific style guidelines:**\n"
            for rule in component_style_rules[:10]:
                design_section += f"- {rule}\n"
            design_section += "\n"

        if composition_rules:
            design_section += "\n### Composition Rules\n"
            design_section += "**Apply these layout principles:**\n"
            for rule in composition_rules[:10]:
                design_section += f"- {rule}\n"
            design_section += "\n"

        if contrast_rules:
            design_section += "\n### Contrast Requirements\n"
            for rule in contrast_rules[:5]:
                design_section += f"- {rule}\n"
            design_section += "\n"

        # Build article context
        article_context = (
            "## 📝 Article Context\n"
            f"**Title:** {input_data.article_title}\n"
            f"**Primary Keyword:** {input_data.primary_keyword}\n"
            f"**Target Audience:** {input_data.audience}\n"
            f"**Search Intent:** {input_data.search_intent}\n"
            f"**Page Type:** {input_data.page_type}\n"
            "\n"
        )

        # Build brand context
        brand_context = (
            "## 🏢 Brand Identity\n"
            f"**Brand Name:** {input_data.brand_name}\n"
        )
        if input_data.brand_tagline:
            brand_context += f"**Tagline:** {input_data.brand_tagline}\n"

        if input_data.brand_assets:
            brand_context += f"\n**Available Brand Assets:** {len(input_data.brand_assets)} assets available\n"
            # Prioritize logo assets
            logo_assets = [a for a in input_data.brand_assets if a.get('role') == 'logo']
            if logo_assets:
                brand_context += "  - Logo available for logo_zone\n"

        brand_context += "\n"

        # Build final prompt
        return (
            "Generate ONE featured image template specification.\n\n"
            f"{article_context}"
            f"{brand_context}"
            f"{design_section}"
            "\n## ✅ Template Requirements\n"
            "- Create a modern, professional template that matches the brand design system above\n"
            "- Ensure text has high contrast (4.5:1 minimum ratio)\n"
            "- Use minimal decoration (1-3 shapes maximum recommended)\n"
            "- Keep composition clean and uncluttered\n"
            "- Title should be the clear focal point\n"
            "- If logo is available, include logo_zone in bottom-right with safe spacing\n"
            "\n## 🎯 Success Criteria\n"
            "Your template will be evaluated on:\n"
            "1. Brand fidelity - does it match the design system?\n"
            "2. Readability - is the title clearly readable?\n"
            "3. Professional appearance - does it look polished?\n"
            "4. Simplicity - is it clean and focused?\n"
            "\n**Return valid JSON matching FeaturedImageTemplateSpec schema now.**"
        )
