"""Tests for brand visual guide agent contracts."""

from app.agents.brand_visual_guide import (
    BrandVisualGuideAgent,
    BrandVisualGuideInput,
    BrandVisualGuideOutput,
)


def test_brand_visual_guide_prompt_includes_required_variables() -> None:
    agent = BrandVisualGuideAgent()
    prompt = agent._build_prompt(
        BrandVisualGuideInput(
            company_name="Acme",
            tone_attributes=["confident", "pragmatic"],
            brand_assets=[{"role": "logo", "object_key": "projects/p1/logo.png"}],
            homepage_visual_signals={"observed_hex_colors": ["#F4E08A"]},
        )
    )

    assert "article_topic" in prompt
    assert "audience" in prompt
    assert "intent" in prompt
    assert "visual_goal" in prompt
    assert "brand_voice" in prompt
    assert "asset_refs" in prompt
    assert "Homepage Visual Signals" in prompt
    assert "#F4E08A" in prompt


def test_brand_visual_guide_output_schema_accepts_strict_payload() -> None:
    payload = {
        "visual_style_guide": {
            "brand_palette": {"primary": "#003366"},
            "contrast_rules": ["Use high contrast text"],
            "composition_rules": ["Single focal point"],
            "subject_rules": ["Show realistic workflows"],
            "camera_lighting_rules": ["Natural light"],
            "logo_usage_rules": ["Keep clear space"],
            "negative_rules": ["Avoid exaggerated claims"],
            "accessibility_rules": ["Readable overlays"],
            "design_tokens": {"colors": {"primary": "#003366"}},
            "component_style_rules": ["Use tokenized spacing"],
            "component_layout_rules": ["Stack on mobile"],
            "component_recipes": [{"name": "hero_card"}],
            "component_negative_rules": ["Avoid framework defaults"],
        },
        "visual_prompt_contract": {
            "template": (
                "{article_topic} {audience} {intent} "
                "{visual_goal} {brand_voice} {asset_refs}"
            ),
            "component_template": (
                "component {article_topic} {audience} {intent} "
                "{visual_goal} {brand_voice} {asset_refs}"
            ),
            "required_variables": [
                "article_topic",
                "audience",
                "intent",
                "visual_goal",
                "brand_voice",
                "asset_refs",
            ],
            "forbidden_terms": ["fake guarantees"],
            "fallback_rules": ["Use neutral composition if no logo is available"],
            "render_targets": ["blog_hero"],
            "render_modes": ["component_render"],
            "component_render_targets": ["hero_card"],
            "component_fallback_rules": ["fallback to template"],
        },
        "extraction_confidence": 0.84,
    }

    result = BrandVisualGuideOutput.model_validate(payload)

    assert result.extraction_confidence == 0.84
    assert result.visual_prompt_contract.required_variables[-1] == "asset_refs"
    assert result.visual_style_guide.component_recipes[0]["name"] == "hero_card"
