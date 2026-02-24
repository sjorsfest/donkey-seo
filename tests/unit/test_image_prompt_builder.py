"""Tests for deterministic image prompt builder behavior."""

from app.services.image_prompt_builder import ImagePromptBuilder

_VISUAL_STYLE_GUIDE = {
    "composition_rules": ["Use one focal subject"],
    "negative_rules": ["Avoid clutter"],
    "design_tokens": {"colors": {"primary": "#EF4FA8"}},
    "component_recipes": [{"name": "hero_card"}],
}

_VISUAL_PROMPT_CONTRACT = {
    "template": (
        "Topic: {article_topic}; Audience: {audience}; Intent: {intent}; "
        "Goal: {visual_goal}; Voice: {brand_voice}; Assets: {asset_refs}"
    ),
    "required_variables": [
        "article_topic",
        "audience",
        "intent",
        "visual_goal",
        "brand_voice",
        "asset_refs",
    ],
    "component_template": (
        "Component: {article_topic}; Audience: {audience}; Intent: {intent}; "
        "Goal: {visual_goal}; Voice: {brand_voice}; Assets: {asset_refs}"
    ),
    "render_modes": ["llm_image_generation", "component_render"],
    "component_render_targets": ["hero_card"],
}


def test_image_prompt_builder_is_deterministic() -> None:
    builder = ImagePromptBuilder()
    assets = [
        {
            "asset_id": "1",
            "object_key": "projects/p1/logo.png",
            "role": "logo",
            "role_confidence": 0.9,
        },
        {
            "asset_id": "2",
            "object_key": "projects/p1/icon.png",
            "role": "icon",
            "role_confidence": 0.7,
        },
    ]

    first = builder.build_prompt_payload(
        visual_style_guide=_VISUAL_STYLE_GUIDE,
        visual_prompt_contract=_VISUAL_PROMPT_CONTRACT,
        article_topic="Customer onboarding",
        audience="Operations leaders",
        intent="commercial",
        visual_goal="Convey efficiency",
        brand_voice="clear and practical",
        assets=assets,
    )
    second = builder.build_prompt_payload(
        visual_style_guide=_VISUAL_STYLE_GUIDE,
        visual_prompt_contract=_VISUAL_PROMPT_CONTRACT,
        article_topic="Customer onboarding",
        audience="Operations leaders",
        intent="commercial",
        visual_goal="Convey efficiency",
        brand_voice="clear and practical",
        assets=assets,
    )

    assert first["prompt"] == second["prompt"]
    assert first["asset_refs"][0]["object_key"] == "projects/p1/logo.png"
    assert first["component_render_context"]["enabled"] is True
    assert first["component_render_context"]["component_recipes"][0]["name"] == "hero_card"


def test_image_prompt_builder_adds_signed_urls_when_signer_is_provided() -> None:
    builder = ImagePromptBuilder()

    payload = builder.build_prompt_payload(
        visual_style_guide=_VISUAL_STYLE_GUIDE,
        visual_prompt_contract=_VISUAL_PROMPT_CONTRACT,
        article_topic="Retention strategies",
        audience="CS managers",
        intent="informational",
        visual_goal="Educate with clarity",
        brand_voice="technical",
        assets=[
            {
                "asset_id": "asset-1",
                "object_key": "projects/p1/brand-assets/logo.png",
                "role": "logo",
                "role_confidence": 0.8,
            }
        ],
        sign_asset_url=lambda object_key: f"https://signed.test/{object_key}",
    )

    assert payload["asset_refs"][0]["signed_url"].startswith("https://signed.test/")
    assert "Component: Retention strategies" in payload["component_render_context"]["component_prompt"]
