"""Unit tests for Step 1 ICP suggestion merge behavior."""

from app.services.steps.setup.brand_shared import (
    fallback_visual_confidence,
    merge_target_audience,
    normalize_prompt_contract,
)


def _empty_target_audience() -> dict[str, list[str]]:
    return {
        "target_roles": [],
        "target_industries": [],
        "company_sizes": [],
        "primary_pains": [],
        "desired_outcomes": [],
        "objections": [],
    }


def test_merge_target_audience_deduplicates_and_merges_niches() -> None:
    merged = merge_target_audience(
        extracted_target_audience={
            "target_roles": ["Support Manager", "Operations Lead"],
            "target_industries": ["SaaS"],
            "company_sizes": ["SMB"],
            "primary_pains": ["Ticket backlog"],
            "desired_outcomes": ["Faster response times"],
            "objections": ["Implementation complexity"],
        },
        suggested_icp_niches=[
            {
                "niche_name": "Healthcare Operations",
                "target_roles": ["Operations Lead", "COO"],
                "target_industries": ["Healthcare", "saas"],
                "company_sizes": ["Mid-market"],
                "primary_pains": ["Manual workflows", "ticket backlog"],
                "desired_outcomes": ["Reduce handling time"],
                "likely_objections": ["Data migration effort"],
            }
        ],
    )

    assert merged["target_roles"] == ["Support Manager", "Operations Lead", "COO"]
    assert merged["target_industries"] == ["SaaS", "Healthcare"]
    assert merged["primary_pains"] == ["Ticket backlog", "Manual workflows"]
    assert merged["objections"] == [
        "Implementation complexity",
        "Data migration effort",
    ]


def test_merge_target_audience_caps_roles_to_20_items() -> None:
    extracted = _empty_target_audience()
    extracted["target_roles"] = [f"Role {index}" for index in range(15)]

    suggested = [
        {
            "niche_name": "Adjacent Segment",
            "target_roles": [f"Role {index}" for index in range(10, 30)],
        }
    ]

    merged = merge_target_audience(
        extracted_target_audience=extracted,
        suggested_icp_niches=suggested,
    )

    assert len(merged["target_roles"]) == 20
    assert merged["target_roles"][0] == "Role 0"
    assert merged["target_roles"][-1] == "Role 19"


def test_normalize_prompt_contract_enforces_required_placeholders() -> None:
    contract = normalize_prompt_contract(
        {
            "template": "Brand visual for {article_topic}",
            "required_variables": ["article_topic"],
            "forbidden_terms": ["  fake claims  "],
            "fallback_rules": ["  keep it clean "],
            "render_targets": [" blog_hero "],
        }
    )

    required_variables = contract["required_variables"]
    for required in (
        "article_topic",
        "audience",
        "intent",
        "visual_goal",
        "brand_voice",
        "asset_refs",
    ):
        assert required in required_variables
        assert f"{{{required}}}" in contract["template"]

    assert contract["forbidden_terms"] == ["fake claims"]
    assert contract["fallback_rules"] == ["keep it clean"]
    assert contract["render_targets"] == ["blog_hero"]


def test_fallback_visual_confidence_boosts_when_assets_exist() -> None:
    low_without_assets = fallback_visual_confidence(
        extraction_confidence=0.1,
        has_assets=False,
    )
    low_with_assets = fallback_visual_confidence(
        extraction_confidence=0.1,
        has_assets=True,
    )

    assert low_without_assets == 0.05
    assert low_with_assets == 0.2
