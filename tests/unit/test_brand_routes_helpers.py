"""Tests for brand route helper behavior."""

from app.api.v1.brand.routes import (
    _asset_models,
    _icp_niche_models,
    _merge_shallow_dict,
    _product_models,
)


def test_asset_models_filters_incomplete_records() -> None:
    models = _asset_models(
        [
            {
                "asset_id": "a1",
                "object_key": "projects/p/brand-assets/a1.png",
                "sha256": "abc",
                "mime_type": "image/png",
                "byte_size": 120,
                "width": 50,
                "height": 50,
                "role": "logo",
                "role_confidence": 0.9,
                "source_url": "https://example.com/logo.png",
                "origin": "step_01_auto",
                "ingested_at": "2026-02-19T12:00:00+00:00",
            },
            {
                "asset_id": "",
                "object_key": "projects/p/brand-assets/a2.png",
            },
        ]
    )

    assert len(models) == 1
    assert models[0].asset_id == "a1"


def test_merge_shallow_dict_overwrites_top_level_values() -> None:
    merged = _merge_shallow_dict(
        {"a": 1, "nested": {"x": 1}},
        {"b": 2, "nested": {"y": 2}},
    )

    assert merged == {"a": 1, "b": 2, "nested": {"y": 2}}


def test_product_models_filters_invalid_and_maps_expected_fields() -> None:
    products = _product_models(
        [
            {
                "name": "DonkeyBot",
                "description": "AI support agent",
                "category": "software",
                "target_audience": "support teams",
                "core_benefits": ["faster responses", "24/7"],
            },
            {"description": "missing name"},
            "invalid",
        ]
    )

    assert len(products) == 1
    assert products[0].name == "DonkeyBot"
    assert products[0].core_benefits == ["faster responses", "24/7"]


def test_icp_niche_models_filters_invalid_and_maps_expected_fields() -> None:
    niches = _icp_niche_models(
        [
            {
                "niche_name": "Healthcare Support",
                "target_roles": ["Support Lead"],
                "target_industries": ["Healthcare"],
                "company_sizes": ["SMB"],
                "primary_pains": ["ticket backlog"],
                "desired_outcomes": ["faster SLA"],
                "likely_objections": ["compliance"],
                "why_good_fit": "Strong fit for triage automation",
            },
            {"target_roles": ["no niche name"]},
            123,
        ]
    )

    assert len(niches) == 1
    assert niches[0].niche_name == "Healthcare Support"
    assert niches[0].target_roles == ["Support Lead"]
