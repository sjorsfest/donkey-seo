"""Tests for brand route helper behavior."""

from app.api.v1.brand.routes import _asset_models, _merge_shallow_dict


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
