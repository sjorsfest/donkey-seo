"""Unit tests for featured image template schema constraints."""

from __future__ import annotations

import pytest

from app.agents.featured_image_template import FeaturedImageTemplateSpec


def _valid_template_payload() -> dict:
    return {
        "style_variant_id": "variant-01",
        "safe_margin_px": 32,
        "background_color": "#F8F9FF",
        "title_zone": {
            "x": 0.08,
            "y": 0.12,
            "width": 0.68,
            "height": 0.62,
            "padding_px": 40,
            "typography": {
                "font_family": "system-ui, sans-serif",
                "font_size_px": 72,
                "font_weight": 700,
                "line_height": 1.1,
                "letter_spacing_em": 0.0,
                "color": "#0B1020",
                "max_lines": 3,
                "align": "left",
            },
        },
        "logo_zone": {
            "enabled": True,
            "include_if_logo_available": True,
            "x": 0.78,
            "y": 0.80,
            "width": 0.16,
            "height": 0.12,
            "opacity": 1.0,
            "anchor": "bottom-right",
        },
        "shapes": [
            {
                "shape_type": "rounded_rect",
                "color": "#2F6BFF",
                "opacity": 0.14,
                "x": 0.76,
                "y": 0.10,
                "width": 0.20,
                "height": 0.30,
                "blur_px": 0,
                "rotation_deg": 12,
                "border_radius_px": 42,
            }
        ],
    }


def test_template_spec_accepts_valid_payload() -> None:
    parsed = FeaturedImageTemplateSpec.model_validate(_valid_template_payload())

    assert parsed.style_variant_id == "variant-01"
    assert len(parsed.shapes) == 1


def test_template_spec_rejects_invalid_shape_type() -> None:
    payload = _valid_template_payload()
    payload["shapes"][0]["shape_type"] = "triangle"

    with pytest.raises(Exception):
        FeaturedImageTemplateSpec.model_validate(payload)


def test_template_spec_rejects_more_than_max_shapes() -> None:
    payload = _valid_template_payload()
    payload["shapes"] = payload["shapes"] * 9

    with pytest.raises(Exception):
        FeaturedImageTemplateSpec.model_validate(payload)
