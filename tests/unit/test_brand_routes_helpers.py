"""Tests for brand route helper behavior."""

import pytest
from fastapi import HTTPException, status

from app.api.v1.brand.routes import (
    _asset_models,
    _build_brand_asset_upload_object_key,
    _find_asset_by_id,
    _find_asset_by_sha,
    _icp_niche_models,
    _merge_shallow_dict,
    _product_models,
    _validate_upload_object_key,
)
from app.schemas.brand import (
    BrandAssetAddRequest,
    BrandAssetIngestRequest,
    BrandAssetSignedUploadRequest,
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


def test_find_asset_by_id_matches_expected_record() -> None:
    asset = _find_asset_by_id(
        raw_assets=[
            {"asset_id": "a1", "object_key": "projects/p1/brand-assets/a1.png"},
            {"asset_id": "a2", "object_key": "projects/p1/brand-assets/a2.png"},
        ],
        asset_id="a2",
    )
    assert asset is not None
    assert asset["asset_id"] == "a2"


def test_brand_asset_add_request_normalizes_and_validates_image_content_type() -> None:
    payload = BrandAssetAddRequest(
        asset_id="asset_1",
        object_key="projects/p1/brand-assets/uploads/asset_1.png",
        content_type="IMAGE/PNG; charset=utf-8",
        byte_size=2048,
        sha256="ABC12345",
        role="logo",
    )
    assert payload.content_type == "image/png"
    assert payload.sha256 == "abc12345"


def test_brand_asset_ingest_request_rejects_non_http_url() -> None:
    with pytest.raises(ValueError):
        BrandAssetIngestRequest(source_urls=["ftp://example.com/logo.png"])


def test_brand_asset_signed_upload_request_rejects_non_image_content_type() -> None:
    with pytest.raises(ValueError):
        BrandAssetSignedUploadRequest(content_type="application/pdf")


def test_find_asset_by_sha_ignores_excluded_asset_id() -> None:
    asset = _find_asset_by_sha(
        raw_assets=[
            {"asset_id": "a1", "sha256": "abc"},
            {"asset_id": "a2", "sha256": "def"},
        ],
        sha256="abc",
        exclude_asset_id="a1",
    )
    assert asset is None


def test_build_brand_asset_upload_object_key_uses_uploads_prefix() -> None:
    key = _build_brand_asset_upload_object_key(
        project_id="project_1",
        asset_id="asset_1",
        content_type="image/png",
    )
    assert key == "projects/project_1/brand-assets/uploads/asset_1.png"


def test_validate_upload_object_key_rejects_wrong_prefix() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _validate_upload_object_key(
            project_id="project_1",
            asset_id="asset_1",
            object_key="projects/project_2/brand-assets/uploads/asset_1.png",
        )
    assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
