"""Brand visual context API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from app.api.v1.brand.constants import (
    BRAND_ASSET_NOT_FOUND_DETAIL,
    BRAND_PROFILE_NOT_FOUND_DETAIL,
)
from app.api.v1.dependencies import get_user_project
from app.core.ids import generate_cuid
from app.dependencies import CurrentUser, DbSession
from app.integrations.asset_store import BrandAssetStore
from app.models.brand import BrandProfile
from app.models.generated_dtos import BrandProfilePatchDTO
from app.schemas.brand import (
    BrandAssetAddRequest,
    BrandAssetIngestRequest,
    BrandAssetIngestResponse,
    BrandAssetMetadata,
    BrandAssetRemoveResponse,
    BrandAssetSignedUploadRequest,
    BrandAssetSignedUploadResponse,
    BrandProductServiceMetadata,
    BrandAssetSignedReadUrlResponse,
    BrandSuggestedICPNiche,
    BrandVisualContextResponse,
    BrandVisualStylePatchRequest,
)
from app.services.steps.setup.brand_shared import normalize_prompt_contract

router = APIRouter()


@router.get(
    "/{project_id}/visual-context",
    response_model=BrandVisualContextResponse,
    summary="Get brand visual context",
)
async def get_brand_visual_context(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> BrandVisualContextResponse:
    """Return reusable visual context for image generation."""
    await get_user_project(project_id, current_user, session)
    brand = await _get_brand_profile_or_404(project_id=project_id, session=session)

    return _to_visual_context_response(project_id=project_id, brand=brand)


@router.post(
    "/{project_id}/assets/ingest",
    response_model=BrandAssetIngestResponse,
    summary="Ingest brand assets from URLs",
)
async def ingest_brand_assets(
    project_id: str,
    payload: BrandAssetIngestRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> BrandAssetIngestResponse:
    """Legacy URL ingestion endpoint (disabled in favor of client-side uploads)."""
    await get_user_project(project_id, current_user, session)
    await _get_brand_profile_or_404(project_id=project_id, session=session)
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=(
            "Server-side URL ingestion is disabled. "
            "Use /brand/{project_id}/assets/signed-upload-url and then POST /brand/{project_id}/assets."
        ),
    )


@router.post(
    "/{project_id}/assets",
    response_model=BrandAssetIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Attach an uploaded brand asset",
)
async def add_brand_asset(
    project_id: str,
    payload: BrandAssetAddRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> BrandAssetIngestResponse:
    """Attach metadata for a client-uploaded brand asset object."""
    await get_user_project(project_id, current_user, session)
    brand = await _get_brand_profile_or_404(project_id=project_id, session=session)
    _validate_upload_object_key(project_id=project_id, asset_id=payload.asset_id, object_key=payload.object_key)

    existing_assets = [
        item
        for item in list(brand.brand_assets or [])
        if isinstance(item, dict)
    ]
    existing_hashes = {
        str(item.get("sha256") or "").strip().lower()
        for item in existing_assets
        if isinstance(item, dict)
    }

    # Keep max-count behavior explicit for manual uploads.
    is_replacement = any(str(item.get("asset_id") or "") == payload.asset_id for item in existing_assets)
    store = BrandAssetStore()
    if not is_replacement and len(existing_assets) >= store.settings.brand_assets_max_count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Brand asset limit reached. Remove an asset before adding a new one.",
        )

    duplicate_by_sha = _find_asset_by_sha(
        raw_assets=existing_assets,
        sha256=payload.sha256,
        exclude_asset_id=payload.asset_id,
    )
    if duplicate_by_sha:
        if str(duplicate_by_sha.get("object_key") or "") != payload.object_key:
            try:
                store.delete_object(object_key=payload.object_key)
            except Exception:
                # Preserve successful metadata response even if duplicate cleanup fails.
                pass
        return BrandAssetIngestResponse(
            ingested_count=0,
            total_assets=len(existing_assets),
            brand_assets=_asset_models(existing_assets),
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    new_asset: dict[str, Any] = {
        "asset_id": payload.asset_id,
        "object_key": payload.object_key,
        "sha256": payload.sha256,
        "mime_type": payload.content_type,
        "byte_size": payload.byte_size,
        "width": payload.width,
        "height": payload.height,
        "dominant_colors": payload.dominant_colors,
        "average_luminance": payload.average_luminance,
        "role": payload.role,
        "role_confidence": payload.role_confidence,
        "source_url": f"client_upload://{payload.asset_id}",
        "origin": "manual_upload",
        "ingested_at": now_iso,
    }

    merged_assets = [
        item
        for item in existing_assets
        if str(item.get("asset_id") or "") != payload.asset_id
    ]
    merged_assets.append(new_asset)
    merged_assets.sort(key=lambda item: float(item.get("role_confidence") or 0.0), reverse=True)

    brand.patch(
        session,
        BrandProfilePatchDTO.from_partial(
            {
                "brand_assets": merged_assets,
                "visual_last_synced_at": datetime.now(timezone.utc),
            }
        ),
    )
    await session.flush()
    await session.refresh(brand)

    ingested_count = 1 if payload.sha256 not in existing_hashes else 0
    return BrandAssetIngestResponse(
        ingested_count=ingested_count,
        total_assets=len(merged_assets),
        brand_assets=_asset_models(merged_assets),
    )


@router.post(
    "/{project_id}/assets/signed-upload-url",
    response_model=BrandAssetSignedUploadResponse,
    summary="Get signed upload URL for a brand asset",
    description="Mint a short-lived signed PUT URL so clients can upload a brand asset directly.",
)
async def get_brand_asset_signed_upload_url(
    project_id: str,
    payload: BrandAssetSignedUploadRequest,
    current_user: CurrentUser,
    session: DbSession,
    ttl_seconds: int | None = Query(default=None, ge=1, le=3600),
) -> BrandAssetSignedUploadResponse:
    """Mint a signed upload URL for direct client-side brand asset uploads."""
    await get_user_project(project_id, current_user, session)
    await _get_brand_profile_or_404(project_id=project_id, session=session)

    asset_id = generate_cuid()
    object_key = _build_brand_asset_upload_object_key(
        project_id=project_id,
        asset_id=asset_id,
        content_type=payload.content_type,
    )
    store = BrandAssetStore()
    upload_url = store.create_signed_upload_url(
        object_key=object_key,
        ttl_seconds=ttl_seconds,
        content_type=payload.content_type,
    )
    expires_in = int(ttl_seconds or store.settings.signed_url_ttl_seconds)

    return BrandAssetSignedUploadResponse(
        asset_id=asset_id,
        object_key=object_key,
        upload_url=upload_url,
        expires_in_seconds=expires_in,
        required_headers={"Content-Type": payload.content_type},
    )


@router.delete(
    "/{project_id}/assets/{asset_id}",
    response_model=BrandAssetRemoveResponse,
    summary="Remove a brand asset",
)
async def remove_brand_asset(
    project_id: str,
    asset_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> BrandAssetRemoveResponse:
    """Remove a brand asset from metadata and private storage."""
    await get_user_project(project_id, current_user, session)
    brand = await _get_brand_profile_or_404(project_id=project_id, session=session)

    existing_assets = [
        item
        for item in list(brand.brand_assets or [])
        if isinstance(item, dict)
    ]
    asset = _find_asset_by_id(existing_assets, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=BRAND_ASSET_NOT_FOUND_DETAIL,
        )

    object_key = str(asset.get("object_key") or "").strip()
    if object_key:
        try:
            BrandAssetStore().delete_object(object_key=object_key)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to remove brand asset from storage: {exc}",
            ) from exc

    remaining_assets = [
        item
        for item in existing_assets
        if str(item.get("asset_id") or "") != asset_id
    ]

    brand.patch(
        session,
        BrandProfilePatchDTO.from_partial(
            {
                "brand_assets": remaining_assets,
                "visual_last_synced_at": datetime.now(timezone.utc),
            }
        ),
    )
    await session.flush()
    await session.refresh(brand)

    return BrandAssetRemoveResponse(
        removed_asset_id=asset_id,
        total_assets=len(remaining_assets),
        brand_assets=_asset_models(remaining_assets),
    )


@router.post(
    "/{project_id}/assets/{asset_id}/signed-read-url",
    response_model=BrandAssetSignedReadUrlResponse,
    summary="Get signed read URL for a private brand asset",
)
async def get_brand_asset_signed_read_url(
    project_id: str,
    asset_id: str,
    current_user: CurrentUser,
    session: DbSession,
    ttl_seconds: int | None = Query(default=None, ge=1, le=3600),
) -> BrandAssetSignedReadUrlResponse:
    """Mint a short-lived signed URL for a brand asset object key."""
    await get_user_project(project_id, current_user, session)
    brand = await _get_brand_profile_or_404(project_id=project_id, session=session)

    asset = _find_asset_by_id(brand.brand_assets or [], asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=BRAND_ASSET_NOT_FOUND_DETAIL,
        )

    object_key = str(asset.get("object_key") or "").strip()
    if not object_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=BRAND_ASSET_NOT_FOUND_DETAIL,
        )

    store = BrandAssetStore()
    signed_url = store.create_signed_read_url(object_key=object_key, ttl_seconds=ttl_seconds)
    expires_in = int(ttl_seconds or store.settings.signed_url_ttl_seconds)

    return BrandAssetSignedReadUrlResponse(
        asset_id=asset_id,
        object_key=object_key,
        expires_in_seconds=expires_in,
        signed_url=signed_url,
    )


@router.patch(
    "/{project_id}/visual-style",
    response_model=BrandVisualContextResponse,
    summary="Patch visual style guide and prompt contract",
)
async def patch_brand_visual_style(
    project_id: str,
    payload: BrandVisualStylePatchRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> BrandVisualContextResponse:
    """Apply manual overrides to visual style fields."""
    await get_user_project(project_id, current_user, session)
    brand = await _get_brand_profile_or_404(project_id=project_id, session=session)

    update_data: dict[str, object] = {"visual_last_synced_at": datetime.now(timezone.utc)}

    if payload.visual_style_guide is not None:
        update_data["visual_style_guide"] = _merge_shallow_dict(
            brand.visual_style_guide or {},
            payload.visual_style_guide,
        )

    if payload.visual_prompt_contract is not None:
        merged_contract = _merge_shallow_dict(
            brand.visual_prompt_contract or {},
            payload.visual_prompt_contract,
        )
        update_data["visual_prompt_contract"] = normalize_prompt_contract(dict(merged_contract))

    brand.patch(session, BrandProfilePatchDTO.from_partial(update_data))
    await session.flush()
    await session.refresh(brand)

    return _to_visual_context_response(project_id=project_id, brand=brand)


async def _get_brand_profile_or_404(*, project_id: str, session: DbSession) -> BrandProfile:
    result = await session.execute(
        select(BrandProfile).where(BrandProfile.project_id == project_id)
    )
    brand = result.scalar_one_or_none()
    if not brand:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=BRAND_PROFILE_NOT_FOUND_DETAIL,
        )
    return brand


def _asset_models(raw_assets: list[dict] | None) -> list[BrandAssetMetadata]:
    models: list[BrandAssetMetadata] = []
    for asset in raw_assets or []:
        if not isinstance(asset, dict):
            continue
        required = (
            "asset_id",
            "object_key",
            "sha256",
            "mime_type",
            "byte_size",
            "role",
            "role_confidence",
            "source_url",
            "origin",
            "ingested_at",
        )
        if any(asset.get(field) in (None, "") for field in required):
            continue
        models.append(BrandAssetMetadata.model_validate(asset))
    return models


def _product_models(raw_products: list[dict] | None) -> list[BrandProductServiceMetadata]:
    models: list[BrandProductServiceMetadata] = []
    for product in raw_products or []:
        if not isinstance(product, dict):
            continue
        name = str(product.get("name") or "").strip()
        if not name:
            continue
        core_benefits = [
            str(item).strip()
            for item in product.get("core_benefits", [])
            if str(item).strip()
        ]
        models.append(
            BrandProductServiceMetadata(
                name=name,
                description=_to_optional_str(product.get("description")),
                category=_to_optional_str(product.get("category")),
                target_audience=_to_optional_str(product.get("target_audience")),
                core_benefits=core_benefits,
            )
        )
    return models


def _icp_niche_models(raw_niches: list[dict] | None) -> list[BrandSuggestedICPNiche]:
    models: list[BrandSuggestedICPNiche] = []
    for niche in raw_niches or []:
        if not isinstance(niche, dict):
            continue
        niche_name = str(niche.get("niche_name") or "").strip()
        if not niche_name:
            continue
        models.append(
            BrandSuggestedICPNiche(
                niche_name=niche_name,
                target_roles=_to_str_list(niche.get("target_roles")),
                target_industries=_to_str_list(niche.get("target_industries")),
                company_sizes=_to_str_list(niche.get("company_sizes")),
                primary_pains=_to_str_list(niche.get("primary_pains")),
                desired_outcomes=_to_str_list(niche.get("desired_outcomes")),
                likely_objections=_to_str_list(niche.get("likely_objections")),
                why_good_fit=_to_optional_str(niche.get("why_good_fit")),
            )
        )
    return models


def _to_optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _to_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _to_visual_context_response(*, project_id: str, brand: BrandProfile) -> BrandVisualContextResponse:
    return BrandVisualContextResponse(
        project_id=str(project_id),
        company_name=brand.company_name,
        tagline=brand.tagline,
        products_services=_product_models(brand.products_services),
        target_roles=_to_str_list(brand.target_roles),
        target_industries=_to_str_list(brand.target_industries),
        differentiators=_to_str_list(brand.differentiators),
        suggested_icp_niches=_icp_niche_models(brand.suggested_icp_niches),
        extraction_confidence=brand.extraction_confidence,
        brand_assets=_asset_models(brand.brand_assets),
        visual_style_guide=brand.visual_style_guide or {},
        visual_prompt_contract=brand.visual_prompt_contract or {},
        visual_extraction_confidence=brand.visual_extraction_confidence,
        visual_last_synced_at=brand.visual_last_synced_at,
    )


def _find_asset_by_id(raw_assets: list[dict], asset_id: str) -> dict | None:
    for asset in raw_assets:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("asset_id") or "") == asset_id:
            return asset
    return None


def _find_asset_by_sha(
    *,
    raw_assets: list[dict],
    sha256: str,
    exclude_asset_id: str | None = None,
) -> dict | None:
    normalized_sha = str(sha256 or "").strip().lower()
    excluded = str(exclude_asset_id or "").strip()
    if not normalized_sha:
        return None
    for asset in raw_assets:
        if not isinstance(asset, dict):
            continue
        if excluded and str(asset.get("asset_id") or "").strip() == excluded:
            continue
        if str(asset.get("sha256") or "").strip().lower() == normalized_sha:
            return asset
    return None


def _build_brand_asset_upload_object_key(*, project_id: str, asset_id: str, content_type: str) -> str:
    extension = BrandAssetStore.extension_for_mime_type(content_type)
    return f"projects/{project_id}/brand-assets/uploads/{asset_id}{extension}"


def _validate_upload_object_key(*, project_id: str, asset_id: str, object_key: str) -> None:
    expected_prefix = f"projects/{project_id}/brand-assets/uploads/"
    normalized_key = str(object_key or "").strip()
    if not normalized_key.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid object_key for project brand asset upload.",
        )
    expected_base = f"{expected_prefix}{asset_id}"
    if not normalized_key.startswith(expected_base):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="object_key does not match asset_id.",
        )


def _merge_shallow_dict(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        merged[key] = value
    return merged
