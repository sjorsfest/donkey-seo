"""Shared featured image generation workflow for content artifacts."""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.featured_image_template import (
    FeaturedImageTemplateAgent,
    FeaturedImageTemplateInput,
    FeaturedImageTemplateSpec,
)
from app.config import settings
from app.integrations.content_image_store import ContentImageStore
from app.models.brand import BrandProfile
from app.models.content import ContentBrief, ContentFeaturedImage
from app.models.generated_dtos import (
    ContentFeaturedImageCreateDTO,
    ContentFeaturedImagePatchDTO,
)
from app.services.featured_image_renderer import FeaturedImageRenderer


@dataclass(slots=True)
class GeneratedFeaturedImage:
    """Result payload from featured image generation."""

    title_text: str
    style_variant_id: str
    template_version: str
    template_spec: dict[str, Any]
    object_key: str
    mime_type: str
    width: int | None
    height: int | None
    byte_size: int
    sha256: str
    source: str


class FeaturedImageGenerationService:
    """Generate, render, and store deterministic featured images from LLM specs."""

    SOURCE = "llm_template_spec"

    def __init__(
        self,
        *,
        template_agent: FeaturedImageTemplateAgent | None = None,
        renderer: FeaturedImageRenderer | None = None,
        image_store: ContentImageStore | None = None,
    ) -> None:
        self.template_agent = template_agent or FeaturedImageTemplateAgent()
        self.renderer = renderer or FeaturedImageRenderer()
        self.image_store = image_store or ContentImageStore()

    @staticmethod
    def locked_title_for_brief(brief: ContentBrief) -> str:
        """Return canonical title used by image + article generation."""
        for candidate in list(brief.working_titles or []):
            title = " ".join(str(candidate).split()).strip()
            if title:
                return title

        keyword = " ".join(str(brief.primary_keyword or "").split()).strip()
        if not keyword:
            return "Untitled"

        words = [word for word in re.split(r"\s+", keyword) if word]
        title = " ".join(words[:12]).strip()
        return title.title() if title else "Untitled"

    async def generate_for_brief(
        self,
        *,
        session: AsyncSession,
        project_id: str,
        brief: ContentBrief,
        brand: BrandProfile | None,
        existing: ContentFeaturedImage | None = None,
        force_regenerate: bool = False,
    ) -> ContentFeaturedImage:
        """Generate and persist featured image metadata for one brief."""
        title_text = self.locked_title_for_brief(brief)

        if (
            existing is not None
            and not force_regenerate
            and existing.status == "ready"
            and bool(existing.object_key)
            and existing.title_text == title_text
        ):
            return existing

        logo_asset = self._select_logo_asset(brand)
        logo_data_url = self._load_logo_data_url(logo_asset)

        template = await self._build_template(
            brief=brief,
            brand=brand,
            title_text=title_text,
            logo_asset=logo_asset,
        )

        rendered = await self.renderer.render_png(
            template=template,
            title_text=title_text,
            logo_data_url=logo_data_url,
        )

        storage_meta = await self.image_store.upload_generated_image(
            project_id=project_id,
            brief_id=str(brief.id),
            payload=rendered.payload,
            source=self.SOURCE,
        )

        generated = GeneratedFeaturedImage(
            title_text=title_text,
            style_variant_id=self._normalize_variant_id(template.style_variant_id, brief_id=str(brief.id)),
            template_version=template.template_version,
            template_spec=template.model_dump(),
            object_key=str(storage_meta["object_key"]),
            mime_type=str(storage_meta["mime_type"]),
            width=storage_meta.get("width"),
            height=storage_meta.get("height"),
            byte_size=int(storage_meta["byte_size"]),
            sha256=str(storage_meta["sha256"]),
            source=str(storage_meta["source"]),
        )

        if existing is None:
            existing = ContentFeaturedImage.create(
                session,
                ContentFeaturedImageCreateDTO(
                    project_id=project_id,
                    brief_id=str(brief.id),
                    status="ready",
                    title_text=generated.title_text,
                    style_variant_id=generated.style_variant_id,
                    template_version=generated.template_version,
                    template_spec=generated.template_spec,
                    object_key=generated.object_key,
                    mime_type=generated.mime_type,
                    width=generated.width,
                    height=generated.height,
                    byte_size=generated.byte_size,
                    sha256=generated.sha256,
                    source=generated.source,
                    generation_error=None,
                    last_generated_at=datetime.now(timezone.utc),
                ),
            )
        else:
            existing.patch(
                session,
                ContentFeaturedImagePatchDTO.from_partial(
                    {
                        "status": "ready",
                        "title_text": generated.title_text,
                        "style_variant_id": generated.style_variant_id,
                        "template_version": generated.template_version,
                        "template_spec": generated.template_spec,
                        "object_key": generated.object_key,
                        "mime_type": generated.mime_type,
                        "width": generated.width,
                        "height": generated.height,
                        "byte_size": generated.byte_size,
                        "sha256": generated.sha256,
                        "source": generated.source,
                        "generation_error": None,
                        "last_generated_at": datetime.now(timezone.utc),
                    }
                ),
            )

        return existing

    async def mark_failure(
        self,
        *,
        session: AsyncSession,
        project_id: str,
        brief_id: str,
        title_text: str,
        error_message: str,
        existing: ContentFeaturedImage | None = None,
    ) -> ContentFeaturedImage:
        """Persist failed status for auditability when generation exhausts retries."""
        if existing is None:
            return ContentFeaturedImage.create(
                session,
                ContentFeaturedImageCreateDTO(
                    project_id=project_id,
                    brief_id=brief_id,
                    status="failed",
                    title_text=title_text,
                    generation_error=error_message,
                    last_generated_at=datetime.now(timezone.utc),
                ),
            )

        existing.patch(
            session,
            ContentFeaturedImagePatchDTO.from_partial(
                {
                    "status": "failed",
                    "title_text": title_text,
                    "generation_error": error_message,
                    "last_generated_at": datetime.now(timezone.utc),
                }
            ),
        )
        return existing

    async def _build_template(
        self,
        *,
        brief: ContentBrief,
        brand: BrandProfile | None,
        title_text: str,
        logo_asset: dict[str, Any] | None,
    ) -> FeaturedImageTemplateSpec:
        style_guide = dict(brand.visual_style_guide or {}) if brand else {}
        assets = self._summarize_assets_for_prompt(brand, logo_asset)

        output = await self.template_agent.run(
            FeaturedImageTemplateInput(
                article_title=title_text,
                primary_keyword=str(brief.primary_keyword or ""),
                audience=str(brief.target_audience or ""),
                search_intent=str(brief.search_intent or ""),
                page_type=str(brief.page_type or ""),
                brand_name=str((brand.company_name if brand else "") or "Brand"),
                brand_tagline=str(brand.tagline or "") if brand else None,
                visual_style_guide=style_guide,
                brand_assets=assets,
            )
        )

        return self._apply_template_guardrails(output.template)

    def _apply_template_guardrails(self, template: FeaturedImageTemplateSpec) -> FeaturedImageTemplateSpec:
        """Apply deterministic constraints after LLM output parsing."""
        safe_margin = max(16, min(template.safe_margin_px, 96))
        template.safe_margin_px = safe_margin

        # Ensure title zone remains readable and reasonably dominant.
        template.title_zone.width = max(0.35, min(template.title_zone.width, 0.9))
        template.title_zone.height = max(0.30, min(template.title_zone.height, 0.85))

        if len(template.shapes) > 8:
            template.shapes = template.shapes[:8]

        return template

    def _select_logo_asset(self, brand: BrandProfile | None) -> dict[str, Any] | None:
        if brand is None:
            return None

        assets = [item for item in list(brand.brand_assets or []) if isinstance(item, dict)]
        logos = [
            item
            for item in assets
            if str(item.get("role") or "").strip().lower() == "logo"
            and str(item.get("object_key") or "").strip()
        ]
        logos.sort(key=lambda item: float(item.get("role_confidence") or 0.0), reverse=True)
        return logos[0] if logos else None

    def _load_logo_data_url(self, logo_asset: dict[str, Any] | None) -> str | None:
        if not logo_asset:
            return None

        object_key = str(logo_asset.get("object_key") or "").strip()
        if not object_key:
            return None

        payload, mime_type = self.image_store.read_object_bytes(object_key=object_key)
        resolved_mime = mime_type or str(logo_asset.get("mime_type") or "image/png")
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{resolved_mime};base64,{encoded}"

    def _summarize_assets_for_prompt(
        self,
        brand: BrandProfile | None,
        logo_asset: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        assets = [item for item in list((brand.brand_assets if brand else []) or []) if isinstance(item, dict)]
        prioritized: list[dict[str, Any]] = []

        if logo_asset is not None:
            prioritized.append(
                {
                    "role": str(logo_asset.get("role") or "logo"),
                    "role_confidence": float(logo_asset.get("role_confidence") or 0.0),
                    "object_key": str(logo_asset.get("object_key") or ""),
                    "dominant_colors": list(logo_asset.get("dominant_colors") or []),
                }
            )

        for asset in assets:
            if len(prioritized) >= 4:
                break
            object_key = str(asset.get("object_key") or "").strip()
            if not object_key:
                continue
            if logo_asset is not None and object_key == str(logo_asset.get("object_key") or ""):
                continue
            prioritized.append(
                {
                    "role": str(asset.get("role") or "reference"),
                    "role_confidence": float(asset.get("role_confidence") or 0.0),
                    "object_key": object_key,
                    "dominant_colors": list(asset.get("dominant_colors") or []),
                }
            )

        return prioritized

    @staticmethod
    def _normalize_variant_id(raw_variant_id: str, *, brief_id: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw_variant_id or "").strip()).strip("-")
        if normalized:
            return normalized[:80]
        return f"brief-{brief_id[:16]}"


async def retry_with_backoff(
    *,
    attempts: int,
    backoff_ms: int,
    coro_factory: Any,
) -> Any:
    """Retry helper for async generation flows."""
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return await coro_factory()
        except Exception as exc:  # pragma: no cover - error path exercised by step tests
            last_error = exc
            if attempt >= attempts - 1:
                break
            await asyncio.sleep((backoff_ms / 1000.0) * (attempt + 1))
    assert last_error is not None
    raise last_error


def modular_featured_image_payload(
    *,
    featured_image: ContentFeaturedImage,
    signed_url: str | None = None,
) -> dict[str, Any]:
    """Build stable modular_document payload for featured image metadata."""
    payload: dict[str, Any] = {
        "object_key": str(featured_image.object_key or ""),
        "mime_type": str(featured_image.mime_type or "image/png"),
        "width": featured_image.width,
        "height": featured_image.height,
        "byte_size": featured_image.byte_size,
        "sha256": str(featured_image.sha256 or ""),
        "title_text": featured_image.title_text,
        "template_version": str(featured_image.template_version or ""),
        "source": str(featured_image.source or ""),
        "style_variant_id": str(featured_image.style_variant_id or ""),
    }
    if signed_url:
        payload["signed_url"] = signed_url
    return payload


def generation_retry_settings() -> tuple[int, int]:
    """Resolve retry settings for featured image generation."""
    return (
        max(1, int(settings.content_image_retry_attempts)),
        max(0, int(settings.content_image_retry_backoff_ms)),
    )
