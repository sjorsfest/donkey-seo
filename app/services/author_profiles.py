"""Helpers for author profile assignment and modular document metadata."""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.author_image_store import AuthorImageStore
from app.models.author import Author
from app.models.generated_dtos import AuthorPatchDTO

logger = logging.getLogger(__name__)


def choose_random_author(authors: Sequence[Author]) -> Author | None:
    """Return a random author from a non-empty sequence."""
    if not authors:
        return None
    return random.choice(list(authors))


def author_modular_document_payload(author: Author) -> dict[str, Any]:
    """Serialize an author for article modular-document byline metadata."""
    payload: dict[str, Any] = {
        "id": str(author.id),
        "name": str(author.name or "").strip(),
        "bio": str(author.bio or "").strip(),
        "social_urls": dict(author.social_urls or {}),
        "basic_info": dict(author.basic_info or {}),
    }
    object_key = str(author.profile_image_object_key or "").strip()
    if object_key:
        payload["profile_image"] = {
            "object_key": object_key,
            "mime_type": str(author.profile_image_mime_type or ""),
            "width": author.profile_image_width,
            "height": author.profile_image_height,
            "byte_size": author.profile_image_byte_size,
            "sha256": str(author.profile_image_sha256 or ""),
        }
    return payload


def enrich_modular_document_with_signed_author_image(
    modular_document: dict[str, Any],
) -> dict[str, Any]:
    """Append signed profile image URL to modular-document author payload when possible."""
    if not isinstance(modular_document, dict):
        return {}

    author_payload = modular_document.get("author")
    if not isinstance(author_payload, dict):
        return dict(modular_document)

    profile_image = author_payload.get("profile_image")
    if not isinstance(profile_image, dict):
        return dict(modular_document)

    object_key = str(profile_image.get("object_key") or "").strip()
    if not object_key:
        return dict(modular_document)

    payload = dict(modular_document)
    enriched_author = dict(author_payload)
    enriched_profile_image = dict(profile_image)
    try:
        store = AuthorImageStore()
        enriched_profile_image["signed_url"] = store.create_signed_read_url(object_key=object_key)
    except Exception as exc:
        logger.warning(
            "Failed to add author profile image signed URL",
            extra={"object_key": object_key, "error": str(exc)},
        )
    enriched_author["profile_image"] = enriched_profile_image
    payload["author"] = enriched_author
    return payload


def build_author_profile_image_signed_url(author: Author) -> str | None:
    """Generate an ephemeral signed read URL for author profile image object keys."""
    object_key = str(author.profile_image_object_key or "").strip()
    if not object_key:
        return None
    try:
        store = AuthorImageStore()
        return store.create_signed_read_url(object_key=object_key)
    except Exception as exc:
        logger.warning(
            "Failed to mint author profile image signed URL",
            extra={"author_id": str(author.id), "error": str(exc)},
        )
        return None


async def sync_author_profile_image_from_source(
    *,
    session: AsyncSession,
    author: Author,
    strict: bool,
) -> bool:
    """Sync a source URL into R2-backed profile image metadata for an author."""
    source_url = str(author.profile_image_source_url or "").strip()
    if not source_url:
        author.patch(
            session,
            AuthorPatchDTO.from_partial(
                {
                    "profile_image_object_key": None,
                    "profile_image_mime_type": None,
                    "profile_image_width": None,
                    "profile_image_height": None,
                    "profile_image_byte_size": None,
                    "profile_image_sha256": None,
                }
            ),
        )
        return False

    store = AuthorImageStore()
    try:
        image_metadata = await store.ingest_source_url(
            project_id=str(author.project_id),
            author_id=str(author.id),
            source_url=source_url,
        )
    except Exception as exc:
        if strict:
            raise
        logger.warning(
            "Failed to sync author profile image from source URL",
            extra={"author_id": str(author.id), "source_url": source_url, "error": str(exc)},
        )
        return False

    author.patch(
        session,
        AuthorPatchDTO.from_partial(image_metadata),
    )
    return True
