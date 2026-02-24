"""Publication webhook scheduling and delivery services."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_context
from app.integrations.content_image_store import ContentImageStore
from app.models.content import (
    ContentArticle,
    ContentArticleVersion,
    ContentBrief,
    PublicationWebhookDelivery,
)
from app.models.generated_dtos import (
    PublicationWebhookDeliveryCreateDTO,
    PublicationWebhookDeliveryPatchDTO,
)
from app.models.project import Project

logger = logging.getLogger(__name__)

PUBLICATION_WEBHOOK_EVENT_TYPE = "content.article.publish_requested"
PUBLICATION_WEBHOOK_PENDING_STATUSES = {"pending", "retrying"}
PUBLICATION_WEBHOOK_TERMINAL_STATUSES = {"delivered", "failed", "canceled"}
PUBLICATION_WEBHOOK_MAX_ATTEMPTS = 5
PUBLICATION_WEBHOOK_BASE_BACKOFF_SECONDS = 60
PUBLICATION_WEBHOOK_DISPATCH_HOUR_UTC = 9
PUBLICATION_WEBHOOK_CLAIM_LEASE_SECONDS = 30


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def scheduled_publication_datetime(publication_date: date) -> datetime:
    """Resolve publication date to default webhook dispatch timestamp."""
    return datetime.combine(
        publication_date,
        time(hour=PUBLICATION_WEBHOOK_DISPATCH_HOUR_UTC, tzinfo=timezone.utc),
    )


def calculate_publication_retry_delay_seconds(attempt_count: int) -> int:
    """Return retry delay for 1-indexed attempt count."""
    normalized_attempt = max(1, int(attempt_count))
    return PUBLICATION_WEBHOOK_BASE_BACKOFF_SECONDS * (2 ** (normalized_attempt - 1))


def sign_publication_webhook_payload(
    *,
    secret: str,
    timestamp: str,
    raw_body: bytes,
) -> str:
    """Create HMAC signature for outbound webhook payload."""
    message = timestamp.encode("utf-8") + b"." + raw_body
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def build_publication_webhook_payload(
    *,
    delivery: PublicationWebhookDelivery,
    project: Project,
    article: ContentArticle,
    article_version: ContentArticleVersion,
    brief: ContentBrief,
    occurred_at: datetime,
) -> dict[str, Any]:
    """Build webhook payload for publish-request event delivery."""
    modular_document = _enrich_modular_document_with_signed_featured_image(
        article_version.modular_document or {}
    )
    return {
        "event_id": str(delivery.id),
        "event_type": PUBLICATION_WEBHOOK_EVENT_TYPE,
        "occurred_at": occurred_at.isoformat(),
        "project": {
            "id": str(project.id),
            "domain": project.domain,
            "locale": project.primary_locale,
        },
        "article": {
            "article_id": str(article.id),
            "brief_id": str(article.brief_id),
            "version_number": article_version.version_number,
            "title": article_version.title,
            "slug": article_version.slug,
            "primary_keyword": article_version.primary_keyword,
            "proposed_publication_date": (
                brief.proposed_publication_date.isoformat()
                if brief.proposed_publication_date is not None
                else None
            ),
        },
        "modular_document": modular_document,
        "rendered_html": article_version.rendered_html,
    }


def _enrich_modular_document_with_signed_featured_image(
    modular_document: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(modular_document, dict):
        return {}

    featured_image = modular_document.get("featured_image")
    if not isinstance(featured_image, dict):
        return dict(modular_document)

    object_key = str(featured_image.get("object_key") or "").strip()
    if not object_key:
        return dict(modular_document)

    payload = dict(modular_document)
    enriched_featured_image = dict(featured_image)
    try:
        store = ContentImageStore()
        enriched_featured_image["signed_url"] = store.create_signed_read_url(object_key=object_key)
    except Exception as exc:
        logger.warning(
            "Failed to add featured image signed URL to webhook payload",
            extra={"object_key": object_key, "error": str(exc)},
        )
    payload["featured_image"] = enriched_featured_image
    return payload


def apply_publication_delivery_attempt_result(
    *,
    delivery: PublicationWebhookDelivery,
    attempted_at: datetime,
    success: bool,
    http_status: int | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Mutate delivery attempt state after one webhook send attempt."""
    attempt_count = int(delivery.attempt_count or 0) + 1
    payload: dict[str, Any] = {
        "attempt_count": attempt_count,
        "last_attempt_at": attempted_at,
        "last_http_status": http_status,
        "last_error": error_message,
    }

    if success:
        payload.update(
            {
                "status": "delivered",
                "delivered_at": attempted_at,
                "next_attempt_at": attempted_at,
                "last_error": None,
            }
        )
    elif attempt_count >= PUBLICATION_WEBHOOK_MAX_ATTEMPTS:
        payload.update(
            {
                "status": "failed",
                "next_attempt_at": attempted_at,
            }
        )
    else:
        payload.update(
            {
                "status": "retrying",
                "next_attempt_at": attempted_at
                + timedelta(
                    seconds=calculate_publication_retry_delay_seconds(attempt_count)
                ),
            }
        )

    return payload


async def _get_publication_delivery(
    session: AsyncSession,
    *,
    article_id: str,
) -> PublicationWebhookDelivery | None:
    result = await session.execute(
        select(PublicationWebhookDelivery).where(
            PublicationWebhookDelivery.article_id == article_id,
            PublicationWebhookDelivery.event_type == PUBLICATION_WEBHOOK_EVENT_TYPE,
        )
    )
    return result.scalar_one_or_none()


async def schedule_publication_webhook_for_article(
    session: AsyncSession,
    *,
    article: ContentArticle,
    proposed_publication_date: date | None,
) -> PublicationWebhookDelivery | None:
    """Create or refresh publication delivery for an article."""
    delivery = await _get_publication_delivery(session, article_id=str(article.id))
    now = _utc_now()

    is_published = (
        article.publish_status == "published" or article.published_at is not None
    )
    if is_published or proposed_publication_date is None:
        if delivery and delivery.status in PUBLICATION_WEBHOOK_PENDING_STATUSES:
            delivery.patch(
                session,
                PublicationWebhookDeliveryPatchDTO.from_partial(
                    {
                        "status": "canceled",
                        "next_attempt_at": now,
                        "last_error": (
                            "article_already_published"
                            if is_published
                            else "publication_date_missing"
                        ),
                    }
                ),
            )
        return delivery

    scheduled_for = scheduled_publication_datetime(proposed_publication_date)
    reset_payload: dict[str, Any] = {
        "project_id": str(article.project_id),
        "article_id": str(article.id),
        "event_type": PUBLICATION_WEBHOOK_EVENT_TYPE,
        "scheduled_for": scheduled_for,
        "status": "pending",
        "attempt_count": 0,
        "next_attempt_at": scheduled_for,
        "last_attempt_at": None,
        "delivered_at": None,
        "last_http_status": None,
        "last_error": None,
    }

    if delivery is None:
        return PublicationWebhookDelivery.create(
            session,
            PublicationWebhookDeliveryCreateDTO(
                project_id=str(article.project_id),
                article_id=str(article.id),
                event_type=PUBLICATION_WEBHOOK_EVENT_TYPE,
                scheduled_for=scheduled_for,
                status="pending",
                attempt_count=0,
                next_attempt_at=scheduled_for,
                last_attempt_at=None,
                delivered_at=None,
                last_http_status=None,
                last_error=None,
            ),
        )

    if delivery.status == "delivered":
        return delivery

    delivery.patch(
        session,
        PublicationWebhookDeliveryPatchDTO.from_partial(reset_payload),
    )
    return delivery


async def backfill_project_publication_webhook_deliveries(
    session: AsyncSession,
    *,
    project_id: str,
) -> int:
    """Ensure publication webhook deliveries exist for all project articles."""
    project = await Project.get(session, project_id)
    if project is None:
        return 0

    endpoint = (project.notification_webhook or "").strip()
    secret = (project.notification_webhook_secret or "").strip()
    if not endpoint or not secret:
        pending_result = await session.execute(
            select(PublicationWebhookDelivery).where(
                PublicationWebhookDelivery.project_id == project_id,
                PublicationWebhookDelivery.event_type == PUBLICATION_WEBHOOK_EVENT_TYPE,
                PublicationWebhookDelivery.status.in_(
                    PUBLICATION_WEBHOOK_PENDING_STATUSES
                ),
            )
        )
        now = _utc_now()
        for delivery in pending_result.scalars().all():
            delivery.patch(
                session,
                PublicationWebhookDeliveryPatchDTO.from_partial(
                    {
                        "status": "canceled",
                        "next_attempt_at": now,
                        "last_error": "project_webhook_not_configured",
                    }
                ),
            )
        return 0

    result = await session.execute(
        select(ContentArticle, ContentBrief.proposed_publication_date)
        .join(ContentBrief, ContentBrief.id == ContentArticle.brief_id)
        .where(ContentArticle.project_id == project_id)
    )
    rows = result.all()
    scheduled_count = 0
    for article, proposed_publication_date in rows:
        await schedule_publication_webhook_for_article(
            session,
            article=article,
            proposed_publication_date=proposed_publication_date,
        )
        if proposed_publication_date is not None:
            scheduled_count += 1
    return scheduled_count


async def reschedule_publication_webhook_for_brief(
    session: AsyncSession,
    *,
    project_id: str,
    brief_id: str,
    proposed_publication_date: date | None,
) -> None:
    """Reschedule publication webhook when brief publication date changes."""
    result = await session.execute(
        select(ContentArticle).where(
            ContentArticle.project_id == project_id,
            ContentArticle.brief_id == brief_id,
        )
    )
    article = result.scalar_one_or_none()
    if article is None:
        return
    await schedule_publication_webhook_for_article(
        session,
        article=article,
        proposed_publication_date=proposed_publication_date,
    )


async def cancel_pending_publication_webhook_deliveries(
    session: AsyncSession,
    *,
    article_id: str,
) -> int:
    """Cancel non-terminal publication webhook deliveries for an article."""
    result = await session.execute(
        select(PublicationWebhookDelivery).where(
            PublicationWebhookDelivery.article_id == article_id,
            PublicationWebhookDelivery.event_type == PUBLICATION_WEBHOOK_EVENT_TYPE,
            PublicationWebhookDelivery.status.in_(PUBLICATION_WEBHOOK_PENDING_STATUSES),
        )
    )
    deliveries = list(result.scalars().all())
    if not deliveries:
        return 0

    now = _utc_now()
    for delivery in deliveries:
        delivery.patch(
            session,
            PublicationWebhookDeliveryPatchDTO.from_partial(
                {
                    "status": "canceled",
                    "next_attempt_at": now,
                    "last_error": "article_marked_published",
                }
            ),
        )
    return len(deliveries)


async def claim_due_publication_webhook_delivery_ids(
    session: AsyncSession,
    *,
    now: datetime,
    batch_size: int,
) -> list[str]:
    """Claim a small batch of due deliveries with row-level locking."""
    due_result = await session.execute(
        select(PublicationWebhookDelivery)
        .join(Project, Project.id == PublicationWebhookDelivery.project_id)
        .where(
            PublicationWebhookDelivery.event_type == PUBLICATION_WEBHOOK_EVENT_TYPE,
            PublicationWebhookDelivery.status.in_(PUBLICATION_WEBHOOK_PENDING_STATUSES),
            PublicationWebhookDelivery.next_attempt_at <= now,
            Project.notification_webhook.isnot(None),
            Project.notification_webhook_secret.isnot(None),
            Project.notification_webhook != "",
            Project.notification_webhook_secret != "",
        )
        .order_by(
            PublicationWebhookDelivery.next_attempt_at.asc(),
            PublicationWebhookDelivery.created_at.asc(),
        )
        .limit(max(1, int(batch_size)))
        .with_for_update(skip_locked=True)
    )
    deliveries = list(due_result.scalars().all())
    if not deliveries:
        return []

    lease_until = now + timedelta(seconds=PUBLICATION_WEBHOOK_CLAIM_LEASE_SECONDS)
    for delivery in deliveries:
        delivery.patch(
            session,
            PublicationWebhookDeliveryPatchDTO.from_partial(
                {
                    "next_attempt_at": lease_until,
                }
            ),
        )
    await session.flush()
    return [str(delivery.id) for delivery in deliveries]


async def _load_delivery_context(
    session: AsyncSession,
    *,
    delivery_id: str,
) -> tuple[
    PublicationWebhookDelivery,
    Project,
    ContentArticle,
    ContentBrief,
    ContentArticleVersion,
] | None:
    delivery = await PublicationWebhookDelivery.get(session, delivery_id)
    if delivery is None:
        return None

    project = await Project.get(session, str(delivery.project_id))
    article = await ContentArticle.get(session, str(delivery.article_id))
    if project is None or article is None:
        return None

    brief = await ContentBrief.get(session, str(article.brief_id))
    if brief is None:
        return None

    version_result = await session.execute(
        select(ContentArticleVersion).where(
            ContentArticleVersion.article_id == article.id,
            ContentArticleVersion.version_number == article.current_version,
        )
    )
    article_version = version_result.scalar_one_or_none()
    if article_version is None:
        return None

    return delivery, project, article, brief, article_version


async def dispatch_publication_webhook_delivery(
    session: AsyncSession,
    *,
    delivery_id: str,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Send one publication webhook delivery attempt and persist state transition."""
    context = await _load_delivery_context(session, delivery_id=delivery_id)
    if context is None:
        delivery = await PublicationWebhookDelivery.get(session, delivery_id)
        if delivery is not None and delivery.status not in PUBLICATION_WEBHOOK_TERMINAL_STATUSES:
            attempted_at = _utc_now()
            patch_payload = apply_publication_delivery_attempt_result(
                delivery=delivery,
                attempted_at=attempted_at,
                success=False,
                error_message="delivery_context_missing",
            )
            delivery.patch(
                session,
                PublicationWebhookDeliveryPatchDTO.from_partial(patch_payload),
            )
            await session.flush()
        return False

    delivery, project, article, brief, article_version = context
    if delivery.status in PUBLICATION_WEBHOOK_TERMINAL_STATUSES:
        return False

    attempted_at = _utc_now()
    endpoint = (project.notification_webhook or "").strip()
    secret = (project.notification_webhook_secret or "").strip()

    success = False
    http_status: int | None = None
    error_message: str | None = None

    if not endpoint or not secret:
        error_message = "project_webhook_not_configured"
    else:
        payload = build_publication_webhook_payload(
            delivery=delivery,
            project=project,
            article=article,
            article_version=article_version,
            brief=brief,
            occurred_at=attempted_at,
        )
        raw_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        timestamp = str(int(attempted_at.timestamp()))
        signature = sign_publication_webhook_payload(
            secret=secret,
            timestamp=timestamp,
            raw_body=raw_body,
        )
        headers = {
            "Content-Type": "application/json",
            "X-Donkey-Event": PUBLICATION_WEBHOOK_EVENT_TYPE,
            "X-Donkey-Delivery-Id": str(delivery.id),
            "X-Donkey-Timestamp": timestamp,
            "X-Donkey-Signature": signature,
        }

        own_client = http_client is None
        client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        try:
            response = await client.post(endpoint, content=raw_body, headers=headers)
            http_status = response.status_code
            success = 200 <= response.status_code < 300
            if not success:
                error_message = (
                    f"webhook_http_{response.status_code}: "
                    f"{response.text[:400]}"
                )
        except httpx.HTTPError as exc:
            error_message = str(exc)
        finally:
            if own_client:
                await client.aclose()

    patch_payload = apply_publication_delivery_attempt_result(
        delivery=delivery,
        attempted_at=attempted_at,
        success=success,
        http_status=http_status,
        error_message=error_message,
    )
    delivery.patch(
        session,
        PublicationWebhookDeliveryPatchDTO.from_partial(patch_payload),
    )
    await session.flush()
    return success


async def process_due_publication_webhook_deliveries(*, batch_size: int = 20) -> int:
    """Claim and process one batch of due publication webhook deliveries."""
    now = _utc_now()
    async with get_session_context() as session:
        delivery_ids = await claim_due_publication_webhook_delivery_ids(
            session,
            now=now,
            batch_size=batch_size,
        )

    if not delivery_ids:
        return 0

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http_client:
        for delivery_id in delivery_ids:
            async with get_session_context() as session:
                try:
                    await dispatch_publication_webhook_delivery(
                        session,
                        delivery_id=delivery_id,
                        http_client=http_client,
                    )
                except Exception:
                    logger.exception(
                        "Publication webhook delivery dispatch failed",
                        extra={"delivery_id": delivery_id},
                    )
    return len(delivery_ids)
