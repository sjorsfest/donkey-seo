"""Unit tests for publication webhook service helpers."""

from __future__ import annotations

import hashlib
import hmac
from datetime import date, datetime, timezone
from types import SimpleNamespace

from app.services.publication_webhook import (
    PUBLICATION_WEBHOOK_EVENT_TYPE,
    apply_publication_delivery_attempt_result,
    build_publication_webhook_payload,
    calculate_publication_retry_delay_seconds,
    scheduled_publication_datetime,
    sign_publication_webhook_payload,
)


def test_scheduled_publication_datetime_defaults_to_9am_utc() -> None:
    scheduled = scheduled_publication_datetime(date(2026, 2, 24))

    assert scheduled.tzinfo == timezone.utc
    assert scheduled.hour == 9
    assert scheduled.minute == 0


def test_sign_publication_webhook_payload_matches_expected_hmac() -> None:
    raw_body = b'{"hello":"world"}'
    timestamp = "1700000000"
    secret = "top-secret"

    signature = sign_publication_webhook_payload(
        secret=secret,
        timestamp=timestamp,
        raw_body=raw_body,
    )
    expected_digest = hmac.new(
        secret.encode("utf-8"),
        msg=f"{timestamp}.{raw_body.decode('utf-8')}".encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    assert signature == f"sha256={expected_digest}"


def test_build_publication_webhook_payload_includes_latest_version_document() -> None:
    occurred_at = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)
    delivery = SimpleNamespace(id="delivery_1")
    project = SimpleNamespace(id="project_1", domain="example.com", primary_locale="en-US")
    article = SimpleNamespace(id="article_1", brief_id="brief_1")
    article_version = SimpleNamespace(
        version_number=4,
        title="How to Publish",
        slug="how-to-publish",
        primary_keyword="publish article webhook",
        modular_document={"schema_version": "1.0", "blocks": [{"block_type": "hero"}]},
        rendered_html="<article>latest</article>",
    )
    brief = SimpleNamespace(proposed_publication_date=date(2026, 2, 25))

    payload = build_publication_webhook_payload(
        delivery=delivery,
        project=project,
        article=article,
        article_version=article_version,
        brief=brief,
        occurred_at=occurred_at,
    )

    assert payload["event_type"] == PUBLICATION_WEBHOOK_EVENT_TYPE
    assert payload["event_id"] == "delivery_1"
    assert payload["article"]["version_number"] == 4
    assert payload["article"]["proposed_publication_date"] == "2026-02-25"
    assert payload["modular_document"]["schema_version"] == "1.0"


def test_retry_backoff_grows_exponentially() -> None:
    assert calculate_publication_retry_delay_seconds(1) == 60
    assert calculate_publication_retry_delay_seconds(2) == 120
    assert calculate_publication_retry_delay_seconds(3) == 240


def test_apply_delivery_attempt_result_success_marks_delivered() -> None:
    attempted_at = datetime(2026, 2, 24, 12, 0, tzinfo=timezone.utc)
    delivery = SimpleNamespace(attempt_count=0)

    payload = apply_publication_delivery_attempt_result(
        delivery=delivery,
        attempted_at=attempted_at,
        success=True,
        http_status=204,
        error_message=None,
    )

    assert payload["status"] == "delivered"
    assert payload["attempt_count"] == 1
    assert payload["delivered_at"] == attempted_at
    assert payload["last_http_status"] == 204
    assert payload["last_error"] is None


def test_apply_delivery_attempt_result_failure_retries_then_fails() -> None:
    attempted_at = datetime(2026, 2, 24, 12, 0, tzinfo=timezone.utc)
    retrying = apply_publication_delivery_attempt_result(
        delivery=SimpleNamespace(attempt_count=1),
        attempted_at=attempted_at,
        success=False,
        http_status=500,
        error_message="webhook_http_500",
    )
    failed = apply_publication_delivery_attempt_result(
        delivery=SimpleNamespace(attempt_count=4),
        attempted_at=attempted_at,
        success=False,
        http_status=500,
        error_message="webhook_http_500",
    )

    assert retrying["status"] == "retrying"
    assert retrying["attempt_count"] == 2
    assert retrying["next_attempt_at"] > attempted_at
    assert failed["status"] == "failed"
    assert failed["attempt_count"] == 5
