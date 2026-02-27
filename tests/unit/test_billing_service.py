"""Unit tests for billing utilities and Stripe webhook verification."""

from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timezone

import pytest

from app.integrations.stripe_billing import StripeBillingClient, StripeSignatureError
from app.services.billing import (
    FREE_LIFETIME_ARTICLE_LIMIT,
    MONTHLY_ARTICLE_LIMITS,
    PROJECT_LIMITS,
    apply_subscription_payload,
    normalize_plan,
    resolve_article_limit,
    resolve_project_limit,
    resolve_plan_from_price_id,
    resolve_price_id,
    resolve_usage_window,
)


class _DummyUser:
    """Simple mutable object for testing subscription sync helper."""

    stripe_subscription_id: str | None = None
    stripe_price_id: str | None = None
    subscription_plan: str | None = None
    subscription_interval: str | None = None
    subscription_status: str | None = None
    subscription_current_period_end: datetime | None = None
    subscription_trial_ends_at: datetime | None = None


def test_resolve_price_id_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.billing.settings.stripe_price_starter_monthly",
        "price_starter_month",
    )
    monkeypatch.setattr(
        "app.services.billing.settings.stripe_price_starter_yearly",
        "price_starter_year",
    )
    monkeypatch.setattr(
        "app.services.billing.settings.stripe_price_growth_monthly",
        "price_growth_month",
    )
    monkeypatch.setattr(
        "app.services.billing.settings.stripe_price_growth_yearly",
        "price_growth_year",
    )
    monkeypatch.setattr(
        "app.services.billing.settings.stripe_price_agency_monthly",
        "price_agency_month",
    )
    monkeypatch.setattr(
        "app.services.billing.settings.stripe_price_agency_yearly",
        "price_agency_year",
    )

    assert resolve_price_id(plan="starter", interval="monthly") == "price_starter_month"
    assert resolve_price_id(plan="agency", interval="yearly") == "price_agency_year"
    assert resolve_plan_from_price_id("price_growth_year") == ("growth", "yearly")


def test_apply_subscription_payload_sets_user_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.billing.settings.stripe_price_starter_monthly",
        "price_starter_month",
    )

    user = _DummyUser()
    now = int(time.time())
    subscription = {
        "id": "sub_123",
        "status": "trialing",
        "current_period_end": now + 3600,
        "trial_end": now + 7200,
        "items": {
            "data": [
                {
                    "price": {"id": "price_starter_month"},
                }
            ]
        },
    }

    apply_subscription_payload(user=user, subscription=subscription)

    assert user.stripe_subscription_id == "sub_123"
    assert user.stripe_price_id == "price_starter_month"
    assert user.subscription_plan == "starter"
    assert user.subscription_interval == "monthly"
    assert user.subscription_status == "trialing"
    assert user.subscription_current_period_end == datetime.fromtimestamp(
        now + 3600,
        tz=timezone.utc,
    )
    assert user.subscription_trial_ends_at == datetime.fromtimestamp(
        now + 7200,
        tz=timezone.utc,
    )


def test_verify_webhook_signature_success() -> None:
    payload = b'{"type":"customer.subscription.created"}'
    webhook_secret = "whsec_test_secret"
    timestamp = int(time.time())
    signed_payload = f"{timestamp}.{payload.decode('utf-8')}".encode("utf-8")
    signature = hmac.new(
        webhook_secret.encode("utf-8"),
        msg=signed_payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    header = f"t={timestamp},v1={signature}"

    client = StripeBillingClient(
        secret_key="sk_test_abc",
        webhook_secret=webhook_secret,
    )
    event = client.verify_webhook_event(
        payload=payload,
        signature_header=header,
    )

    assert event["type"] == "customer.subscription.created"


def test_verify_webhook_signature_failure() -> None:
    client = StripeBillingClient(
        secret_key="sk_test_abc",
        webhook_secret="whsec_test_secret",
    )
    payload = b'{"type":"customer.subscription.created"}'

    with pytest.raises(StripeSignatureError):
        client.verify_webhook_event(
            payload=payload,
            signature_header="t=123,v1=invalid",
        )


def test_resolve_usage_window_for_monthly_plan() -> None:
    now = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
    window = resolve_usage_window("starter", now=now)
    assert window.kind == "monthly"
    assert window.period_start == datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert window.period_end == datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_resolve_usage_window_for_free_plan() -> None:
    window = resolve_usage_window(None)
    assert window.kind == "lifetime"
    assert window.period_start is None
    assert window.period_end is None


def test_resolve_article_limit() -> None:
    assert resolve_article_limit("starter") == MONTHLY_ARTICLE_LIMITS["starter"]
    assert resolve_article_limit("growth") == MONTHLY_ARTICLE_LIMITS["growth"]
    assert resolve_article_limit("agency") == MONTHLY_ARTICLE_LIMITS["agency"]
    assert resolve_article_limit(None) == FREE_LIFETIME_ARTICLE_LIMIT


def test_normalize_plan_and_resolve_project_limit() -> None:
    assert normalize_plan("starter") == "starter"
    assert normalize_plan("growth") == "growth"
    assert normalize_plan("agency") == "agency"
    assert normalize_plan("enterprise") is None
    assert normalize_plan(None) is None

    assert resolve_project_limit("starter") == PROJECT_LIMITS["starter"]
    assert resolve_project_limit("growth") == PROJECT_LIMITS["growth"]
    assert resolve_project_limit("agency") == PROJECT_LIMITS["agency"]
    assert resolve_project_limit(None) == PROJECT_LIMITS[None]
