"""Billing API schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl

from app.services.billing import PlanInterval, PlanKey


class PlanPriceOptionResponse(BaseModel):
    """Single plan interval option resolved from Stripe."""

    plan: PlanKey
    interval: PlanInterval
    price_id: str
    amount_cents: int
    currency: str
    product_name: str | None
    nickname: str | None


class BillingPlansResponse(BaseModel):
    """All plan options for dashboard pricing cards."""

    publishable_key: str | None
    trial_days: int
    plans: list[PlanPriceOptionResponse]


class BillingStatusResponse(BaseModel):
    """Current user Stripe billing state."""

    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    stripe_price_id: str | None
    subscription_plan: PlanKey | None
    subscription_interval: PlanInterval | None
    subscription_status: str | None
    subscription_current_period_end: datetime | None
    subscription_trial_ends_at: datetime | None


class BillingUsageResponse(BaseModel):
    """Current usage counters for dashboard progress bars."""

    plan: PlanKey | None
    window_kind: Literal["monthly", "lifetime"]
    period_start: datetime | None
    period_end: datetime | None
    article_limit: int
    used_articles: int
    remaining_articles: int
    usage_percent: float


class CheckoutSessionRequest(BaseModel):
    """Create Stripe checkout session request payload."""

    plan: Literal["starter", "growth", "agency"]
    interval: Literal["monthly", "yearly"]
    success_url: HttpUrl
    cancel_url: HttpUrl


class CheckoutSessionResponse(BaseModel):
    """Stripe checkout session response payload."""

    id: str
    url: str


class BillingPortalRequest(BaseModel):
    """Create billing portal session request."""

    return_url: HttpUrl


class BillingPortalResponse(BaseModel):
    """Stripe billing portal session response payload."""

    url: str


class StripeWebhookResponse(BaseModel):
    """Webhook ack response."""

    received: bool
    event_type: str | None = None
