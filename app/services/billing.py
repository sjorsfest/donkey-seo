"""Billing helper utilities for Stripe plan mapping and subscription sync."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

from app.config import settings
from app.integrations.stripe_billing import stripe_unix_to_datetime

if TYPE_CHECKING:
    from app.models.user import User

PlanKey = Literal["starter", "growth", "agency"]
PlanInterval = Literal["monthly", "yearly"]


@dataclass(frozen=True)
class BillingPlanPrice:
    """Configured Stripe price descriptor for one plan interval."""

    plan: PlanKey
    interval: PlanInterval
    price_id: str


MONTHLY_ARTICLE_LIMITS: dict[PlanKey, int] = {
    "starter": 30,
    "growth": 100,
    "agency": 350,
}
PROJECT_LIMITS: dict[PlanKey | None, int] = {
    None: 1,
    "starter": 1,
    "growth": 3,
    "agency": 10,
}
FREE_LIFETIME_ARTICLE_LIMIT = 3


@dataclass(frozen=True)
class UsageWindow:
    """Usage accounting window metadata."""

    kind: Literal["monthly", "lifetime"]
    period_start: datetime | None
    period_end: datetime | None


def configured_plan_prices() -> list[BillingPlanPrice]:
    """Return configured plan prices from settings."""
    prices: list[BillingPlanPrice] = []
    if settings.stripe_price_starter_monthly:
        prices.append(
            BillingPlanPrice(
                plan="starter",
                interval="monthly",
                price_id=settings.stripe_price_starter_monthly,
            )
        )
    if settings.stripe_price_starter_yearly:
        prices.append(
            BillingPlanPrice(
                plan="starter",
                interval="yearly",
                price_id=settings.stripe_price_starter_yearly,
            )
        )
    if settings.stripe_price_growth_monthly:
        prices.append(
            BillingPlanPrice(
                plan="growth",
                interval="monthly",
                price_id=settings.stripe_price_growth_monthly,
            )
        )
    if settings.stripe_price_growth_yearly:
        prices.append(
            BillingPlanPrice(
                plan="growth",
                interval="yearly",
                price_id=settings.stripe_price_growth_yearly,
            )
        )
    if settings.stripe_price_agency_monthly:
        prices.append(
            BillingPlanPrice(
                plan="agency",
                interval="monthly",
                price_id=settings.stripe_price_agency_monthly,
            )
        )
    if settings.stripe_price_agency_yearly:
        prices.append(
            BillingPlanPrice(
                plan="agency",
                interval="yearly",
                price_id=settings.stripe_price_agency_yearly,
            )
        )
    return prices


def resolve_plan_from_price_id(price_id: str | None) -> tuple[PlanKey | None, PlanInterval | None]:
    """Resolve app plan + interval from configured Stripe price id."""
    if not price_id:
        return None, None
    for item in configured_plan_prices():
        if item.price_id == price_id:
            return item.plan, item.interval
    return None, None


def resolve_price_id(*, plan: PlanKey, interval: PlanInterval) -> str | None:
    """Resolve configured Stripe price id for a plan interval."""
    for item in configured_plan_prices():
        if item.plan == plan and item.interval == interval:
            return item.price_id
    return None


def extract_subscription_price_id(subscription: dict[str, Any]) -> str | None:
    """Extract the first subscription item price id."""
    items = subscription.get("items")
    if not isinstance(items, dict):
        return None
    data = items.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    price = first.get("price")
    if not isinstance(price, dict):
        return None
    value = price.get("id")
    return str(value) if isinstance(value, str) else None


def apply_subscription_payload(*, user: User, subscription: dict[str, Any]) -> None:
    """Update user billing fields from Stripe subscription payload."""
    price_id = extract_subscription_price_id(subscription)
    plan, interval = resolve_plan_from_price_id(price_id)

    raw_subscription_id = subscription.get("id")
    if isinstance(raw_subscription_id, str) and raw_subscription_id:
        user.stripe_subscription_id = raw_subscription_id
    user.stripe_price_id = price_id
    user.subscription_plan = plan
    user.subscription_interval = interval

    raw_status = subscription.get("status")
    if isinstance(raw_status, str) and raw_status:
        user.subscription_status = raw_status

    user.subscription_current_period_end = stripe_unix_to_datetime(
        subscription.get("current_period_end")
    )
    user.subscription_trial_ends_at = stripe_unix_to_datetime(subscription.get("trial_end"))


def normalize_plan(value: str | None) -> PlanKey | None:
    """Normalize raw subscription plan to known internal plan key."""
    if value in {"starter", "growth", "agency"}:
        return cast(PlanKey, value)
    return None


def resolve_article_limit(plan: PlanKey | None) -> int:
    """Resolve article limit for current subscription plan."""
    if plan is None:
        return FREE_LIFETIME_ARTICLE_LIMIT
    return MONTHLY_ARTICLE_LIMITS.get(plan, FREE_LIFETIME_ARTICLE_LIMIT)


def resolve_project_limit(plan: PlanKey | None) -> int:
    """Resolve project limit for current subscription plan."""
    return PROJECT_LIMITS.get(plan, PROJECT_LIMITS[None])


def resolve_usage_window(plan: PlanKey | None, *, now: datetime | None = None) -> UsageWindow:
    """Resolve usage window for usage accounting."""
    if plan is None:
        return UsageWindow(kind="lifetime", period_start=None, period_end=None)

    now_utc = now or datetime.now(timezone.utc)
    start = datetime(
        year=now_utc.year,
        month=now_utc.month,
        day=1,
        tzinfo=timezone.utc,
    )
    if start.month == 12:
        end = datetime(
            year=start.year + 1,
            month=1,
            day=1,
            tzinfo=timezone.utc,
        )
    else:
        end = datetime(
            year=start.year,
            month=start.month + 1,
            day=1,
            tzinfo=timezone.utc,
        )
    return UsageWindow(kind="monthly", period_start=start, period_end=end)
