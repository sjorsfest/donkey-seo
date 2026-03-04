"""Billing API endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import cast

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select

from app.api.v1.billing.constants import (
    INVALID_STRIPE_SIGNATURE_DETAIL,
    STRIPE_CUSTOMER_NOT_FOUND_DETAIL,
    STRIPE_NOT_CONFIGURED_DETAIL,
    STRIPE_PRICE_NOT_CONFIGURED_DETAIL,
)
from app.config import settings
from app.dependencies import CurrentUser, DbSession
from app.integrations.stripe_billing import StripeBillingClient, StripeSignatureError
from app.models.content import ContentArticle
from app.models.pipeline import PipelineRun
from app.models.project import Project
from app.models.user import User
from app.schemas.billing import (
    BillingPlansResponse,
    BillingPortalRequest,
    BillingPortalResponse,
    BillingStatusResponse,
    BillingUsageResponse,
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    PlanPriceOptionResponse,
    StripeWebhookResponse,
)
from app.services.billing import (
    PlanInterval,
    PlanKey,
    apply_subscription_payload,
    configured_plan_prices,
    normalize_plan,
    resolve_article_limit,
    resolve_price_id,
    resolve_usage_window,
)
from app.services.pipeline_task_manager import (
    PipelineQueueFullError,
    get_discovery_pipeline_task_manager,
)
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

router = APIRouter()


def _assert_stripe_enabled() -> None:
    if not settings.stripe_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STRIPE_NOT_CONFIGURED_DETAIL,
        )


def _normalize_plan(value: str | None) -> PlanKey | None:
    return normalize_plan(value)


def _normalize_interval(value: str | None) -> PlanInterval | None:
    if value in {"monthly", "yearly"}:
        return cast(PlanInterval, value)
    return None


def _to_billing_status(user: User) -> BillingStatusResponse:
    return BillingStatusResponse(
        stripe_customer_id=user.stripe_customer_id,
        stripe_subscription_id=user.stripe_subscription_id,
        stripe_price_id=user.stripe_price_id,
        subscription_plan=_normalize_plan(user.subscription_plan),
        subscription_interval=_normalize_interval(user.subscription_interval),
        subscription_status=user.subscription_status,
        subscription_current_period_end=user.subscription_current_period_end,
        subscription_trial_ends_at=user.subscription_trial_ends_at,
    )


async def _create_customer_for_user(user: User) -> str:
    async with StripeBillingClient() as stripe:
        customer = await stripe.create_customer(
            email=user.email,
            full_name=user.full_name,
            metadata={"app_user_id": str(user.id)},
        )
    customer_id = customer.get("id")
    if not isinstance(customer_id, str) or not customer_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Stripe customer creation failed",
        )
    return customer_id


async def _get_user_by_customer_id(
    *,
    session: DbSession,
    stripe_customer_id: str | None,
) -> User | None:
    if not stripe_customer_id:
        return None
    result = await session.execute(
        select(User).where(User.stripe_customer_id == stripe_customer_id)
    )
    return result.scalar_one_or_none()


def _is_free_to_paid_upgrade(
    *,
    previous_plan: PlanKey | None,
    current_plan: PlanKey | None,
) -> bool:
    return previous_plan is None and current_plan is not None


async def _enqueue_resume_for_halted_discovery_runs(
    *,
    session: DbSession,
    user: User,
) -> int:
    paused_result = await session.execute(
        select(
            PipelineRun.id,
            PipelineRun.project_id,
            PipelineRun.paused_at_step,
            PipelineRun.source_topic_id,
        )
        .join(Project, PipelineRun.project_id == Project.id)
        .where(
            Project.user_id == user.id,
            PipelineRun.pipeline_module == "discovery",
            PipelineRun.status == "paused",
            PipelineRun.error_message.is_not(None),
        )
        .order_by(PipelineRun.project_id.asc(), PipelineRun.created_at.desc())
    )
    paused_rows = list(paused_result.all())
    latest_by_project: dict[str, tuple[str, int | None, str | None]] = {}
    for run_id, project_id, paused_at_step, source_topic_id in paused_rows:
        project_id_str = str(project_id)
        if project_id_str in latest_by_project:
            continue
        latest_by_project[project_id_str] = (
            str(run_id),
            int(paused_at_step) if paused_at_step is not None else None,
            str(source_topic_id) if source_topic_id is not None else None,
        )

    if not latest_by_project:
        return 0

    project_ids = list(latest_by_project.keys())
    running_result = await session.execute(
        select(PipelineRun.project_id).where(
            PipelineRun.project_id.in_(project_ids),
            PipelineRun.pipeline_module == "discovery",
            PipelineRun.status == "running",
        )
    )
    running_project_ids = {
        str(project_id)
        for project_id in running_result.scalars().all()
        if project_id is not None
    }

    queue = get_discovery_pipeline_task_manager()
    task_manager = TaskManager()
    resumed_count = 0
    for project_id, (run_id, paused_at_step, source_topic_id) in latest_by_project.items():
        if project_id in running_project_ids:
            continue

        await task_manager.set_task_state(
            task_id=run_id,
            status="queued",
            stage="Queued pipeline resume",
            project_id=project_id,
            pipeline_module="discovery",
            source_topic_id=source_topic_id,
            current_step=paused_at_step,
            current_step_name=None,
            error_message=None,
        )
        try:
            await queue.enqueue_resume(
                project_id=project_id,
                run_id=run_id,
            )
            resumed_count += 1
        except PipelineQueueFullError:
            await task_manager.set_task_state(
                task_id=run_id,
                status="paused",
                stage="Pipeline queue is full",
                error_message="Pipeline queue is full, try again shortly",
            )
            logger.warning(
                "Skipping discovery resume after free-to-paid upgrade because queue is full",
                extra={
                    "user_id": str(user.id),
                    "project_id": project_id,
                    "run_id": run_id,
                },
            )
    return resumed_count


async def _sync_subscription_for_user(
    *,
    session: DbSession,
    user: User,
    subscription: dict[str, object],
) -> None:
    previous_plan = _normalize_plan(user.subscription_plan)
    apply_subscription_payload(user=user, subscription=subscription)
    current_plan = _normalize_plan(user.subscription_plan)
    await session.flush()

    if not _is_free_to_paid_upgrade(
        previous_plan=previous_plan,
        current_plan=current_plan,
    ):
        return

    await session.commit()
    try:
        resumed_count = await _enqueue_resume_for_halted_discovery_runs(
            session=session,
            user=user,
        )
    except Exception:
        logger.exception(
            "Failed to enqueue discovery resume after free-to-paid upgrade",
            extra={"user_id": str(user.id)},
        )
        return
    logger.info(
        "Processed free-to-paid discovery resume",
        extra={
            "user_id": str(user.id),
            "from_plan": previous_plan,
            "to_plan": current_plan,
            "resumed_discovery_runs": resumed_count,
        },
    )


@router.get(
    "/plans",
    response_model=BillingPlansResponse,
    summary="List billing plans",
    description="Return configured Stripe-backed pricing options for dashboard plan selection.",
)
async def list_billing_plans() -> BillingPlansResponse:
    _assert_stripe_enabled()
    configured = configured_plan_prices()
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No Stripe prices configured",
        )

    async with StripeBillingClient() as stripe:
        responses = await asyncio.gather(
            *[stripe.retrieve_price(item.price_id) for item in configured],
            return_exceptions=True,
        )

    options: list[PlanPriceOptionResponse] = []
    for item, response in zip(configured, responses, strict=True):
        if isinstance(response, Exception):
            logger.warning(
                "Failed to load Stripe price",
                extra={"price_id": item.price_id, "error": str(response)},
            )
            continue

        product_name: str | None = None
        product = response.get("product")
        if isinstance(product, dict):
            raw_name = product.get("name")
            product_name = raw_name if isinstance(raw_name, str) else None

        amount_cents = response.get("unit_amount")
        currency = response.get("currency")
        if not isinstance(amount_cents, int) or not isinstance(currency, str):
            continue

        nickname = response.get("nickname")
        options.append(
            PlanPriceOptionResponse(
                plan=item.plan,
                interval=item.interval,
                price_id=item.price_id,
                amount_cents=amount_cents,
                currency=currency,
                product_name=product_name,
                nickname=nickname if isinstance(nickname, str) else None,
            )
        )

    plan_rank = {"starter": 0, "growth": 1, "agency": 2}
    interval_rank = {"monthly": 0, "yearly": 1}
    options.sort(key=lambda p: (plan_rank[p.plan], interval_rank[p.interval]))

    return BillingPlansResponse(
        publishable_key=settings.stripe_publishable_key,
        trial_days=0,
        plans=options,
    )


@router.get(
    "/me",
    response_model=BillingStatusResponse,
    summary="Get current billing status",
)
async def get_my_billing_status(current_user: CurrentUser) -> BillingStatusResponse:
    return _to_billing_status(current_user)


@router.get(
    "/usage",
    response_model=BillingUsageResponse,
    summary="Get current usage",
    description=(
        "Return article usage counters for the current account. "
        "Paid plans use monthly windows; free users use lifetime window."
    ),
)
async def get_my_usage(
    current_user: CurrentUser,
    session: DbSession,
) -> BillingUsageResponse:
    plan = _normalize_plan(current_user.subscription_plan)
    usage_window = resolve_usage_window(plan)
    article_limit = resolve_article_limit(plan)

    usage_query = (
        select(func.count())
        .select_from(ContentArticle)
        .join(Project, ContentArticle.project_id == Project.id)
        .where(Project.user_id == current_user.id)
    )
    if usage_window.period_start is not None:
        usage_query = usage_query.where(ContentArticle.generated_at >= usage_window.period_start)
    if usage_window.period_end is not None:
        usage_query = usage_query.where(ContentArticle.generated_at < usage_window.period_end)

    used_articles = int(await session.scalar(usage_query) or 0)
    remaining_articles = max(article_limit - used_articles, 0)
    usage_percent = (
        min((used_articles / article_limit) * 100, 100.0)
        if article_limit > 0
        else 0.0
    )

    return BillingUsageResponse(
        plan=plan,
        window_kind=usage_window.kind,
        period_start=usage_window.period_start,
        period_end=usage_window.period_end,
        article_limit=article_limit,
        used_articles=used_articles,
        remaining_articles=remaining_articles,
        usage_percent=round(usage_percent, 2),
    )


@router.post(
    "/checkout-session",
    response_model=CheckoutSessionResponse,
    summary="Create Stripe checkout session",
)
async def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> CheckoutSessionResponse:
    _assert_stripe_enabled()

    price_id = resolve_price_id(plan=request.plan, interval=request.interval)
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=STRIPE_PRICE_NOT_CONFIGURED_DETAIL,
        )

    if not current_user.stripe_customer_id:
        current_user.stripe_customer_id = await _create_customer_for_user(current_user)
        await session.flush()
    customer_id = current_user.stripe_customer_id
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Stripe customer provisioning failed",
        )

    async with StripeBillingClient() as stripe:
        checkout = await stripe.create_checkout_session(
            customer_id=customer_id,
            price_id=price_id,
            success_url=str(request.success_url),
            cancel_url=str(request.cancel_url),
        )

    checkout_id = checkout.get("id")
    checkout_url = checkout.get("url")
    if not isinstance(checkout_id, str) or not isinstance(checkout_url, str):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Stripe checkout session creation failed",
        )
    return CheckoutSessionResponse(id=checkout_id, url=checkout_url)


@router.post(
    "/portal-session",
    response_model=BillingPortalResponse,
    summary="Create Stripe customer portal session",
)
async def create_billing_portal_session(
    request: BillingPortalRequest,
    current_user: CurrentUser,
) -> BillingPortalResponse:
    _assert_stripe_enabled()
    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=STRIPE_CUSTOMER_NOT_FOUND_DETAIL,
        )

    async with StripeBillingClient() as stripe:
        portal = await stripe.create_billing_portal_session(
            customer_id=current_user.stripe_customer_id,
            return_url=str(request.return_url),
        )

    url = portal.get("url")
    if not isinstance(url, str) or not url:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Stripe billing portal session creation failed",
        )
    return BillingPortalResponse(url=url)


@router.post("/webhooks/stripe/", include_in_schema=False)
@router.post(
    "/webhooks/stripe",
    response_model=StripeWebhookResponse,
    summary="Stripe webhook receiver",
)
async def stripe_webhook(
    request: Request,
    session: DbSession,
) -> StripeWebhookResponse:
    _assert_stripe_enabled()

    signature_header = request.headers.get("Stripe-Signature")
    if not signature_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=INVALID_STRIPE_SIGNATURE_DETAIL,
        )

    payload = await request.body()
    async with StripeBillingClient() as stripe:
        try:
            event = stripe.verify_webhook_event(
                payload=payload,
                signature_header=signature_header,
            )
        except StripeSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=INVALID_STRIPE_SIGNATURE_DETAIL,
            ) from exc

        event_type = event.get("type")
        if not isinstance(event_type, str):
            return StripeWebhookResponse(received=True)

        data = event.get("data")
        event_object = (data or {}).get("object") if isinstance(data, dict) else None
        if not isinstance(event_object, dict):
            return StripeWebhookResponse(received=True, event_type=event_type)

        if event_type in {
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
        }:
            customer_id = event_object.get("customer")
            user = await _get_user_by_customer_id(
                session=session,
                stripe_customer_id=customer_id if isinstance(customer_id, str) else None,
            )
            if user:
                await _sync_subscription_for_user(
                    session=session,
                    user=user,
                    subscription=event_object,
                )

        elif event_type == "checkout.session.completed":
            mode = event_object.get("mode")
            customer_id = event_object.get("customer")
            subscription_id = event_object.get("subscription")
            if mode == "subscription" and isinstance(customer_id, str):
                user = await _get_user_by_customer_id(
                    session=session,
                    stripe_customer_id=customer_id,
                )
                if user and isinstance(subscription_id, str):
                    subscription = await stripe.retrieve_subscription(subscription_id)
                    await _sync_subscription_for_user(
                        session=session,
                        user=user,
                        subscription=subscription,
                    )

        elif event_type in {"invoice.paid", "invoice.payment_failed"}:
            customer_id = event_object.get("customer")
            subscription_id = event_object.get("subscription")
            if isinstance(customer_id, str) and isinstance(subscription_id, str):
                user = await _get_user_by_customer_id(
                    session=session,
                    stripe_customer_id=customer_id,
                )
                if user:
                    subscription = await stripe.retrieve_subscription(subscription_id)
                    await _sync_subscription_for_user(
                        session=session,
                        user=user,
                        subscription=subscription,
                    )

        return StripeWebhookResponse(received=True, event_type=event_type)
