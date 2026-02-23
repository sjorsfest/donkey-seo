#!/usr/bin/env python3
"""Provision DonkeySEO Stripe product and prices.

Idempotent behavior:
- Reuses existing product by metadata.internal_code
- Reuses existing prices by Stripe lookup_key
- Creates missing resources

Usage:
  STRIPE_SECRET_KEY=sk_test_xxx python scripts/setup_stripe_catalog.py

Optional:
  python scripts/setup_stripe_catalog.py --version v2 --starter-cents 5900
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class PriceDefinition:
    slug: str
    env_var: str
    nickname: str
    unit_amount: int
    recurring_interval: str | None


class StripeAPIError(RuntimeError):
    """Raised when Stripe returns an API error."""


class StripeClient:
    """Minimal Stripe API client using existing project dependencies."""

    def __init__(self, *, secret_key: str, timeout_seconds: float = 30.0) -> None:
        self._http = httpx.Client(
            base_url="https://api.stripe.com/v1",
            headers={
                "Authorization": f"Bearer {secret_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._http.request(method, path, data=data)
        payload = response.json()
        if response.status_code >= 400:
            error = payload.get("error", {})
            message = error.get("message", "Unknown Stripe error")
            code = error.get("code")
            param = error.get("param")
            details = f"{message} (code={code}, param={param})"
            raise StripeAPIError(details)
        return payload

    def _paginate_list(self, path: str, data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        all_items: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = dict(data or {})
            params.setdefault("limit", 100)
            if cursor:
                params["starting_after"] = cursor
            page = self._request("GET", path, data=params)
            items = page.get("data", [])
            if not isinstance(items, list):
                return all_items
            all_items.extend(item for item in items if isinstance(item, dict))
            if not page.get("has_more") or not items:
                return all_items
            last_id = items[-1].get("id")
            if not isinstance(last_id, str) or not last_id:
                return all_items
            cursor = last_id

    def find_product_by_internal_code(self, internal_code: str) -> dict[str, Any] | None:
        products = self._paginate_list("/products")
        for product in products:
            metadata = product.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("internal_code") == internal_code:
                return product
        return None

    def create_product(
        self,
        *,
        name: str,
        description: str,
        internal_code: str,
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "description": description,
            "metadata[internal_code]": internal_code,
        }
        return self._request("POST", "/products", data=payload)

    def update_product(
        self,
        *,
        product_id: str,
        name: str,
        description: str,
        internal_code: str,
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "description": description,
            "metadata[internal_code]": internal_code,
            "active": "true",
        }
        return self._request("POST", f"/products/{product_id}", data=payload)

    def find_price_by_lookup_key(self, lookup_key: str) -> dict[str, Any] | None:
        prices = self._paginate_list("/prices", data={"lookup_keys[]": lookup_key})
        if not prices:
            return None
        active_prices = [price for price in prices if price.get("active") is True]
        return active_prices[0] if active_prices else prices[0]

    def create_price(
        self,
        *,
        product_id: str,
        lookup_key: str,
        currency: str,
        nickname: str,
        unit_amount: int,
        recurring_interval: str | None,
        internal_code: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "product": product_id,
            "currency": currency,
            "unit_amount": str(unit_amount),
            "lookup_key": lookup_key,
            "nickname": nickname,
            "metadata[internal_code]": internal_code,
            "active": "true",
        }
        if recurring_interval:
            payload["recurring[interval]"] = recurring_interval
        return self._request("POST", "/prices", data=payload)


def _build_price_definitions(args: argparse.Namespace) -> list[PriceDefinition]:
    return [
        PriceDefinition(
            slug="starter_monthly",
            env_var="STRIPE_PRICE_STARTER_MONTHLY",
            nickname="DonkeySEO Starter Monthly",
            unit_amount=args.starter_cents,
            recurring_interval="month",
        ),
        PriceDefinition(
            slug="starter_yearly",
            env_var="STRIPE_PRICE_STARTER_YEARLY",
            nickname="DonkeySEO Starter Yearly",
            unit_amount=args.starter_yearly_cents,
            recurring_interval="year",
        ),
        PriceDefinition(
            slug="growth_monthly",
            env_var="STRIPE_PRICE_GROWTH_MONTHLY",
            nickname="DonkeySEO Growth Monthly",
            unit_amount=args.growth_cents,
            recurring_interval="month",
        ),
        PriceDefinition(
            slug="growth_yearly",
            env_var="STRIPE_PRICE_GROWTH_YEARLY",
            nickname="DonkeySEO Growth Yearly",
            unit_amount=args.growth_yearly_cents,
            recurring_interval="year",
        ),
        PriceDefinition(
            slug="agency_monthly",
            env_var="STRIPE_PRICE_AGENCY_MONTHLY",
            nickname="DonkeySEO Agency Monthly",
            unit_amount=args.agency_cents,
            recurring_interval="month",
        ),
        PriceDefinition(
            slug="agency_yearly",
            env_var="STRIPE_PRICE_AGENCY_YEARLY",
            nickname="DonkeySEO Agency Yearly",
            unit_amount=args.agency_yearly_cents,
            recurring_interval="year",
        ),
        PriceDefinition(
            slug="article_addon",
            env_var="STRIPE_PRICE_ARTICLE_ADDON",
            nickname="DonkeySEO Article Add-on",
            unit_amount=args.article_addon_cents,
            recurring_interval=None,
        ),
    ]


def _validate_existing_price(
    *,
    price: dict[str, Any],
    product_id: str,
    currency: str,
    expected_unit_amount: int,
    recurring_interval: str | None,
    lookup_key: str,
) -> None:
    existing_product = price.get("product")
    if existing_product != product_id:
        raise StripeAPIError(
            f"Lookup key '{lookup_key}' belongs to product "
            f"'{existing_product}', expected '{product_id}'."
        )

    existing_currency = str(price.get("currency", "")).lower()
    if existing_currency != currency:
        raise StripeAPIError(
            f"Lookup key '{lookup_key}' has currency '{existing_currency}', expected '{currency}'."
        )

    existing_amount = price.get("unit_amount")
    if existing_amount != expected_unit_amount:
        raise StripeAPIError(
            f"Lookup key '{lookup_key}' has unit_amount={existing_amount}, "
            f"expected {expected_unit_amount}. Bump --version to create a new price."
        )

    recurring_payload = price.get("recurring")
    existing_interval = None
    if isinstance(recurring_payload, dict):
        interval_value = recurring_payload.get("interval")
        existing_interval = interval_value if isinstance(interval_value, str) else None

    if existing_interval != recurring_interval:
        raise StripeAPIError(
            f"Lookup key '{lookup_key}' recurring mismatch. "
            f"existing={existing_interval} expected={recurring_interval}. "
            "Bump --version."
        )


def _build_lookup_key(price_slug: str, version: str) -> str:
    return f"donkeyseo_{price_slug}_{version}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision Stripe product/prices for DonkeySEO subscriptions."
    )
    parser.add_argument(
        "--product-name",
        default="DonkeySEO",
        help="Stripe product display name.",
    )
    parser.add_argument(
        "--product-description",
        default="AI SEO pipeline with article generation subscriptions.",
        help="Stripe product description.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Price catalog version suffix used in lookup keys.",
    )
    parser.add_argument("--currency", default="usd", help="Price currency (default: usd).")
    parser.add_argument("--starter-cents", type=int, default=4900, help="Starter monthly price.")
    parser.add_argument(
        "--starter-yearly-cents",
        type=int,
        default=49000,
        help="Starter yearly price (recommended discounted).",
    )
    parser.add_argument("--growth-cents", type=int, default=14900, help="Growth monthly price.")
    parser.add_argument(
        "--growth-yearly-cents",
        type=int,
        default=149000,
        help="Growth yearly price (recommended discounted).",
    )
    parser.add_argument("--agency-cents", type=int, default=39900, help="Agency monthly price.")
    parser.add_argument(
        "--agency-yearly-cents",
        type=int,
        default=399000,
        help="Agency yearly price (recommended discounted).",
    )
    parser.add_argument(
        "--article-addon-cents",
        type=int,
        default=200,
        help="One-time add-on article credit price.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print intended lookup keys without creating resources.",
    )
    return parser.parse_args()


def _print_result(
    *,
    product_id: str,
    prices: dict[str, str],
    account_mode: str,
    version: str,
) -> None:
    print("")
    print("Stripe catalog ready")
    print(f"- mode: {account_mode}")
    print(f"- catalog version: {version}")
    print(f"- STRIPE_PRODUCT_DONKEYSEO={product_id}")
    for key, value in prices.items():
        print(f"- {key}={value}")
    print("")
    print("Paste into your environment:")
    print(f"STRIPE_PRODUCT_DONKEYSEO={product_id}")
    for key, value in prices.items():
        print(f"{key}={value}")


def main() -> int:
    args = _parse_args()
    currency = args.currency.lower().strip()
    if len(currency) != 3:
        print("Currency must be a 3-letter ISO code.", file=sys.stderr)
        return 1

    price_defs = _build_price_definitions(args)
    product_internal_code = "donkeyseo_product"

    if args.dry_run:
        print("Dry run mode. Intended lookup keys:")
        for price_def in price_defs:
            print(f"- {_build_lookup_key(price_def.slug, args.version)}")
        return 0

    secret_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not secret_key:
        print("Missing STRIPE_SECRET_KEY in environment.", file=sys.stderr)
        return 1

    if secret_key.startswith("sk_test_"):
        account_mode = "test"
    elif secret_key.startswith("sk_live_"):
        account_mode = "live"
    else:
        account_mode = "unknown"

    client = StripeClient(secret_key=secret_key, timeout_seconds=args.timeout_seconds)
    try:
        product = client.find_product_by_internal_code(product_internal_code)
        if product is None:
            product = client.create_product(
                name=args.product_name,
                description=args.product_description,
                internal_code=product_internal_code,
            )
        else:
            product = client.update_product(
                product_id=str(product["id"]),
                name=args.product_name,
                description=args.product_description,
                internal_code=product_internal_code,
            )

        product_id = str(product["id"])
        price_env_map: dict[str, str] = {}
        for price_def in price_defs:
            lookup_key = _build_lookup_key(price_def.slug, args.version)
            internal_code = f"donkeyseo_price_{price_def.slug}_{args.version}"
            existing_price = client.find_price_by_lookup_key(lookup_key)

            if existing_price is not None:
                _validate_existing_price(
                    price=existing_price,
                    product_id=product_id,
                    currency=currency,
                    expected_unit_amount=price_def.unit_amount,
                    recurring_interval=price_def.recurring_interval,
                    lookup_key=lookup_key,
                )
                price_id = str(existing_price["id"])
            else:
                new_price = client.create_price(
                    product_id=product_id,
                    lookup_key=lookup_key,
                    currency=currency,
                    nickname=price_def.nickname,
                    unit_amount=price_def.unit_amount,
                    recurring_interval=price_def.recurring_interval,
                    internal_code=internal_code,
                )
                price_id = str(new_price["id"])

            price_env_map[price_def.env_var] = price_id

        _print_result(
            product_id=product_id,
            prices=price_env_map,
            account_mode=account_mode,
            version=args.version,
        )
        return 0
    except (httpx.HTTPError, StripeAPIError) as exc:
        print(f"Stripe provisioning failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
