"""OpenRouter model/ranking collector."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.services.model_selector.types import ModelCandidate

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class OpenRouterClient:
    """Client for OpenRouter model listings and ranking order."""

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 20) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def fetch_models(
        self,
        *,
        category: str,
        max_price: float,
        order: str = "top-weekly",
    ) -> tuple[list[ModelCandidate], dict[str, Any]]:
        """Fetch ranked models for a category with max price filter."""
        params = {"order": order, "max_price": max_price}
        category_params = {**params, "category": category}
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response: httpx.Response | None = None
        category_error: str | None = None
        category_fallback_used = False

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    OPENROUTER_MODELS_URL,
                    params=category_params,
                    headers=headers,
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    category_error = str(exc)
                    # Some categories can become invalid/unsupported over time.
                    # Retry once without category so we still get ranked models.
                    fallback_response = await client.get(
                        OPENROUTER_MODELS_URL,
                        params=params,
                        headers=headers,
                    )
                    fallback_response.raise_for_status()
                    response = fallback_response
                    category_fallback_used = True
        except Exception as exc:
            logger.warning(
                "OpenRouter fetch failed",
                extra={"category": category, "max_price": max_price, "error": str(exc)},
            )
            return [], {
                "source": "openrouter",
                "category": category,
                "order": order,
                "ok": False,
                "error": str(exc),
            }

        assert response is not None
        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []

        candidates: list[ModelCandidate] = []
        row_count = len(rows)

        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue

            raw_model_id = str(row.get("id", "")).strip()
            if not raw_model_id:
                continue

            model_id = normalize_openrouter_model_id(raw_model_id)
            price_per_million = extract_price_per_million(row)

            # Enforce max-price in case upstream changes semantics.
            if price_per_million > max_price:
                continue

            rank_score = normalize_rank_score(index, row_count)
            popularity = coerce_float(row.get("popularity"))

            candidates.append(
                ModelCandidate(
                    model=model_id,
                    raw_model_id=raw_model_id,
                    price_per_million=price_per_million,
                    openrouter_rank=index,
                    openrouter_rank_score=rank_score,
                    openrouter_popularity=popularity,
                    source_metadata={
                        "openrouter_category": category,
                        "openrouter_order": order,
                        "openrouter_rank": index,
                    },
                )
            )

        return candidates, {
            "source": "openrouter",
            "category": category,
            "category_fallback_used": category_fallback_used,
            "category_error": category_error,
            "order": order,
            "ok": True,
            "fetched": row_count,
            "kept": len(candidates),
        }


def normalize_openrouter_model_id(raw_model_id: str) -> str:
    """Normalize OpenRouter model ID into pydantic-ai provider format."""
    cleaned = raw_model_id.strip()
    if cleaned.startswith("openrouter:"):
        return cleaned
    return f"openrouter:{cleaned}"


def normalize_rank_score(rank: int, total: int) -> float:
    """Convert rank position into normalized score in [0, 1]."""
    if total <= 1:
        return 1.0
    relative = (rank - 1) / (total - 1)
    return max(0.0, min(1.0, 1.0 - relative))


def extract_price_per_million(row: dict[str, Any]) -> float:
    """Estimate effective max token price in USD per 1M tokens."""
    raw_model_id = str(row.get("id", "")).lower()
    if raw_model_id.endswith(":free"):
        return 0.0

    pricing = row.get("pricing")
    if not isinstance(pricing, dict):
        return float("inf")

    # OpenRouter usually reports token prices in USD/token.
    prompt_price = coerce_float(pricing.get("prompt"))
    completion_price = coerce_float(pricing.get("completion"))

    per_token_prices = [p for p in (prompt_price, completion_price) if p is not None]
    if not per_token_prices:
        return float("inf")

    max_per_token = max(per_token_prices)
    if max_per_token <= 0:
        return 0.0

    return max_per_token * 1_000_000


def coerce_float(value: Any) -> float | None:
    """Safely parse float values from API payloads."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
