"""Unit tests for model selector components."""

from __future__ import annotations

from typing import Any

import pytest

from app.services.model_selector.arena_client import (
    extract_use_case_slugs_from_leaderboard_html,
    extract_use_case_slugs_from_sitemap_xml,
    parse_arena_leaderboard_rows,
)
from app.services.model_selector.model_aliases import resolve_arena_model_to_candidate
from app.services.model_selector.openrouter_client import OpenRouterClient
from app.services.model_selector.scoring import select_best_model
from app.services.model_selector.types import ModelCandidate


@pytest.mark.asyncio
async def test_openrouter_client_filters_by_max_price(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenRouter collector keeps only models under price cap."""
    payload = {
        "data": [
            {
                "id": "google/gemma-3-27b-it:free",
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "openai/gpt-4o",
                "pricing": {"prompt": "0.000005", "completion": "0.000015"},
            },
        ]
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return payload

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "app.services.model_selector.openrouter_client.httpx.AsyncClient",
        FakeAsyncClient,
    )

    client = OpenRouterClient(api_key=None)
    candidates, meta = await client.fetch_models(category="reasoning", max_price=0.0)

    assert meta["ok"] is True
    assert len(candidates) == 1
    assert candidates[0].model == "openrouter:google/gemma-3-27b-it:free"


@pytest.mark.asyncio
async def test_openrouter_client_retries_without_category_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter collector retries without category when category query fails."""
    payload = {
        "data": [
            {
                "id": "google/gemma-3-27b-it:free",
                "pricing": {"prompt": "0", "completion": "0"},
            }
        ]
    }

    class FakeResponse:
        def __init__(self, status_code: int, body: dict[str, Any] | None = None) -> None:
            self.status_code = status_code
            self._body = body or {}
            self.request = httpx.Request("GET", "https://openrouter.ai/api/v1/models")

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("bad request", request=self.request, response=self)

        def json(self) -> dict[str, Any]:
            return self._body

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls = 0

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(status_code=422)
            return FakeResponse(status_code=200, body=payload)

    import httpx

    monkeypatch.setattr(
        "app.services.model_selector.openrouter_client.httpx.AsyncClient",
        FakeAsyncClient,
    )

    client = OpenRouterClient(api_key=None)
    candidates, meta = await client.fetch_models(category="invalid", max_price=0.0)

    assert meta["ok"] is True
    assert meta["category_fallback_used"] is True
    assert len(candidates) == 1
    assert candidates[0].model == "openrouter:google/gemma-3-27b-it:free"


def test_arena_parsers_extract_use_cases_and_rows() -> None:
    """Arena sitemap + leaderboard parsers return expected values."""
    xml = """
    <urlset>
      <url><loc>https://arena.ai/nl/leaderboard?use_case=reasoning</loc></url>
      <url><loc>https://arena.ai/nl/leaderboard?use_case=tool_use</loc></url>
    </urlset>
    """
    html = """
    <table>
      <tbody>
        <tr><td>Gemma 3 27B IT</td><td>1452.1</td><td>$0.00</td></tr>
        <tr><td>Claude Sonnet 4.5</td><td>1501.4</td><td>$3.00</td></tr>
      </tbody>
    </table>
    """

    slugs = extract_use_case_slugs_from_sitemap_xml(xml)
    rows = parse_arena_leaderboard_rows(html)

    assert "reasoning" in slugs
    assert "tool_use" in slugs
    assert len(rows) == 2
    assert rows[0]["model_name"] == "Gemma 3 27B IT"
    assert rows[1]["score"] == 1501.4


def test_arena_leaderboard_use_case_fallback_parser() -> None:
    """Leaderboard HTML parser extracts use_case slugs from links/scripts."""
    html = """
    <html>
      <body>
        <a href="/nl/leaderboard?sort_by=score&use_case=tool_use&time_period=day">Tool use</a>
        <a href="/nl/leaderboard?sort_by=score&use_case=classification&time_period=day">Classify</a>
        <script>
          const state = {"filters":[{"use_case":"reasoning"}]};
        </script>
      </body>
    </html>
    """

    slugs = extract_use_case_slugs_from_leaderboard_html(html)

    assert slugs == {"tool_use", "classification", "reasoning"}


def test_alias_matching_resolves_arena_name_to_candidate() -> None:
    """Alias matcher maps Arena display names to OpenRouter IDs."""
    candidates = [
        "openrouter:anthropic/claude-sonnet-4-5",
        "openrouter:google/gemma-3-27b-it:free",
    ]

    matched = resolve_arena_model_to_candidate("Claude Sonnet 4.5", candidates)
    assert matched == "openrouter:anthropic/claude-sonnet-4-5"


def test_scoring_without_arena_signal_prefers_openrouter_rank() -> None:
    """Missing Arena signal leaves OpenRouter rank as the deciding factor."""
    candidates = [
        ModelCandidate(
            model="openrouter:model-top",
            raw_model_id="provider/model-top",
            price_per_million=0.0,
            openrouter_rank=1,
            openrouter_rank_score=1.0,
        ),
        ModelCandidate(
            model="openrouter:model-second",
            raw_model_id="provider/model-second",
            price_per_million=0.0,
            openrouter_rank=2,
            openrouter_rank_score=0.5,
        ),
    ]

    selection = select_best_model(
        agent_class="DummyAgent",
        candidates=candidates,
        raw_arena_scores={},
        openrouter_weight=0.75,
        arena_weight=0.25,
        max_price=0.0,
        fallback_model="openrouter:fallback:free",
        openrouter_meta={"ok": True},
        arena_meta={"ok": False},
    )

    assert selection.model == "openrouter:model-top"
    assert selection.fallback_used is False


def test_scoring_falls_back_when_no_candidate_under_cap() -> None:
    """Policy fallback is used when no candidate survives price filter."""
    candidates = [
        ModelCandidate(
            model="openrouter:model-paid",
            raw_model_id="provider/model-paid",
            price_per_million=5.0,
            openrouter_rank=1,
            openrouter_rank_score=1.0,
        )
    ]

    selection = select_best_model(
        agent_class="DummyAgent",
        candidates=candidates,
        raw_arena_scores={},
        openrouter_weight=0.75,
        arena_weight=0.25,
        max_price=0.0,
        fallback_model="openrouter:fallback:free",
        openrouter_meta={"ok": True},
        arena_meta={"ok": False},
    )

    assert selection.model == "openrouter:fallback:free"
    assert selection.fallback_used is True
