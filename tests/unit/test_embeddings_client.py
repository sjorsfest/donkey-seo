"""Unit tests for embeddings client behavior."""

from __future__ import annotations

from typing import Any

import pytest

from app.config import settings
from app.core.exceptions import APIKeyMissingError
from app.integrations.embeddings import EmbeddingsClient


@pytest.mark.asyncio
async def test_embeddings_client_uses_openrouter_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddings client calls OpenRouter embeddings endpoint with expected payload."""
    captured: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            # Out-of-order indices to verify client sorting behavior.
            return {
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["init"] = {"args": args, "kwargs": kwargs}

        async def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
            captured["post"] = {"url": url, "json": json}
            return FakeResponse()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("app.integrations.embeddings.httpx.AsyncClient", FakeAsyncClient)

    async with EmbeddingsClient(api_key="sk-or-test-key") as client:
        vectors = await client.get_embeddings(["alpha", "beta"])

    assert captured["post"]["url"] == EmbeddingsClient.EMBEDDING_URL
    assert captured["post"]["json"]["model"] == settings.embeddings_model
    assert captured["post"]["json"]["provider"] == {
        "order": [settings.embeddings_provider],
        "allow_fallbacks": settings.embeddings_allow_fallbacks,
    }
    assert vectors == [[0.1, 0.2], [0.2, 0.3]]


def test_embeddings_client_requires_openrouter_key_from_settings() -> None:
    """Missing OpenRouter key raises API key error at client construction time."""
    original_key = settings.openrouter_api_key
    try:
        settings.openrouter_api_key = None
        with pytest.raises(APIKeyMissingError):
            EmbeddingsClient()
    finally:
        settings.openrouter_api_key = original_key
