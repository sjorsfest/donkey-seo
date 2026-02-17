"""Unit tests for Step 8 SERP validation heuristics and resilience."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.exceptions import APIKeyMissingError
from app.services.steps.step_08_serp import Step08SerpValidationService


@pytest.mark.parametrize(
    ("keyword_text", "organic_results", "serp_features", "expected_intent"),
    [
        (
            "how to set up helpdesk workflow",
            [
                {
                    "title": "How to Build a Helpdesk Workflow",
                    "url": "https://example.com/blog/helpdesk-workflow-guide",
                    "snippet": "Step-by-step tutorial for support teams.",
                }
            ],
            ["paa", "featured_snippet"],
            "informational",
        ),
        (
            "best helpdesk software for startups",
            [
                {
                    "title": "10 Best Helpdesk Software Tools Compared",
                    "url": "https://example.com/reviews/helpdesk-tools",
                    "snippet": "Comparison of top providers and pricing.",
                }
            ],
            [],
            "commercial",
        ),
        (
            "helpdesk pricing",
            [
                {
                    "title": "Helpdesk Pricing Plans",
                    "url": "https://example.com/pricing",
                    "snippet": "Start free trial and pick a plan.",
                }
            ],
            ["shopping"],
            "transactional",
        ),
    ],
)
def test_infer_validated_intent(
    keyword_text: str,
    organic_results: list[dict[str, str]],
    serp_features: list[str],
    expected_intent: str,
) -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)

    result = service._infer_validated_intent(
        keyword_text=keyword_text,
        organic_results=organic_results,
        serp_features=serp_features,
    )

    assert result == expected_intent


@pytest.mark.parametrize(
    ("keyword_text", "expected_page_type"),
    [
        ("zendesk vs freshdesk", "comparison"),
        ("best zendesk alternatives", "alternatives"),
        ("helpdesk pricing", "landing"),
        ("how to set up helpdesk automations", "guide"),
    ],
)
def test_infer_validated_page_type_from_keyword_patterns(
    keyword_text: str,
    expected_page_type: str,
) -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)

    result = service._infer_validated_page_type(
        keyword_text=keyword_text,
        organic_results=[],
        serp_features=[],
        validated_intent="informational",
    )

    assert result == expected_page_type


def test_build_mismatch_flags_soft_validation_does_not_mutate_step5_fields() -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    keyword_model = SimpleNamespace(
        intent="informational",
        recommended_page_type="guide",
    )

    flags = service._build_mismatch_flags(
        keyword_model=keyword_model,
        validated_intent="commercial",
        validated_page_type="comparison",
        fetch_failed=False,
        no_organic=False,
    )

    assert "intent_mismatch" in flags
    assert "page_type_mismatch" in flags
    assert keyword_model.intent == "informational"
    assert keyword_model.recommended_page_type == "guide"


def test_merge_keyword_candidates_includes_primary_and_dedupes() -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    primary_keyword = SimpleNamespace(id="kw-1")
    duplicate_keyword = SimpleNamespace(id="kw-1")
    flagged_keyword = SimpleNamespace(id="kw-2")

    merged = service._merge_keyword_candidates(
        primary_keywords=[primary_keyword],
        flagged_keywords=[duplicate_keyword, flagged_keyword],
    )

    assert {str(item.id) for item in merged} == {"kw-1", "kw-2"}
    assert len(merged) == 2


def test_topic_needs_serp_validation_by_note_or_coherence() -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    by_note = SimpleNamespace(
        cluster_notes="[needs_serp_validation] low confidence",
        cluster_coherence=0.95,
    )
    by_coherence = SimpleNamespace(cluster_notes="", cluster_coherence=0.4)
    normal = SimpleNamespace(cluster_notes="", cluster_coherence=0.95)

    assert service._topic_needs_serp_validation(by_note) is True
    assert service._topic_needs_serp_validation(by_coherence) is True
    assert service._topic_needs_serp_validation(normal) is False


@pytest.mark.asyncio
async def test_fetch_serp_payloads_missing_credentials_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    keywords = [SimpleNamespace(keyword="keyword one"), SimpleNamespace(keyword="keyword two")]

    class MissingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise APIKeyMissingError("DataForSEO")

    monkeypatch.setattr("app.services.steps.step_08_serp.DataForSEOClient", MissingClient)

    payloads, api_calls, warnings = await service._fetch_serp_payloads(
        keywords,
        location_code=2840,
        language_code="en",
    )

    assert len(payloads) == 2
    assert all(payload.get("error") for payload in payloads)
    assert api_calls == 0
    assert any("credentials" in warning.lower() for warning in warnings)


@pytest.mark.asyncio
async def test_fetch_serp_payloads_partial_batch_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    keywords = [SimpleNamespace(keyword=f"keyword {idx}") for idx in range(30)]

    class PartialFailureClient:
        call_count = 0

        async def __aenter__(self) -> "PartialFailureClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get_serp_batch(
            self,
            keywords: list[str],
            location_code: int,
            language_code: str,
            device: str,
            depth: int,
        ) -> list[dict[str, object]]:
            type(self).call_count += 1
            if type(self).call_count == 1:
                raise RuntimeError("batch failed")
            return [
                {
                    "keyword": keyword,
                    "organic_results": [{"title": "Guide", "url": "https://example.com/guide"}],
                    "serp_features": ["paa"],
                }
                for keyword in keywords
            ]

    monkeypatch.setattr("app.services.steps.step_08_serp.DataForSEOClient", PartialFailureClient)

    payloads, api_calls, warnings = await service._fetch_serp_payloads(
        keywords,
        location_code=2840,
        language_code="en",
    )

    assert len(payloads) == 30
    assert api_calls == 1
    assert any("batch" in warning.lower() for warning in warnings)
    assert all(payload.get("error") for payload in payloads[:25])
    assert all(not payload.get("error") for payload in payloads[25:])
