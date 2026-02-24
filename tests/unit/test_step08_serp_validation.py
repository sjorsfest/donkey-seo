"""Unit tests for Step 8 SERP validation heuristics and resilience."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.core.exceptions import APIKeyMissingError
from app.services.steps.discovery.step_08_serp import Step08SerpValidationService


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


def test_sort_keyword_candidates_orders_by_priority_then_volume() -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    prioritized_topics = [
        SimpleNamespace(id="topic-1", priority_rank=2),
        SimpleNamespace(id="topic-2", priority_rank=1),
    ]
    keyword_models = [
        SimpleNamespace(id="kw-1", topic_id="topic-1", search_volume=300, keyword="topic1 high"),
        SimpleNamespace(id="kw-2", topic_id="topic-2", search_volume=100, keyword="topic2 low"),
        SimpleNamespace(id="kw-3", topic_id="topic-2", search_volume=250, keyword="topic2 high"),
        SimpleNamespace(id="kw-4", topic_id=None, search_volume=1000, keyword="unranked"),
    ]

    sorted_keywords = service._sort_keyword_candidates(
        keyword_models,  # type: ignore[arg-type]
        prioritized_topics=prioritized_topics,  # type: ignore[arg-type]
    )

    assert [kw.id for kw in sorted_keywords] == ["kw-3", "kw-2", "kw-1", "kw-4"]


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


def test_keyword_serp_strength_treats_messy_serp_as_low_servedness() -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    keyword_model = SimpleNamespace(
        keyword="connect slack to notion",
        serp_top_results=[
            {"domain": "reddit.com", "url": "https://reddit.com/r/noise", "title": "Need help connecting Slack to Notion"},
            {"domain": "github.com", "url": "https://github.com/issues/1", "title": "Webhook issue"},
            {"domain": "community.example.com", "url": "https://community.example.com/forum/x", "title": "Forum thread"},
        ],
    )

    servedness, competitor_density = service._keyword_serp_strength(keyword_model)  # type: ignore[attr-defined]

    assert servedness < 0.4
    assert competitor_density == 0.0


class _ScalarProxy:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _ResultProxy:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarProxy:
        return _ScalarProxy(self._rows)


class _SessionWithKeywords:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    async def execute(self, _query: object) -> _ResultProxy:
        return _ResultProxy(self._rows)


@pytest.mark.asyncio
async def test_update_topic_serp_signals_uses_alternate_when_primary_missing() -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    topic_id = str(uuid.uuid4())
    primary_id = str(uuid.uuid4())
    alternate_id = str(uuid.uuid4())

    topic = SimpleNamespace(
        id=topic_id,
        primary_keyword_id=primary_id,
        dominant_intent="informational",
        serp_servedness_score=None,
        serp_competitor_density=None,
        serp_intent_confidence=None,
        serp_evidence_source=None,
        serp_evidence_keyword_id=None,
        serp_evidence_keyword_count=None,
    )
    primary_keyword = SimpleNamespace(
        id=primary_id,
        topic_id=topic_id,
        status="active",
        search_volume=10,
        keyword="primary keyword",
        serp_top_results=[],
        validated_intent=None,
    )
    alternate_keyword = SimpleNamespace(
        id=alternate_id,
        topic_id=topic_id,
        status="active",
        search_volume=80,
        keyword="alternate keyword",
        serp_top_results=[
            {"domain": "vendor-a.com", "url": "https://vendor-a.com/integrations", "title": "Integrations"},
            {"domain": "vendor-b.com", "url": "https://vendor-b.com/pricing", "title": "Pricing"},
        ],
        validated_intent="informational",
    )
    service.session = _SessionWithKeywords([primary_keyword, alternate_keyword])

    await service._update_topic_serp_signals([topic])  # type: ignore[attr-defined]

    assert topic.serp_servedness_score is not None
    assert topic.serp_competitor_density is not None
    assert topic.serp_evidence_source == "alternate"


@pytest.mark.asyncio
async def test_fetch_serp_payloads_missing_credentials_returns_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Step08SerpValidationService.__new__(Step08SerpValidationService)
    keywords = [SimpleNamespace(keyword="keyword one"), SimpleNamespace(keyword="keyword two")]

    class MissingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise APIKeyMissingError("DataForSEO")

    monkeypatch.setattr("app.services.steps.discovery.step_08_serp.DataForSEOClient", MissingClient)

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

    monkeypatch.setattr("app.services.steps.discovery.step_08_serp.DataForSEOClient", PartialFailureClient)

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
