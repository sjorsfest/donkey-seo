"""Unit tests for Step 12 duplicate-coverage guardrails."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.discovery.topic_overlap import normalize_text_tokens
from app.services.steps.content.step_12_brief import ExistingBriefSignature, Step12BriefService


def test_should_skip_when_existing_pair_already_covered() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    topic = SimpleNamespace(
        name="Zendesk vs Intercom",
        description="",
        dominant_intent="commercial",
        dominant_page_type="comparison",
    )
    primary_keyword = SimpleNamespace(keyword="zendesk vs intercom")
    existing = [
        ExistingBriefSignature(
            comparison_key="pair:intercom|zendesk",
            family_key="family:cmp:zendesk",
            intent="commercial",
            page_type="comparison",
            keyword_tokens=normalize_text_tokens("zendesk intercom"),
            text_tokens=normalize_text_tokens("zendesk vs intercom"),
        )
    ]

    should_skip, reason, sibling_allowed = service._should_skip_as_covered(  # type: ignore[attr-defined]
        topic=topic,
        primary_keyword=primary_keyword,
        existing_signatures=existing,
    )

    assert should_skip
    assert reason == "existing_pair_covered"
    assert not sibling_allowed


def test_sibling_pair_is_allowed() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    topic = SimpleNamespace(
        name="Zendesk vs Tidio",
        description="",
        dominant_intent="commercial",
        dominant_page_type="comparison",
    )
    primary_keyword = SimpleNamespace(keyword="zendesk vs tidio")
    existing = [
        ExistingBriefSignature(
            comparison_key="pair:intercom|zendesk",
            family_key="family:cmp:zendesk",
            intent="commercial",
            page_type="comparison",
            keyword_tokens=normalize_text_tokens("zendesk intercom"),
            text_tokens=normalize_text_tokens("zendesk vs intercom"),
        )
    ]

    should_skip, reason, sibling_allowed = service._should_skip_as_covered(  # type: ignore[attr-defined]
        topic=topic,
        primary_keyword=primary_keyword,
        existing_signatures=existing,
    )

    assert not should_skip
    assert reason is None
    assert sibling_allowed


def test_should_skip_high_overlap_same_intent_page_type() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    topic = SimpleNamespace(
        name="Free Cloud Helpdesk CRM",
        description="cloud helpdesk crm workflow",
        dominant_intent="commercial",
        dominant_page_type="landing",
    )
    primary_keyword = SimpleNamespace(keyword="free cloud helpdesk crm")
    existing = [
        ExistingBriefSignature(
            comparison_key=None,
            family_key="family:cloud|crm|free",
            intent="commercial",
            page_type="landing",
            keyword_tokens=normalize_text_tokens("free cloud helpdesk crm"),
            text_tokens=normalize_text_tokens("best free cloud helpdesk crm"),
        )
    ]

    should_skip, reason, sibling_allowed = service._should_skip_as_covered(  # type: ignore[attr-defined]
        topic=topic,
        primary_keyword=primary_keyword,
        existing_signatures=existing,
    )

    assert should_skip
    assert reason == "existing_content_overlap"
    assert not sibling_allowed


@pytest.mark.asyncio
async def test_apply_batch_diversification_selection_uses_llm_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    topics = [
        SimpleNamespace(
            id="t1",
            name="ServiceNow Pricing",
            description="",
            dominant_intent="transactional",
            dominant_page_type="landing",
            funnel_stage="bofu",
            fit_tier="primary",
            fit_score=0.9,
            priority_rank=1,
            priority_score=88.0,
        ),
        SimpleNamespace(
            id="t2",
            name="Zendesk vs Intercom",
            description="",
            dominant_intent="commercial",
            dominant_page_type="comparison",
            funnel_stage="mofu",
            fit_tier="primary",
            fit_score=0.85,
            priority_rank=2,
            priority_score=80.0,
        ),
    ]
    primary_keywords = {
        "t1": SimpleNamespace(keyword="servicenow pricing"),
        "t2": SimpleNamespace(keyword="zendesk vs intercom"),
    }

    async def _fake_run(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return SimpleNamespace(
            decisions=[
                SimpleNamespace(topic_id="t2", decision="include", rationale="better diversity"),
                SimpleNamespace(topic_id="t1", decision="exclude", rationale="near-duplicate with history"),
            ],
            overall_notes="llm_diversified",
        )

    monkeypatch.setattr(service, "_run_brief_diversifier_with_retry", _fake_run)

    selected, skipped, notes = await service._apply_batch_diversification_selection(  # type: ignore[attr-defined]
        topics=topics,  # type: ignore[arg-type]
        primary_keywords_by_topic_id=primary_keywords,  # type: ignore[arg-type]
        existing_history=[],
        target_count=2,
    )

    assert [str(topic.id) for topic in selected] == ["t2"]
    assert skipped == 1
    assert notes == "llm_diversified"


def test_deterministic_batch_diversification_fallback_drops_in_batch_duplicates() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    topics = [
        SimpleNamespace(
            id="t1",
            name="ServiceNow Pricing",
            description="",
            dominant_intent="transactional",
            dominant_page_type="landing",
        ),
        SimpleNamespace(
            id="t2",
            name="ServiceNow Pricing Model",
            description="",
            dominant_intent="transactional",
            dominant_page_type="landing",
        ),
        SimpleNamespace(
            id="t3",
            name="Zendesk vs Intercom",
            description="",
            dominant_intent="commercial",
            dominant_page_type="comparison",
        ),
    ]
    primary_keywords = {
        "t1": SimpleNamespace(keyword="servicenow pricing"),
        "t2": SimpleNamespace(keyword="servicenow pricing model"),
        "t3": SimpleNamespace(keyword="zendesk vs intercom"),
    }

    selected = service._deterministic_batch_diversification_fallback(  # type: ignore[attr-defined]
        topics=topics,  # type: ignore[arg-type]
        primary_keywords_by_topic_id=primary_keywords,  # type: ignore[arg-type]
        target_count=3,
    )

    selected_ids = [str(topic.id) for topic in selected]
    assert selected_ids == ["t1", "t3"]
