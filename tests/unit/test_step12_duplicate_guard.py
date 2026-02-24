"""Unit tests for Step 12 duplicate-coverage guardrails."""

from __future__ import annotations

from types import SimpleNamespace

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
