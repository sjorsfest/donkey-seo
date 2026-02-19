"""Unit tests for Step 12 SERP-enriched brief input behavior."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from app.services.steps.step_12_brief import (
    BriefInput,
    BriefOutput,
    PublicationScheduleConfig,
    Step12BriefService,
)


def test_resolve_brief_serp_profile_prefers_validated_values() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    primary_kw = SimpleNamespace(
        validated_intent="commercial",
        validated_page_type="comparison",
        serp_features=["paa", "featured_snippet"],
        format_requirements=["comparison_content", "comparison_table"],
        serp_mismatch_flags=["page_type_mismatch"],
    )
    topic = SimpleNamespace(
        dominant_intent="informational",
        dominant_page_type="guide",
    )

    profile = service._resolve_brief_serp_profile(primary_kw, topic)

    assert profile["search_intent"] == "commercial"
    assert profile["page_type"] == "comparison"
    assert profile["serp_features"] == ["paa", "featured_snippet"]
    assert profile["competitors_content_types"] == [
        "comparison_content",
        "comparison_table",
    ]
    assert profile["serp_mismatch_flags"] == ["page_type_mismatch"]


def test_resolve_brief_serp_profile_falls_back_to_topic_values() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    primary_kw = SimpleNamespace(
        validated_intent=None,
        validated_page_type=None,
        serp_features=None,
        format_requirements=None,
        serp_mismatch_flags=None,
    )
    topic = SimpleNamespace(
        dominant_intent="informational",
        dominant_page_type="guide",
    )

    profile = service._resolve_brief_serp_profile(primary_kw, topic)

    assert profile["search_intent"] == "informational"
    assert profile["page_type"] == "guide"
    assert profile["serp_features"] == []
    assert profile["competitors_content_types"] == []
    assert profile["serp_mismatch_flags"] == []


def test_collect_warnings_includes_serp_mismatch_flags() -> None:
    service = Step12BriefService.__new__(Step12BriefService)

    warnings = service._collect_warnings(
        collision_status="safe",
        do_not_target=[],
        overlap_status="checked",
        serp_mismatch_flags=[
            "intent_mismatch",
            "page_type_mismatch",
            "serp_fetch_failed",
        ],
    )

    assert any("SERP intent differs" in warning for warning in warnings)
    assert any("SERP page type differs" in warning for warning in warnings)
    assert any("fetch failed" in warning for warning in warnings)


class _ScalarProxy:
    def __init__(self, item: object | None) -> None:
        self._item = item

    def first(self) -> object | None:
        return self._item


class _ResultProxy:
    def __init__(self, item: object | None) -> None:
        self._item = item

    def scalars(self) -> _ScalarProxy:
        return _ScalarProxy(self._item)


class _SessionWithExisting:
    async def execute(self, _query: object) -> _ResultProxy:
        return _ResultProxy(object())


@pytest.mark.asyncio
async def test_validate_output_allows_zero_when_existing_brief_already_present() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    service.session = _SessionWithExisting()

    await service._validate_output(
        BriefOutput(briefs_generated=0, briefs_with_warnings=0, briefs=[]),
        BriefInput(project_id="project-1"),
    )


def test_build_publication_slots_supports_multi_post_weekly_cadence() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    config = PublicationScheduleConfig(
        posts_per_week=3,
        weekdays=[1, 2, 4],
        min_lead_days=7,
        start_date=None,
        use_llm_timing_hints=True,
        llm_timing_flex_days=14,
    )

    slots = service._build_publication_slots(
        topic_count=4,
        today=date(2026, 2, 17),  # Tuesday
        config=config,
    )

    assert slots == [
        date(2026, 2, 24),  # Tue
        date(2026, 2, 25),  # Wed
        date(2026, 2, 27),  # Fri
        date(2026, 3, 3),   # Tue (next week)
    ]


def test_select_proposed_publication_date_uses_close_llm_hint_slot() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    config = PublicationScheduleConfig(
        posts_per_week=3,
        weekdays=[0, 2, 4],
        min_lead_days=7,
        start_date=None,
        use_llm_timing_hints=True,
        llm_timing_flex_days=2,
    )
    slots = [date(2026, 2, 24), date(2026, 2, 25), date(2026, 2, 27)]

    selected = service._select_proposed_publication_date(
        llm_date=date(2026, 2, 26),  # Thursday -> nearest scheduled slot is Friday
        available_slots=slots,
        today=date(2026, 2, 17),
        config=config,
    )

    assert selected == date(2026, 2, 27)
    assert slots == [date(2026, 2, 24), date(2026, 2, 25)]


def test_select_proposed_publication_date_falls_back_when_out_of_range() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    config = PublicationScheduleConfig(
        posts_per_week=2,
        weekdays=[1, 3],
        min_lead_days=5,
        start_date=None,
        use_llm_timing_hints=True,
        llm_timing_flex_days=14,
    )
    slots = [date(2026, 2, 24), date(2026, 2, 26)]

    selected = service._select_proposed_publication_date(
        llm_date=date(2027, 3, 1),
        available_slots=slots,
        today=date(2026, 2, 17),
        config=config,
    )

    assert selected == date(2026, 2, 24)


def test_select_topics_for_briefs_reserves_zero_data_slots() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    topics = [
        SimpleNamespace(
            id="t1",
            priority_rank=1,
            priority_factors={"fit_score": 0.70, "fit_tier": "primary"},
        ),
        SimpleNamespace(
            id="t2",
            priority_rank=2,
            priority_factors={"fit_score": 0.72, "fit_tier": "primary"},
        ),
        SimpleNamespace(
            id="t3",
            priority_rank=3,
            priority_factors={"fit_score": 0.71, "fit_tier": "primary"},
        ),
        SimpleNamespace(
            id="t4",
            priority_rank=4,
            priority_factors={"fit_score": 0.93, "fit_tier": "secondary"},
        ),
    ]
    primary_keywords = {
        "t1": SimpleNamespace(search_volume=120, metrics_confidence=1.0),
        "t2": SimpleNamespace(search_volume=80, metrics_confidence=1.0),
        "t3": SimpleNamespace(search_volume=40, metrics_confidence=1.0),
        "t4": SimpleNamespace(search_volume=None, metrics_confidence=0.1),
    }
    input_data = BriefInput(
        project_id="project-1",
        max_briefs=3,
        include_zero_data_topics=True,
        zero_data_topic_share=0.34,
        zero_data_fit_score_min=0.8,
    )

    selected = service._select_topics_for_briefs(
        topics=topics,  # type: ignore[arg-type]
        primary_keywords_by_topic_id=primary_keywords,  # type: ignore[arg-type]
        input_data=input_data,
    )
    selected_ids = [topic.id for topic in selected]

    assert selected_ids == ["t1", "t2", "t4"]


def test_select_topics_for_briefs_skips_zero_data_when_disabled() -> None:
    service = Step12BriefService.__new__(Step12BriefService)
    topics = [
        SimpleNamespace(
            id="t1",
            priority_rank=1,
            priority_factors={"fit_score": 0.70, "fit_tier": "primary"},
        ),
        SimpleNamespace(
            id="t2",
            priority_rank=2,
            priority_factors={"fit_score": 0.72, "fit_tier": "primary"},
        ),
        SimpleNamespace(
            id="t3",
            priority_rank=3,
            priority_factors={"fit_score": 0.93, "fit_tier": "secondary"},
        ),
    ]
    primary_keywords = {
        "t1": SimpleNamespace(search_volume=120, metrics_confidence=1.0),
        "t2": SimpleNamespace(search_volume=80, metrics_confidence=1.0),
        "t3": SimpleNamespace(search_volume=None, metrics_confidence=0.1),
    }
    input_data = BriefInput(
        project_id="project-1",
        max_briefs=2,
        include_zero_data_topics=False,
        zero_data_topic_share=0.5,
        zero_data_fit_score_min=0.8,
    )

    selected = service._select_topics_for_briefs(
        topics=topics,  # type: ignore[arg-type]
        primary_keywords_by_topic_id=primary_keywords,  # type: ignore[arg-type]
        input_data=input_data,
    )
    selected_ids = [topic.id for topic in selected]

    assert selected_ids == ["t1", "t2"]
