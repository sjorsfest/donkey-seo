"""Unit tests for Step 7 V2 hybrid final-cut helpers."""

from types import SimpleNamespace

from app.services.run_strategy import resolve_run_strategy
from app.services.steps.discovery.step_07_prioritization import Step07PrioritizationService


def test_final_cut_pool_limit_respects_min_and_max_bounds() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)

    tiny_target_strategy = resolve_run_strategy(
        strategy_payload={"min_eligible_target": 1},
        brand=None,
        primary_goal=None,
    )
    large_target_strategy = resolve_run_strategy(
        strategy_payload={"min_eligible_target": 20},
        brand=None,
        primary_goal=None,
    )

    assert service._final_cut_pool_limit(tiny_target_strategy) == 20  # type: ignore[attr-defined]
    assert service._final_cut_pool_limit(large_target_strategy) == 40  # type: ignore[attr-defined]


def test_resolve_final_cut_tier_allows_primary_within_tolerance() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)

    fit_tier, reason = service._resolve_final_cut_tier(  # type: ignore[attr-defined]
        llm_tier="primary",
        adjusted_fit=0.66,
        primary_threshold=0.70,
        secondary_threshold=0.58,
    )

    assert fit_tier == "primary"
    assert reason == "llm_primary"


def test_resolve_final_cut_tier_excludes_when_below_secondary_tolerance() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)

    fit_tier, reason = service._resolve_final_cut_tier(  # type: ignore[attr-defined]
        llm_tier="secondary",
        adjusted_fit=0.51,
        primary_threshold=0.70,
        secondary_threshold=0.58,
    )

    assert fit_tier == "excluded"
    assert reason == "below_threshold_after_llm"


def test_zero_result_fallback_promotes_only_non_hard_excluded_candidates() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    strategy = resolve_run_strategy(
        strategy_payload={"min_eligible_target": 2},
        brand=None,
        primary_goal=None,
    )

    hard_excluded = {
        "fit_assessment": {
            "fit_tier": "excluded",
            "hard_exclusion_reason": "competitor_branded",
            "final_cut_reason_code": "hard_exclusion",
        }
    }
    clean_candidate = {
        "fit_assessment": {
            "fit_tier": "excluded",
            "hard_exclusion_reason": None,
            "final_cut_reason_code": "below_threshold_after_llm",
        }
    }
    scored_topics = [hard_excluded, clean_candidate]

    service._apply_zero_result_fallback(  # type: ignore[attr-defined]
        scored_topics=scored_topics,
        deterministic_candidates=[hard_excluded, clean_candidate],
        secondary_threshold=0.55,
        strategy=strategy,
    )

    assert hard_excluded["fit_assessment"]["fit_tier"] == "excluded"
    assert clean_candidate["fit_assessment"]["fit_tier"] == "secondary"
    assert clean_candidate["fit_assessment"]["final_cut_reason_code"] == "llm_zero_result_fallback"


def test_diversification_never_reincludes_hard_exclusions() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    hard_excluded_topic = SimpleNamespace(
        id="t1",
        name="Off-brand competitor noise",
        description="",
        cluster_notes="",
        dominant_intent="informational",
        dominant_page_type="guide",
        primary_keyword_id="kw-t1",
        serp_evidence_keyword_id=None,
    )
    kept_topic = SimpleNamespace(
        id="t2",
        name="Zendesk vs Intercom",
        description="",
        cluster_notes="",
        dominant_intent="commercial",
        dominant_page_type="comparison",
        primary_keyword_id="kw-t2",
        serp_evidence_keyword_id=None,
    )
    hard_excluded = {
        "topic_id": "t1",
        "topic": hard_excluded_topic,
        "keywords": [SimpleNamespace(id="kw-t1", keyword="off-brand competitor", serp_top_results=[])],
        "priority_score": 99.0,
        "fit_assessment": {
            "fit_tier": "excluded",
            "hard_exclusion_reason": "competitor_branded",
            "final_cut_reason_code": "hard_exclusion",
            "fit_threshold_used": 0.5,
        },
    }
    eligible = {
        "topic_id": "t2",
        "topic": kept_topic,
        "keywords": [SimpleNamespace(id="kw-t2", keyword="zendesk vs intercom", serp_top_results=[])],
        "priority_score": 80.0,
        "fit_assessment": {
            "fit_tier": "primary",
            "hard_exclusion_reason": None,
            "final_cut_reason_code": "llm_primary",
            "fit_threshold_used": 0.5,
        },
    }

    service._apply_diversification(  # type: ignore[attr-defined]
        scored_topics=[hard_excluded, eligible],
        primary_threshold=0.7,
        secondary_threshold=0.55,
    )

    assert hard_excluded["fit_assessment"]["fit_tier"] == "excluded"
    assert hard_excluded["fit_assessment"]["final_cut_reason_code"] == "hard_exclusion"


def test_fit_tier_sort_order_keeps_primary_before_secondary() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    items = [
        {"fit_assessment": {"fit_tier": "secondary"}, "priority_score": 95.0},
        {"fit_assessment": {"fit_tier": "primary"}, "priority_score": 80.0},
        {"fit_assessment": {"fit_tier": "excluded"}, "priority_score": 100.0},
    ]

    items.sort(
        key=lambda item: (
            service._fit_tier_sort_value(item["fit_assessment"].get("fit_tier")),  # type: ignore[attr-defined]
            -float(item.get("priority_score") or 0.0),
        )
    )

    assert [item["fit_assessment"]["fit_tier"] for item in items] == [
        "primary",
        "secondary",
        "excluded",
    ]


def test_rank_keywords_for_topic_prioritization_prefers_topic_relevant_keyword() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    topic = SimpleNamespace(
        name="Chat Widget Implementation",
        description="Custom chat widgets for support teams",
        cluster_notes="Developer implementation guides",
    )
    relevant = SimpleNamespace(
        id="kw-1",
        keyword="custom chat widget implementation",
        adjusted_volume=200,
        search_volume=200,
        intent_score=0.8,
        difficulty=25.0,
        discovery_signals={"has_integration_term": True},
    )
    outlier = SimpleNamespace(
        id="kw-2",
        keyword="elementor google reviews widget",
        adjusted_volume=900,
        search_volume=900,
        intent_score=0.5,
        difficulty=35.0,
        discovery_signals={},
    )

    ranked = service._rank_keywords_for_topic_prioritization(  # type: ignore[attr-defined]
        topic=topic,
        keywords=[outlier, relevant],  # type: ignore[arg-type]
    )

    assert [kw.id for kw in ranked][:2] == ["kw-1", "kw-2"]


def test_resolve_post_prioritization_primary_keyword_uses_llm_recommendation() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    topic = SimpleNamespace(
        primary_keyword_id=None,
        name="Support Software Alternatives",
        description="",
        cluster_notes="",
    )
    first = SimpleNamespace(
        id="kw-1",
        keyword="generic support software",
        adjusted_volume=1000,
        search_volume=1000,
        intent_score=0.5,
        difficulty=50.0,
        discovery_signals={},
    )
    second = SimpleNamespace(
        id="kw-2",
        keyword="donkey support alternatives",
        adjusted_volume=120,
        search_volume=120,
        intent_score=0.9,
        difficulty=20.0,
        discovery_signals={"is_comparison": True},
    )
    resolved = service._resolve_post_prioritization_primary_keyword(  # type: ignore[attr-defined]
        topic=topic,
        keywords=[first, second],  # type: ignore[arg-type]
        prioritization={"recommended_primary_keyword": "donkey support alternatives"},
    )

    assert resolved is not None
    assert resolved.id == "kw-2"
