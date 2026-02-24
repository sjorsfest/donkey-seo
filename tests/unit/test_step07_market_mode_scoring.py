"""Unit tests for Step 7 market-aware prioritization scoring."""

from types import SimpleNamespace

from app.services.steps.discovery.step_07_prioritization import (
    DFI_WORKFLOW_THRESHOLD,
    Step07PrioritizationService,
)


def test_dfi_formula_matches_spec() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)

    dfi = service._calculate_demand_fragmentation_index(  # type: ignore[attr-defined]
        keyword_count=12,
        raw_volume_sum=80,
    )

    assert dfi == 0.15


def test_mixed_market_mode_switches_by_dfi_threshold() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)

    workflow_mode = service._resolve_topic_market_mode(  # type: ignore[attr-defined]
        base_market_mode="mixed",
        dfi=DFI_WORKFLOW_THRESHOLD + 0.01,
    )
    established_mode = service._resolve_topic_market_mode(  # type: ignore[attr-defined]
        base_market_mode="mixed",
        dfi=DFI_WORKFLOW_THRESHOLD - 0.01,
    )

    assert workflow_mode == "fragmented_workflow"
    assert established_mode == "established_category"


def test_mode_aware_priority_scoring_uses_expected_weights() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    topic = SimpleNamespace(cluster_coherence=0.7)
    factors = {
        "mean_intent_score": 0.8,
        "difficulty_ease": 0.6,
        "adjusted_volume_norm": 0.4,
        "strategic_fit": 0.9,
        "serp_opportunity_signal": 0.5,
        "raw_volume_norm": 0.7,
        "serp_signal": 0.3,
    }
    fit_assessment = {"components": {"comparison_relevance": 0.2}}

    workflow_opp = service._calculate_opportunity_score(  # type: ignore[attr-defined]
        topic=topic,
        factors=factors,
        effective_market_mode="fragmented_workflow",
        fit_assessment=fit_assessment,
    )
    established_opp = service._calculate_opportunity_score(  # type: ignore[attr-defined]
        topic=topic,
        factors=factors,
        effective_market_mode="established_category",
        fit_assessment=fit_assessment,
    )

    workflow_score = service._calculate_mode_aware_priority_score(  # type: ignore[attr-defined]
        factors=factors,
        effective_market_mode="fragmented_workflow",
        brand_fit_score=0.75,
        opportunity_score=workflow_opp,
    )
    established_score = service._calculate_mode_aware_priority_score(  # type: ignore[attr-defined]
        factors=factors,
        effective_market_mode="established_category",
        brand_fit_score=0.75,
        opportunity_score=established_opp,
    )

    assert round(workflow_opp, 2) == 0.58
    assert round(established_opp, 2) == 0.62
    # deterministic score = (brand_fit*0.7 + opportunity*0.3) * 100
    assert workflow_score == 69.9
    assert established_score == 71.1
