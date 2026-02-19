"""Unit tests for Step 7 market-aware prioritization scoring."""

from app.services.steps.step_07_prioritization import (
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
    factors = {
        "mean_intent_score": 0.8,
        "difficulty_ease": 0.6,
        "adjusted_volume_norm": 0.4,
        "strategic_fit": 0.9,
        "serp_opportunity_signal": 0.5,
        "raw_volume_norm": 0.7,
        "serp_signal": 0.3,
    }

    workflow_score = service._calculate_mode_aware_priority_score(  # type: ignore[attr-defined]
        factors=factors,
        effective_market_mode="fragmented_workflow",
    )
    established_score = service._calculate_mode_aware_priority_score(  # type: ignore[attr-defined]
        factors=factors,
        effective_market_mode="established_category",
    )

    # workflow = (0.8*0.45 + 0.6*0.2 + 0.4*0.15 + 0.9*0.1 + 0.5*0.1) * 100 = 68.0
    assert workflow_score == 68.0
    # established = (0.7*0.4 + 0.6*0.25 + 0.8*0.2 + 0.3*0.15) * 100 = 63.5
    assert established_score == 63.5
