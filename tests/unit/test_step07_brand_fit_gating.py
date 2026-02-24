"""Tests for Step 7 brand-first prioritization behavior."""

from types import SimpleNamespace

import pytest

from app.services.run_strategy import resolve_run_strategy
from app.services.steps.discovery.step_07_prioritization import Step07PrioritizationService


def _kw(
    keyword: str,
    *,
    volume: int,
    difficulty: float,
    intent_score: float,
    is_comparison: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        keyword=keyword,
        search_volume=volume,
        adjusted_volume=volume,
        difficulty=difficulty,
        intent_score=intent_score,
        risk_flags=[],
        discovery_signals={"is_comparison": is_comparison},
        serp_mismatch_flags=[],
    )


def test_brand_fit_dominates_over_raw_volume() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    strategy = resolve_run_strategy(
        strategy_payload={
            "include_topics": ["helpdesk", "support", "live chat", "pricing", "alternatives"],
            "fit_threshold_profile": "moderate",
        },
        brand=None,
        primary_goal="revenue_content",
    )
    base_market_mode = "mixed"

    high_volume_offbrand = SimpleNamespace(
        name="Generic Chat Noise",
        description="random chat and gaming terms",
        cluster_notes="irrelevant app and gaming chatter",
        total_volume=3_500_000,
        adjusted_volume_sum=3_500_000,
        avg_difficulty=48.0,
        keyword_count=12,
        cluster_coherence=0.2,
        dominant_intent="informational",
        dominant_page_type="list",
        funnel_stage="tofu",
        primary_keyword_id=None,
        pillar_seed_topic_id=None,
    )
    comparison_topic = SimpleNamespace(
        name="Helpdesk Alternatives Pricing",
        description="support software alternatives and pricing",
        cluster_notes="commercial comparison opportunity",
        total_volume=4_000,
        adjusted_volume_sum=4_000,
        avg_difficulty=18.0,
        keyword_count=8,
        cluster_coherence=0.85,
        dominant_intent="commercial",
        dominant_page_type="comparison",
        funnel_stage="mofu",
        primary_keyword_id=None,
        pillar_seed_topic_id=None,
    )

    offbrand_keywords = [
        _kw("random gaming chat", volume=400_000, difficulty=50, intent_score=0.45),
        _kw("discord game player", volume=300_000, difficulty=45, intent_score=0.40),
    ]
    comparison_keywords = [
        _kw("zendesk alternatives", volume=2000, difficulty=20, intent_score=0.82, is_comparison=True),
        _kw("helpdesk pricing comparison", volume=1800, difficulty=17, intent_score=0.85, is_comparison=True),
    ]

    raw_ref = 3_500_000
    adjusted_ref = 3_500_000

    off_factors = service._calculate_factors(  # type: ignore[attr-defined]
        topic=high_volume_offbrand,
        keywords=offbrand_keywords,
        brand=None,
        money_pages=[],
        raw_volume_reference=raw_ref,
        adjusted_volume_reference=adjusted_ref,
    )
    off_fit = service._calculate_fit_assessment(  # type: ignore[attr-defined]
        topic=high_volume_offbrand,
        keywords=offbrand_keywords,
        brand=None,
        strategy=strategy,
        business_alignment_score=off_factors["business_alignment"],
    )
    off_factors["strategic_fit"] = off_fit["fit_score"]
    off_mode = service._resolve_topic_market_mode(  # type: ignore[attr-defined]
        base_market_mode=base_market_mode,
        dfi=off_factors["demand_fragmentation_index"],
    )
    off_opp = service._calculate_opportunity_score(  # type: ignore[attr-defined]
        topic=high_volume_offbrand,
        factors=off_factors,
        effective_market_mode=off_mode,
        fit_assessment=off_fit,
    )
    off_score = service._calculate_mode_aware_priority_score(  # type: ignore[attr-defined]
        factors=off_factors,
        effective_market_mode=off_mode,
        brand_fit_score=off_fit["fit_score"],
        opportunity_score=off_opp,
    )

    cmp_factors = service._calculate_factors(  # type: ignore[attr-defined]
        topic=comparison_topic,
        keywords=comparison_keywords,
        brand=None,
        money_pages=[],
        raw_volume_reference=raw_ref,
        adjusted_volume_reference=adjusted_ref,
    )
    cmp_fit = service._calculate_fit_assessment(  # type: ignore[attr-defined]
        topic=comparison_topic,
        keywords=comparison_keywords,
        brand=None,
        strategy=strategy,
        business_alignment_score=cmp_factors["business_alignment"],
    )
    cmp_factors["strategic_fit"] = cmp_fit["fit_score"]
    cmp_mode = service._resolve_topic_market_mode(  # type: ignore[attr-defined]
        base_market_mode=base_market_mode,
        dfi=cmp_factors["demand_fragmentation_index"],
    )
    cmp_opp = service._calculate_opportunity_score(  # type: ignore[attr-defined]
        topic=comparison_topic,
        factors=cmp_factors,
        effective_market_mode=cmp_mode,
        fit_assessment=cmp_fit,
    )
    cmp_score = service._calculate_mode_aware_priority_score(  # type: ignore[attr-defined]
        factors=cmp_factors,
        effective_market_mode=cmp_mode,
        brand_fit_score=cmp_fit["fit_score"],
        opportunity_score=cmp_opp,
    )

    assert cmp_fit["fit_score"] > off_fit["fit_score"]
    assert cmp_score > off_score


def test_dynamic_threshold_calibration_handles_compressed_fit_distribution() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    strategy = resolve_run_strategy(
        strategy_payload={"fit_threshold_profile": "aggressive", "min_eligible_target": 6},
        brand=None,
        primary_goal=None,
    )
    scored_topics = [
        {
            "fit_assessment": {"fit_score": 0.48, "fit_tier": "excluded", "hard_exclusion_reason": None, "fit_threshold_used": None, "reasons": [], "components": {}},
            "scoring_factors": {"opportunity_score": 0.60},
            "deterministic_priority_score": 50.0,
        },
        {
            "fit_assessment": {"fit_score": 0.45, "fit_tier": "excluded", "hard_exclusion_reason": None, "fit_threshold_used": None, "reasons": [], "components": {}},
            "scoring_factors": {"opportunity_score": 0.58},
            "deterministic_priority_score": 49.0,
        },
        {
            "fit_assessment": {"fit_score": 0.38, "fit_tier": "excluded", "hard_exclusion_reason": None, "fit_threshold_used": None, "reasons": [], "components": {}},
            "scoring_factors": {"opportunity_score": 0.55},
            "deterministic_priority_score": 46.0,
        },
        {
            "fit_assessment": {"fit_score": 0.31, "fit_tier": "excluded", "hard_exclusion_reason": None, "fit_threshold_used": None, "reasons": [], "components": {}},
            "scoring_factors": {"opportunity_score": 0.50},
            "deterministic_priority_score": 44.0,
        },
        {
            "fit_assessment": {"fit_score": 0.24, "fit_tier": "excluded", "hard_exclusion_reason": None, "fit_threshold_used": None, "reasons": [], "components": {}},
            "scoring_factors": {"opportunity_score": 0.47},
            "deterministic_priority_score": 41.0,
        },
    ]

    service._apply_dynamic_scores(scored_topics)  # type: ignore[attr-defined]
    primary_threshold, secondary_threshold = service._calibrate_dynamic_thresholds(  # type: ignore[attr-defined]
        scored_topics,
        strategy,
    )

    assert primary_threshold > secondary_threshold
    assert primary_threshold <= strategy.base_threshold()
    assert secondary_threshold <= strategy.relaxed_threshold()
    assert secondary_threshold >= 0.27


def test_deterministic_prefilter_excludes_hard_exclusions() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    scored_topics = [
        {
            "dynamic_fit_score": 0.46,
            "fit_assessment": {"hard_exclusion_reason": None, "fit_tier": "excluded"},
        },
        {
            "dynamic_fit_score": 0.44,
            "fit_assessment": {"hard_exclusion_reason": "competitor_branded", "fit_tier": "excluded"},
        },
        {
            "dynamic_fit_score": 0.30,
            "fit_assessment": {"hard_exclusion_reason": None, "fit_tier": "excluded"},
        },
    ]
    candidates = service._deterministic_prefilter(  # type: ignore[attr-defined]
        scored_topics=scored_topics,
        secondary_threshold=0.35,
    )
    assert len(candidates) == 1
    assert candidates[0]["dynamic_fit_score"] == 0.46


@pytest.mark.asyncio
async def test_llm_retry_then_success() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)

    class _FlakyAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, _input: object) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return SimpleNamespace(prioritizations=[], overall_strategy_notes="")

    agent = _FlakyAgent()
    topic = SimpleNamespace(
        id="topic-1",
        name="Helpdesk Alternatives",
        primary_keyword_id=None,
        dominant_intent="commercial",
        funnel_stage="mofu",
        total_volume=1000,
        avg_difficulty=20,
    )
    batch = [
        {
            "topic_id": "topic-1",
            "topic": topic,
            "keywords": [],
            "deterministic_priority_score": 75.0,
            "fit_assessment": {"fit_score": 0.8},
            "scoring_factors": {},
            "effective_market_mode": "established_category",
        }
    ]

    output = await service._run_prioritization_agent_with_retry(  # type: ignore[attr-defined]
        agent=agent,  # type: ignore[arg-type]
        batch=batch,
        brand_context="Company: Donkey Support",
        money_pages=[],
        primary_goal="revenue_content",
    )

    assert output is not None
    assert agent.calls == 2
