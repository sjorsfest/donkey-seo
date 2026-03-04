"""Tests for run strategy and fit-gating behavior."""

from types import SimpleNamespace

import pytest

from app.schemas.pipeline import PipelineStartRequest
from app.services.run_strategy import (
    build_adaptive_target_mix,
    resolve_run_strategy,
)
from app.services.steps.discovery.step_03_expansion import Step03ExpansionService
from app.services.steps.discovery.step_07_prioritization import Step07PrioritizationService


def test_pipeline_start_request_accepts_strategy_payload() -> None:
    """Pipeline start accepts strategy payload and keeps values."""
    req = PipelineStartRequest.model_validate(
        {
            "start_step": 0,
            "end_step": 14,
            "strategy": {
                "conversion_intents": ["demo", "trial"],
                "scope_mode": "balanced_adjacent",
                "branded_keyword_mode": "comparisons_only",
                "fit_threshold_profile": "aggressive",
                "include_topics": ["customer onboarding"],
                "exclude_topics": ["medical advice"],
                "icp_roles": ["support manager"],
                "icp_industries": ["saas"],
                "icp_pains": ["slow response times"],
                "market_mode_override": "mixed",
                "intent_mix": {
                    "informational": 0.5,
                    "commercial": 0.3,
                    "transactional": 0.2,
                    "influence": 0.4,
                },
                "funnel_mix": {
                    "tofu": 0.45,
                    "mofu": 0.35,
                    "bofu": 0.2,
                    "influence": 0.3,
                },
            },
        }
    )

    assert req.strategy is not None
    assert req.strategy.conversion_intents == ["demo", "trial"]
    assert req.strategy.scope_mode == "balanced_adjacent"
    assert req.strategy.branded_keyword_mode == "comparisons_only"
    assert req.strategy.market_mode_override == "mixed"
    assert req.strategy.intent_mix.informational == 0.5
    assert req.strategy.funnel_mix.tofu == 0.45


def test_pipeline_start_request_accepts_mode_and_configs() -> None:
    """Pipeline start accepts discovery/content mode payloads."""
    req = PipelineStartRequest.model_validate(
        {
            "mode": "discovery",
            "strategy": {
                "fit_threshold_profile": "aggressive",
                "market_mode_override": "auto",
            },
            "discovery": {
                "max_iterations": 3,
                "min_eligible_topics": 6,
                "require_serp_gate": True,
                "max_keyword_difficulty": 65,
                "min_domain_diversity": 0.5,
                "require_intent_match": True,
                "max_serp_servedness": 0.75,
                "max_serp_competitor_density": 0.7,
                "min_serp_intent_confidence": 0.35,
                "auto_dispatch_content_tasks": True,
                "auto_resume_on_exhaustion": True,
                "exhaustion_cooldown_minutes": 45,
            },
            "content": {
                "max_briefs": 15,
                "posts_per_week": 3,
                "preferred_weekdays": [0, 2, 4],
                "min_lead_days": 7,
            },
        }
    )

    assert req.mode == "discovery"
    assert req.discovery is not None
    assert req.discovery.max_iterations == 3
    assert req.discovery.min_eligible_topics == 6
    assert req.discovery.max_serp_servedness == 0.75
    assert req.discovery.max_serp_competitor_density == 0.7
    assert req.discovery.min_serp_intent_confidence == 0.35
    assert req.discovery.auto_resume_on_exhaustion is True
    assert req.discovery.exhaustion_cooldown_minutes == 45
    assert req.content is not None
    assert req.content.max_briefs == 15
    assert req.content.posts_per_week == 3
    assert req.content.preferred_weekdays == [0, 2, 4]
    assert req.content.min_lead_days == 7
    assert req.content.include_zero_data_topics is True
    assert req.content.zero_data_topic_share == 0.2
    assert req.content.zero_data_fit_score_min == 0.65


def test_pipeline_start_request_accepts_setup_mode() -> None:
    req = PipelineStartRequest.model_validate({"mode": "setup"})

    assert req.mode == "setup"


def test_pipeline_start_request_rejects_invalid_preferred_weekday() -> None:
    with pytest.raises(ValueError):
        PipelineStartRequest.model_validate(
            {
                "mode": "content",
                "content": {
                    "max_briefs": 10,
                    "preferred_weekdays": [0, 7],
                },
            }
        )


def test_resolve_run_strategy_merges_brand_defaults_and_overrides() -> None:
    """Run strategy merges brand defaults with run-level overrides."""
    brand = SimpleNamespace(
        in_scope_topics=["knowledge base"],
        out_of_scope_topics=["gambling"],
        target_roles=["support lead"],
        target_industries=["saas"],
        primary_pains=["ticket overload"],
    )
    strategy = resolve_run_strategy(
        strategy_payload={
            "include_topics": ["chat widget"],
            "exclude_topics": ["adult content"],
            "conversion_intents": ["demo"],
        },
        brand=brand,  # type: ignore[arg-type]
    )

    assert "knowledge base" in strategy.include_topics
    assert "chat widget" in strategy.include_topics
    assert "gambling" in strategy.exclude_topics
    assert "adult content" in strategy.exclude_topics
    assert strategy.conversion_intents == ["demo"]
    assert strategy.icp_roles == ["support lead"]


def test_step03_branded_keyword_policy_comparisons_only() -> None:
    """Competitor-branded keywords are blocked unless comparison intent exists."""
    service = Step03ExpansionService.__new__(Step03ExpansionService)
    strategy = resolve_run_strategy(
        strategy_payload={"branded_keyword_mode": "comparisons_only"},
        brand=None,
    )

    blocked, reason = service._evaluate_keyword_policy(  # type: ignore[attr-defined]
        keyword_normalized="acme pricing",
        seen=set(),
        out_of_scope=set(),
        strategy=strategy,
        own_brand_terms=set(),
        competitor_terms={"acme"},
    )
    allowed, _ = service._evaluate_keyword_policy(  # type: ignore[attr-defined]
        keyword_normalized="acme vs zendesk",
        seen=set(),
        out_of_scope=set(),
        strategy=strategy,
        own_brand_terms=set(),
        competitor_terms={"acme", "zendesk"},
    )

    assert blocked is False
    assert reason == "competitor_branded_non_comparison"
    assert allowed is True


def test_step07_dynamic_calibration_outputs_ordered_thresholds() -> None:
    """Dynamic calibration should produce primary > secondary under compressed scores."""
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    strategy = resolve_run_strategy(
        strategy_payload={"fit_threshold_profile": "aggressive", "min_eligible_target": 1},
        brand=None,
    )

    scored_topics = [
        {
            "scoring_factors": {"opportunity_score": 0.62},
            "fit_assessment": {
                "fit_score": 0.48,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "scoring_factors": {"opportunity_score": 0.58},
            "fit_assessment": {
                "fit_score": 0.44,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "scoring_factors": {"opportunity_score": 0.53},
            "fit_assessment": {
                "fit_score": 0.39,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "scoring_factors": {"opportunity_score": 0.49},
            "fit_assessment": {
                "fit_score": 0.33,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "scoring_factors": {"opportunity_score": 0.45},
            "fit_assessment": {
                "fit_score": 0.29,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "scoring_factors": {"opportunity_score": 0.40},
            "fit_assessment": {
                "fit_score": 0.25,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
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


def test_step07_prefilter_excludes_hard_excluded_topics() -> None:
    """Deterministic prefilter should never include hard exclusions."""
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)

    scored_topics = [
        {
                "fit_assessment": {
                    "fit_score": 0.58,
                    "fit_tier": "excluded",
                    "hard_exclusion_reason": None,
                    "fit_threshold_used": None,
                    "reasons": [],
                    "components": {"icp_relevance": 0.2},
            },
            "dynamic_fit_score": 0.58,
        },
        {
                "fit_assessment": {
                    "fit_score": 0.57,
                    "fit_tier": "excluded",
                    "hard_exclusion_reason": "competitor_branded",
                    "fit_threshold_used": None,
                    "reasons": [],
                    "components": {"icp_relevance": 0.2},
            },
            "dynamic_fit_score": 0.57,
        },
        {
                "fit_assessment": {
                    "fit_score": 0.45,
                    "fit_tier": "excluded",
                    "hard_exclusion_reason": None,
                    "fit_threshold_used": None,
                    "reasons": [],
                    "components": {"icp_relevance": 0.2},
            },
            "dynamic_fit_score": 0.45,
        },
    ]

    candidates = service._deterministic_prefilter(  # type: ignore[attr-defined]
        scored_topics=scored_topics,
        secondary_threshold=0.5,
    )
    assert len(candidates) == 1
    assert candidates[0]["dynamic_fit_score"] == 0.58


def test_resolve_run_strategy_defaults_to_balanced_mix_config() -> None:
    strategy = resolve_run_strategy(
        strategy_payload=None,
        brand=None,
    )

    assert strategy.conversion_intents == []
    assert strategy.intent_mix.to_shares() == {
        "informational": 0.4,
        "commercial": 0.35,
        "transactional": 0.25,
    }
    assert strategy.funnel_mix.to_shares() == {
        "tofu": 0.4,
        "mofu": 0.35,
        "bofu": 0.25,
    }


def test_build_adaptive_target_mix_boosts_underrepresented_bucket() -> None:
    target = build_adaptive_target_mix(
        base_mix={"informational": 0.4, "commercial": 0.35, "transactional": 0.25},
        observed_mix={"informational": 0.2, "commercial": 0.6, "transactional": 0.2},
        influence=0.5,
    )

    assert abs(sum(target.values()) - 1.0) < 0.0001
    assert target["informational"] > 0.4
    assert target["commercial"] < 0.35
