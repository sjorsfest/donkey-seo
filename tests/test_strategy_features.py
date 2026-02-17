"""Tests for run strategy and fit-gating behavior."""

from types import SimpleNamespace

import pytest

from app.schemas.pipeline import PipelineStartRequest
from app.services.run_strategy import resolve_run_strategy
from app.services.steps.step_03_expansion import Step03ExpansionService
from app.services.steps.step_07_prioritization import Step07PrioritizationService


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
            },
        }
    )

    assert req.strategy is not None
    assert req.strategy.conversion_intents == ["demo", "trial"]
    assert req.strategy.scope_mode == "balanced_adjacent"
    assert req.strategy.branded_keyword_mode == "comparisons_only"


def test_pipeline_start_request_accepts_mode_and_configs() -> None:
    """Pipeline start accepts discovery/content mode payloads."""
    req = PipelineStartRequest.model_validate(
        {
            "mode": "discovery_loop",
            "strategy": {
                "fit_threshold_profile": "aggressive",
            },
            "discovery": {
                "max_iterations": 3,
                "min_eligible_topics": 6,
                "require_serp_gate": True,
                "max_keyword_difficulty": 65,
                "min_domain_diversity": 0.5,
                "require_intent_match": True,
                "auto_start_content": True,
            },
            "content": {
                "max_briefs": 15,
                "posts_per_week": 3,
                "preferred_weekdays": [0, 2, 4],
                "min_lead_days": 7,
            },
        }
    )

    assert req.mode == "discovery_loop"
    assert req.discovery is not None
    assert req.discovery.max_iterations == 3
    assert req.discovery.min_eligible_topics == 6
    assert req.content is not None
    assert req.content.max_briefs == 15
    assert req.content.posts_per_week == 3
    assert req.content.preferred_weekdays == [0, 2, 4]
    assert req.content.min_lead_days == 7


def test_pipeline_start_request_rejects_invalid_preferred_weekday() -> None:
    with pytest.raises(ValueError):
        PipelineStartRequest.model_validate(
            {
                "mode": "content_production",
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
        primary_goal="lead_generation",
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
        primary_goal=None,
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


def test_step07_fit_gating_relaxes_to_secondary_when_needed() -> None:
    """Fit gating relaxes threshold once and labels additional topics as secondary."""
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    strategy = resolve_run_strategy(
        strategy_payload={"fit_threshold_profile": "aggressive"},
        brand=None,
        primary_goal=None,
    )

    scored_topics = [
        {
            "fit_assessment": {
                "fit_score": 0.72,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "fit_assessment": {
                "fit_score": 0.68,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "fit_assessment": {
                "fit_score": 0.65,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "fit_assessment": {
                "fit_score": 0.63,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "fit_assessment": {
                "fit_score": 0.61,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "fit_assessment": {
                "fit_score": 0.59,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "components": {"icp_relevance": 0.2},
            }
        },
    ]

    service._apply_fit_gating(scored_topics, strategy)  # type: ignore[attr-defined]

    tiers = [item["fit_assessment"]["fit_tier"] for item in scored_topics]
    assert tiers[0] == "primary"
    assert tiers.count("secondary") >= 4


def test_step07_fit_gating_adaptive_fallback_promotes_when_relaxed_still_low() -> None:
    """Fallback promotion should prevent zero-eligible output for plausible topics."""
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    strategy = resolve_run_strategy(
        strategy_payload={"fit_threshold_profile": "aggressive", "min_eligible_target": 3},
        brand=None,
        primary_goal=None,
    )

    scored_topics = [
        {
            "fit_assessment": {
                "fit_score": 0.49,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "reasons": [],
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "fit_assessment": {
                "fit_score": 0.47,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "reasons": [],
                "components": {"icp_relevance": 0.2},
            }
        },
        {
            "fit_assessment": {
                "fit_score": 0.45,
                "fit_tier": "excluded",
                "hard_exclusion_reason": None,
                "fit_threshold_used": None,
                "reasons": [],
                "components": {"icp_relevance": 0.2},
            }
        },
    ]

    service._apply_fit_gating(scored_topics, strategy)  # type: ignore[attr-defined]
    tiers = [item["fit_assessment"]["fit_tier"] for item in scored_topics]
    assert tiers.count("secondary") == 3
