"""Tests for prioritization agent prompt construction."""

from app.agents.prioritization_agent import PrioritizationAgent, PrioritizationAgentInput


def test_prompt_formatting_handles_mixed_factor_types() -> None:
    agent = PrioritizationAgent()
    prompt = agent._build_prompt(  # type: ignore[attr-defined]
        PrioritizationAgentInput(
            topics=[
                {
                    "name": "Helpdesk Alternatives",
                    "primary_keyword": "helpdesk alternatives",
                    "keyword_candidates": ["helpdesk alternatives", "zendesk alternatives", "support software alternatives"],
                    "dominant_intent": "commercial",
                    "funnel_stage": "mofu",
                    "total_volume": 1200,
                    "avg_difficulty": 18,
                    "keyword_count": 6,
                    "priority_score": 72.4,
                    "scoring_factors": {
                        "fit_score": 0.81,
                        "effective_market_mode": "established_category",
                        "has_money_page": True,
                        "note": None,
                    },
                }
            ],
            brand_context="Company: Donkey Support",
            money_pages=["https://example.com/pricing"],
            primary_goal="revenue_content",
        )
    )

    assert "effective_market_mode: established_category" in prompt
    assert "has_money_page: true" in prompt
    assert "note: null" in prompt
    assert "Keyword Candidates: helpdesk alternatives, zendesk alternatives, support software alternatives" in prompt


def test_prompt_compact_mode_uses_concise_instruction_block() -> None:
    agent = PrioritizationAgent()
    prompt = agent._build_prompt(  # type: ignore[attr-defined]
        PrioritizationAgentInput(
            topics=[
                {
                    "name": "Support Pricing",
                    "primary_keyword": "support pricing",
                    "keyword_candidates": ["support pricing", "helpdesk pricing"],
                    "dominant_intent": "commercial",
                    "funnel_stage": "bofu",
                    "total_volume": 900,
                    "avg_difficulty": 15,
                    "keyword_count": 3,
                    "priority_score": 80.0,
                    "scoring_factors": {"fit_score": 0.9},
                }
            ],
            compact_mode=True,
        )
    )

    assert "Keep responses concise." in prompt
    assert "recommended_primary_keyword" in prompt


def test_prompt_includes_keyword_candidate_profiles_and_tiebreaker_guidance() -> None:
    agent = PrioritizationAgent()
    prompt = agent._build_prompt(  # type: ignore[attr-defined]
        PrioritizationAgentInput(
            topics=[
                {
                    "name": "Support Automation Guide",
                    "primary_keyword": "support automation guide",
                    "keyword_candidates": ["support automation guide", "helpdesk pricing"],
                    "keyword_candidate_profiles": [
                        {
                            "keyword": "support automation guide",
                            "blog_compatibility": 0.92,
                            "brand_overlap": 0.88,
                            "intent": "informational",
                            "recommended_page_type": "guide",
                            "adjusted_volume": 320,
                        }
                    ],
                    "dominant_intent": "informational",
                    "funnel_stage": "tofu",
                    "total_volume": 1500,
                    "avg_difficulty": 22,
                    "keyword_count": 4,
                    "priority_score": 74.3,
                    "scoring_factors": {"fit_score": 0.84},
                }
            ]
        )
    )

    assert "Keyword Candidate Profiles: support automation guide" in prompt
    assert "blog_fit=0.92" in prompt
    assert "volume only as a tie-breaker" in prompt
