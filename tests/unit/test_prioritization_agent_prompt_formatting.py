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


def test_prompt_compact_mode_uses_concise_instruction_block() -> None:
    agent = PrioritizationAgent()
    prompt = agent._build_prompt(  # type: ignore[attr-defined]
        PrioritizationAgentInput(
            topics=[
                {
                    "name": "Support Pricing",
                    "primary_keyword": "support pricing",
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
