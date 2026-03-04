"""Unit tests for topic generator prompt shaping."""

from app.agents.topic_generator import TopicGeneratorAgent, TopicGeneratorInput


def _build_input(*, existing_coverage: list[str] | None = None) -> TopicGeneratorInput:
    return TopicGeneratorInput(
        company_name="Donkey SEO",
        products_services=[{"name": "Content Pipeline", "description": "Automated SEO content"}],
        offer_categories=["SEO automation"],
        target_audience={
            "target_roles": ["Marketing Lead"],
            "target_industries": ["SaaS"],
            "primary_pains": ["not enough content"],
            "desired_outcomes": ["publish consistently"],
        },
        buyer_jobs=["plan weekly SEO content"],
        conversion_intents=["demo"],
        unique_value_props=["Fully automated pipeline"],
        in_scope_topics=["seo content automation"],
        out_of_scope_topics=["paid ads"],
        existing_coverage=existing_coverage or [],
        learning_context="",
    )


def test_prompt_includes_existing_coverage_section_when_provided() -> None:
    agent = TopicGeneratorAgent(model_override="openai:gpt-4.1-mini")
    prompt = agent._build_prompt(
        _build_input(
            existing_coverage=[
                "Brief keyword: seo content planner",
                "Article title: SEO content planner guide [keyword: seo content planner]",
            ]
        )
    )

    assert "## Existing Coverage (Diversify From These)" in prompt
    assert "Brief keyword: seo content planner" in prompt
    assert "Avoid repeating existing covered angles" in prompt


def test_prompt_omits_existing_coverage_section_when_empty() -> None:
    agent = TopicGeneratorAgent(model_override="openai:gpt-4.1-mini")
    prompt = agent._build_prompt(_build_input(existing_coverage=[]))

    assert "## Existing Coverage (Diversify From These)" not in prompt
