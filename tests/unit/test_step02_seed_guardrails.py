"""Unit tests for Step 2 seed quality guardrails."""

from types import SimpleNamespace

from app.services.steps.step_02_seeds import Step02SeedsService


def test_sanitize_seed_candidates_filters_long_duplicates_and_out_of_scope() -> None:
    service = Step02SeedsService.__new__(Step02SeedsService)

    sanitized = service._sanitize_seed_candidates(  # type: ignore[attr-defined]
        seeds=[
            {"keyword": "Discord support", "bucket_name": "Core Offer", "relevance_score": 0.9},
            {"keyword": "Discord support", "bucket_name": "Core Offer", "relevance_score": 0.8},
            {
                "keyword": "affordable helpdesk webhook client support",
                "bucket_name": "Workflow Integrations",
                "relevance_score": 0.62,
            },
            {"keyword": "phone support software", "bucket_name": "Core Offer", "relevance_score": 0.7},
        ],
        out_of_scope_topics=["Phone support"],
    )

    assert [item["keyword"] for item in sanitized] == ["Discord support"]


def test_union_workflow_seed_expansion_caps_count_and_enforces_shape() -> None:
    service = Step02SeedsService.__new__(Step02SeedsService)
    brand = SimpleNamespace(
        company_name="Donkey Support",
        products_services=[
            {"name": "Donkey Widget", "category": "Support Widget"},
            {"name": "Discord Bot", "category": "Support Integration"},
        ],
        competitor_positioning=[],
    )
    strategy = SimpleNamespace(
        include_topics=[
            "Discord integration for support",
            "Slack integration for support",
            "Telegram integration for support",
            "Automatic email follow-ups",
        ],
    )
    buckets = []
    seeds = [{"keyword": "Discord support", "bucket_name": "Core Offer", "relevance_score": 1.0}]

    _, expanded = service._union_workflow_seed_expansion(  # type: ignore[attr-defined]
        brand=brand,
        strategy=strategy,
        buckets=buckets,
        seeds=seeds,
    )

    synthetic = [seed for seed in expanded if seed["bucket_name"] == "Workflow Integrations"]
    assert len(synthetic) <= service.MAX_WORKFLOW_SYNTHETIC_SEEDS
    assert all(len(seed["keyword"].split()) <= service.MAX_SEED_WORDS for seed in synthetic)
