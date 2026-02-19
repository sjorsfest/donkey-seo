"""Unit tests for Step 3 discovery signal extraction and filtering behavior."""

from types import SimpleNamespace

from app.services.market_diagnosis import extract_keyword_discovery_signals
from app.services.run_strategy import resolve_run_strategy
from app.services.steps.discovery.step_03_expansion import Step03ExpansionService


def test_extract_keyword_discovery_signals_workflow_and_entities() -> None:
    signals = extract_keyword_discovery_signals(
        "connect slack to notion webhook",
        known_entities={"slack", "notion"},
    )

    assert signals["has_action_verb"] is True
    assert signals["has_integration_term"] is True
    assert signals["has_two_entities"] is True
    assert signals["word_count"] >= 4
    assert set(signals["matched_entities"]) == {"slack", "notion"}


def test_extract_keyword_discovery_signals_comparison() -> None:
    signals = extract_keyword_discovery_signals(
        "zapier alternative",
        known_entities={"zapier"},
    )

    assert signals["is_comparison"] is True
    assert signals["comparison_target"] == "zapier"


def test_step03_policy_does_not_reject_for_low_or_zero_volume() -> None:
    service = Step03ExpansionService.__new__(Step03ExpansionService)
    strategy = resolve_run_strategy(
        strategy_payload={"branded_keyword_mode": "allow_all"},
        brand=None,
        primary_goal=None,
    )

    include, reason = service._evaluate_keyword_policy(  # type: ignore[attr-defined]
        keyword_normalized="slack to notion integration",
        seen=set(),
        out_of_scope=set(),
        strategy=strategy,
        own_brand_terms=set(),
        competitor_terms=set(),
    )

    assert include is True
    assert reason is None


def test_step03_policy_rejects_low_strategic_relevance_when_terms_are_available() -> None:
    service = Step03ExpansionService.__new__(Step03ExpansionService)
    strategy = resolve_run_strategy(
        strategy_payload={"branded_keyword_mode": "allow_all"},
        brand=None,
        primary_goal=None,
    )

    include, reason = service._evaluate_keyword_policy(  # type: ignore[attr-defined]
        keyword_normalized="iflow support",
        seen=set(),
        out_of_scope=set(),
        strategy=strategy,
        own_brand_terms=set(),
        competitor_terms=set(),
        strategic_terms={"discord", "slack", "telegram", "webhook", "widget"},
    )

    assert include is False
    assert reason == "low_strategic_relevance"


def test_step03_extract_competitor_terms_from_seed_fallback() -> None:
    service = Step03ExpansionService.__new__(Step03ExpansionService)
    seeds = [
        SimpleNamespace(name="Zendesk alternative"),
        SimpleNamespace(name="freshdesk alternatives"),
        SimpleNamespace(name="intercom vs helpscout"),
    ]

    competitors = service._extract_competitor_terms_from_seeds(seeds)  # type: ignore[attr-defined]

    assert {"zendesk", "freshdesk", "intercom", "helpscout"}.issubset(competitors)


def test_step03_build_seed_caps_distributes_budget_evenly() -> None:
    service = Step03ExpansionService.__new__(Step03ExpansionService)
    seeds = [
        SimpleNamespace(id="s1"),
        SimpleNamespace(id="s2"),
        SimpleNamespace(id="s3"),
    ]

    caps = service._build_seed_caps(seeds, 10)  # type: ignore[attr-defined]

    assert sum(caps.values()) == 10
    assert sorted(caps.values(), reverse=True) == [4, 3, 3]


def test_step03_ingest_rows_respects_seed_cap() -> None:
    service = Step03ExpansionService.__new__(Step03ExpansionService)
    strategy = resolve_run_strategy(
        strategy_payload={"branded_keyword_mode": "allow_all"},
        brand=None,
        primary_goal=None,
    )
    seed = SimpleNamespace(id="seed-1", name="Discord support")

    seen_keywords: set[str] = set()
    all_keywords: list[dict[str, object]] = []
    active_by_seed = {"seed-1": 0}

    excluded = service._ingest_seed_keyword_rows(  # type: ignore[attr-defined]
        rows=[
            {"keyword": "discord support", "search_volume": 100, "cpc": 1.0, "competition": 0.4},
            {"keyword": "discord ticket bot", "search_volume": 90, "cpc": 1.1, "competition": 0.5},
        ],
        seed=seed,
        source_method="suggestion",
        known_entities={"discord"},
        seen_keywords=seen_keywords,
        out_of_scope=set(),
        strategy=strategy,
        own_brand_terms=set(),
        competitor_terms=set(),
        strategic_terms={"discord", "widget", "support", "ticket"},
        all_keywords=all_keywords,
        active_by_seed=active_by_seed,
        seed_active_cap=1,
    )

    assert excluded == 0
    assert active_by_seed["seed-1"] == 1
    assert len(all_keywords) == 1
