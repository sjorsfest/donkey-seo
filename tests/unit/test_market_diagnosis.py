"""Unit tests for market diagnosis heuristics."""

from app.services.market_diagnosis import diagnose_market_mode, extract_keyword_discovery_signals


def test_market_diagnosis_override_takes_precedence() -> None:
    result = diagnose_market_mode(
        source="step2_initial",
        override="fragmented_workflow",
        seed_terms=["crm software", "helpdesk"],
        keyword_rows=[{"keyword": "crm software", "search_volume": 1000}],
    )

    assert result.mode == "fragmented_workflow"
    assert "override" in result.signals
    assert result.confidence == 1.0


def test_market_diagnosis_fragmented_from_long_tail_workflow_signals() -> None:
    rows = []
    for keyword in [
        "slack to notion webhook",
        "connect jira to slack",
        "replace zapier without code",
        "api integration workflow",
        "sync github to linear",
    ]:
        rows.append({
            "keyword": keyword,
            "search_volume": 0,
            "discovery_signals": extract_keyword_discovery_signals(keyword),
        })

    result = diagnose_market_mode(
        source="step3_refresh",
        override="auto",
        seed_terms=["integration automation", "webhook routing"],
        keyword_rows=rows,
    )

    assert result.mode == "fragmented_workflow"
    assert "workflow_term_ratio" in result.reasons or "two_entity_ratio" in result.reasons


def test_market_diagnosis_established_from_head_terms_and_vendor_serp() -> None:
    rows = [
        {"keyword": "crm software", "search_volume": 12000},
        {"keyword": "best crm", "search_volume": 8000},
        {"keyword": "crm pricing", "search_volume": 5000},
    ]
    serp_rows = [
        {
            "organic_results": [
                {"domain": "vendor-a.com", "url": "https://vendor-a.com/pricing", "title": "CRM Pricing"},
                {"domain": "vendor-b.com", "url": "https://vendor-b.com/product", "title": "CRM Product"},
                {"domain": "vendor-c.com", "url": "https://vendor-c.com/software", "title": "CRM Software"},
            ]
        }
    ]

    result = diagnose_market_mode(
        source="step3_refresh",
        override="auto",
        seed_terms=["crm software"],
        keyword_rows=rows,
        serp_rows=serp_rows,
    )

    assert result.mode == "established_category"
    assert result.signals["strong_head_term_ratio"] > 0.2
