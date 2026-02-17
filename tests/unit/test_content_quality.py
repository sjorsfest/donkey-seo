"""Tests for modular article QA checks."""

from app.services.content_quality import evaluate_article_quality


def _base_document() -> dict:
    return {
        "schema_version": "1.0",
        "seo_meta": {
            "h1": "Helpdesk Comparison",
            "meta_title": "Helpdesk Comparison",
            "meta_description": "Compare options",
            "slug": "helpdesk-comparison",
            "primary_keyword": "helpdesk comparison",
        },
        "conversion_plan": {"primary_intent": "mofu", "cta_strategy": []},
        "blocks": [
            {
                "block_type": "hero",
                "semantic_tag": "header",
                "heading": "Helpdesk Comparison",
                "body": "Find the right fit",
            },
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Quick Comparison",
                "level": 2,
                "body": "Evaluate price, speed, and integrations.",
            },
            {
                "block_type": "cta",
                "semantic_tag": "aside",
                "heading": "Try Donkey Support",
                "body": "Start now",
            },
        ],
    }


def test_quality_passes_for_compliant_document() -> None:
    document = _base_document()
    rendered_html = (
        "<article><header><h1>Helpdesk Comparison</h1></header>"
        "<section><h2>Quick Comparison</h2><p>Details</p>"
        '<a href="https://example.org/research">Research</a>'
        '<a href="/blog/helpdesk-basics">Guide</a>'
        "</section><aside>CTA</aside></article>"
    )

    report = evaluate_article_quality(
        document,
        rendered_html,
        required_sections=["Quick Comparison"],
        forbidden_claims=["Unlimited unbranded Pro"],
        target_word_count_min=5,
        target_word_count_max=200,
        min_internal_links=1,
        min_external_links=1,
        require_cta=True,
        first_party_domain="donkey.support",
    )

    assert report["passed"] is True
    assert report["required_failures"] == []


def test_quality_flags_forbidden_claims_and_missing_sections() -> None:
    document = _base_document()
    document["blocks"][1]["heading"] = "Overview"
    document["blocks"][1]["body"] = "Unlimited unbranded Pro"

    rendered_html = "<article><header><h1>Helpdesk Comparison</h1></header></article>"

    report = evaluate_article_quality(
        document,
        rendered_html,
        required_sections=["Quick Comparison"],
        forbidden_claims=["Unlimited unbranded Pro"],
        target_word_count_min=5,
        target_word_count_max=200,
        min_internal_links=1,
        min_external_links=1,
        require_cta=True,
        first_party_domain="donkey.support",
    )

    assert report["passed"] is False
    assert "required_sections" in report["required_failures"]
    assert "forbidden_claims" in report["required_failures"]
