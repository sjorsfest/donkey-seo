"""Tests for article generation normalization and defaults."""

from app.services.article_generation import ArticleGenerationService


def test_normalize_document_inserts_hero_and_cta_defaults() -> None:
    service = ArticleGenerationService("donkey.support")

    document = {
        "schema_version": "1.0",
        "seo_meta": {
            "h1": "Pricing Models",
            "meta_title": "",
            "meta_description": "",
            "slug": "",
            "primary_keyword": "",
        },
        "conversion_plan": {},
        "blocks": [
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Overview",
                "body": "Context",
            }
        ],
    }
    brief = {
        "primary_keyword": "software as service pricing models",
        "funnel_stage": "bofu",
        "money_page_links": [{"url": "https://donkey.support"}],
    }
    writer_instructions = {"internal_linking_minimums": {"min_internal": 1, "min_external": 1}}

    normalized = service._normalize_document(document, brief, writer_instructions, ["signup"])

    assert normalized["seo_meta"]["slug"] == "software-as-service-pricing-models"
    assert normalized["seo_meta"]["primary_keyword"] == "software as service pricing models"
    assert normalized["blocks"][0]["block_type"] == "hero"
    assert any(block["block_type"] == "cta" for block in normalized["blocks"])


def test_required_sections_dedupes_sources() -> None:
    service = ArticleGenerationService("donkey.support")

    required = service._required_sections(
        {"must_include_sections": ["Introduction", "FAQ"]},
        {"must_include_sections": ["FAQ", "Conclusion"]},
    )

    assert required == ["Introduction", "FAQ", "Conclusion"]
