"""Tests for deterministic article renderer."""

from app.services.content_renderer import render_modular_document


def test_renderer_maps_blocks_to_semantic_html() -> None:
    document = {
        "schema_version": "1.0",
        "seo_meta": {
            "h1": "Best Helpdesk Tools",
            "meta_title": "Best Helpdesk Tools",
            "meta_description": "Compare top tools",
            "slug": "best-helpdesk-tools",
            "primary_keyword": "best helpdesk tools",
        },
        "conversion_plan": {
            "primary_intent": "mofu",
            "cta_strategy": ["Conclusion CTA"],
        },
        "blocks": [
            {
                "block_type": "hero",
                "semantic_tag": "header",
                "heading": "Best Helpdesk Tools",
                "body": "A practical comparison.",
                "links": [],
            },
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "What to Look For",
                "level": 2,
                "body": "Prioritize speed and integrations.",
                "links": [{"anchor": "See pricing", "href": "https://example.com/pricing"}],
            },
            {
                "block_type": "comparison_table",
                "semantic_tag": "table",
                "heading": "Quick Comparison",
                "table_columns": ["Tool", "Best For"],
                "table_rows": [["Donkey Support", "Small teams"]],
            },
            {
                "block_type": "cta",
                "semantic_tag": "aside",
                "heading": "Try it",
                "body": "Start in 5 minutes.",
                "cta": {"label": "Learn more", "href": "/pricing"},
            },
        ],
    }

    html = render_modular_document(document)

    assert html.startswith("<article>")
    assert '<header data-block-type="hero"><h1>Best Helpdesk Tools</h1>' in html
    assert "<section data-block-type=\"section\">" in html
    assert "<table><thead><tr><th>Tool</th><th>Best For</th></tr></thead>" in html
    assert "<aside data-block-type=\"cta\">" in html
    assert html.count("<h1") == 1


def test_renderer_inserts_fallback_hero_when_missing() -> None:
    document = {
        "schema_version": "1.0",
        "seo_meta": {
            "h1": "Fallback Heading",
            "meta_title": "x",
            "meta_description": "y",
            "slug": "fallback-heading",
            "primary_keyword": "fallback heading",
        },
        "conversion_plan": {},
        "blocks": [
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Body",
                "level": 2,
                "body": "Some content",
            }
        ],
    }

    html = render_modular_document(document)

    assert "<h1>Fallback Heading</h1>" in html
    assert html.count("<h1") == 1
