"""Tests for deterministic SEO checklist checks."""

from app.services.content_renderer import render_modular_document
from app.services.seo_checklist import (
    run_deterministic_checklist,
    select_content_type_module,
    should_apply_risk_module,
)


def _base_document() -> dict:
    return {
        "schema_version": "1.0",
        "seo_meta": {
            "h1": "Automation Guide",
            "meta_title": "Automation Guide",
            "meta_description": "Guide",
            "slug": "automation-guide",
            "primary_keyword": "helpdesk automation software",
        },
        "conversion_plan": {"primary_intent": "informational", "cta_strategy": []},
        "blocks": [
            {
                "block_type": "hero",
                "semantic_tag": "header",
                "heading": "Automation Guide",
                "body": " ".join(["context"] * 180),
            },
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Overview",
                "level": 2,
                "body": "Useful details for implementation teams.",
            },
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Keyword Strategy",
                "level": 4,
                "body": "helpdesk automation software appears later.",
            },
        ],
    }


def test_select_content_type_module_mapping_and_fallback() -> None:
    assert select_content_type_module("guide", "informational") == "A"
    assert select_content_type_module("comparison", "commercial") == "B"
    assert select_content_type_module("tool", "informational") == "C"
    assert select_content_type_module("opinion", "informational") == "D"
    assert select_content_type_module("unknown", "informational") == "A"


def test_should_apply_risk_module_from_compliance_or_risk_terms() -> None:
    assert should_apply_risk_module(["HIPAA"], ["general content"]) is True
    assert should_apply_risk_module([], ["How to handle personal data safely"]) is True
    assert should_apply_risk_module([], ["General SaaS onboarding guide"]) is False


def test_deterministic_checks_flag_keyword_intro_and_heading_hierarchy() -> None:
    document = _base_document()
    rendered_html = render_modular_document(document)

    report = run_deterministic_checklist(
        document,
        rendered_html,
        primary_keyword="helpdesk automation software",
        page_type="guide",
        search_intent="informational",
        required_sections=["Overview"],
        forbidden_claims=[],
        target_word_count_min=50,
        target_word_count_max=1000,
        min_internal_links=0,
        min_external_links=0,
        require_cta=False,
        first_party_domain="donkey.support",
        compliance_notes=[],
        brief_text_fields=[],
    )

    assert "primary_keyword_first_150_words" in report.hard_failures
    assert "heading_hierarchy" in report.hard_failures


def test_keyword_density_out_of_band_is_soft_warning_only() -> None:
    keyword = "helpdesk automation software"
    high_density_text = (f"{keyword} " * 25) + ("extra words " * 30)
    document = {
        "schema_version": "1.0",
        "seo_meta": {
            "h1": "Helpdesk Automation Software Guide",
            "meta_title": "Helpdesk Automation Software Guide",
            "meta_description": "Guide",
            "slug": "helpdesk-automation-software-guide",
            "primary_keyword": keyword,
        },
        "conversion_plan": {"primary_intent": "informational", "cta_strategy": []},
        "blocks": [
            {
                "block_type": "hero",
                "semantic_tag": "header",
                "heading": "Helpdesk Automation Software Guide",
                "body": high_density_text,
            },
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Helpdesk Automation Software Benefits",
                "level": 2,
                "body": high_density_text,
            },
        ],
    }
    rendered_html = render_modular_document(document)

    report = run_deterministic_checklist(
        document,
        rendered_html,
        primary_keyword=keyword,
        page_type="guide",
        search_intent="informational",
        required_sections=["Benefits"],
        forbidden_claims=[],
        target_word_count_min=50,
        target_word_count_max=2000,
        min_internal_links=0,
        min_external_links=0,
        require_cta=False,
        first_party_domain="donkey.support",
        compliance_notes=[],
        brief_text_fields=[],
    )

    assert "keyword_density_soft_band" in report.soft_warnings
    assert "keyword_density_soft_band" not in report.hard_failures
