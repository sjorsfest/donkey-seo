"""Tests for source requirements in the article writer schema contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.article_writer import ArticleDocumentResult


def _base_payload() -> dict:
    return {
        "schema_version": "1.0",
        "seo_meta": {
            "h1": "Pricing Guide",
            "meta_title": "Pricing Guide",
            "meta_description": "A practical pricing guide.",
            "slug": "pricing-guide",
            "primary_keyword": "pricing guide",
        },
        "conversion_plan": {
            "primary_intent": "mofu",
            "cta_strategy": [],
        },
    }


def test_allows_document_without_source_dependent_claims() -> None:
    payload = _base_payload()
    payload["blocks"] = [
        {
            "block_type": "hero",
            "semantic_tag": "header",
            "heading": "Pricing Guide",
            "body": "Learn how to evaluate plans.",
        },
        {
            "block_type": "summary",
            "semantic_tag": "section",
            "heading": "Summary",
            "body": "Use this guide to compare options quickly.",
        },
        {
            "block_type": "section",
            "semantic_tag": "section",
            "heading": "How to compare plans",
            "body": "Review scope, support, and implementation details.",
        },
    ]

    result = ArticleDocumentResult.model_validate(payload)

    assert result.schema_version == "1.0"


def test_rejects_source_dependent_claims_without_sources_block() -> None:
    payload = _base_payload()
    payload["blocks"] = [
        {
            "block_type": "hero",
            "semantic_tag": "header",
            "heading": "Pricing Guide",
            "body": "Plan comparison overview.",
        },
        {
            "block_type": "summary",
            "semantic_tag": "section",
            "heading": "Summary",
            "body": "This guide compares pricing models.",
        },
        {
            "block_type": "section",
            "semantic_tag": "section",
            "heading": "Typical SaaS pricing",
            "body": "Most teams pay $49/month and see a 25% lower ticket backlog.",
        },
    ]

    with pytest.raises(ValidationError):
        ArticleDocumentResult.model_validate(payload)


def test_rejects_empty_sources_block_for_source_dependent_claims() -> None:
    payload = _base_payload()
    payload["blocks"] = [
        {
            "block_type": "hero",
            "semantic_tag": "header",
            "heading": "Pricing Guide",
            "body": "Plan comparison overview.",
        },
        {
            "block_type": "summary",
            "semantic_tag": "section",
            "heading": "Summary",
            "body": "This guide compares pricing models.",
        },
        {
            "block_type": "section",
            "semantic_tag": "section",
            "heading": "Typical SaaS pricing",
            "body": "Most teams pay $49/month and see a 25% lower ticket backlog.",
        },
        {
            "block_type": "sources",
            "semantic_tag": "section",
            "heading": "Sources",
            "body": "",
            "items": [],
            "links": [],
        },
    ]

    with pytest.raises(ValidationError):
        ArticleDocumentResult.model_validate(payload)


def test_allows_source_dependent_claims_with_non_empty_sources_block() -> None:
    payload = _base_payload()
    payload["blocks"] = [
        {
            "block_type": "hero",
            "semantic_tag": "header",
            "heading": "Pricing Guide",
            "body": "Plan comparison overview.",
        },
        {
            "block_type": "summary",
            "semantic_tag": "section",
            "heading": "Summary",
            "body": "This guide compares pricing models.",
        },
        {
            "block_type": "section",
            "semantic_tag": "section",
            "heading": "Typical SaaS pricing",
            "body": "Most teams pay $49/month and see a 25% lower ticket backlog.",
        },
        {
            "block_type": "sources",
            "semantic_tag": "section",
            "heading": "Sources",
            "links": [
                {
                    "anchor": "Vendor pricing page",
                    "href": "https://example.com/pricing",
                }
            ],
        },
    ]

    result = ArticleDocumentResult.model_validate(payload)

    assert any(block.block_type == "sources" for block in result.blocks)
