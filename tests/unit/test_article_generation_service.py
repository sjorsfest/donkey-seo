"""Tests for article generation normalization and defaults."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


class _FakeWriterAgent:
    documents: list[dict] = []
    instances: list["_FakeWriterAgent"] = []

    def __init__(self) -> None:
        self._model = "fake-writer-model"
        self.temperature = 0.0
        self.calls: list[object] = []
        _FakeWriterAgent.instances.append(self)

    async def run(self, input_data: object) -> object:
        self.calls.append(input_data)
        index = min(len(self.calls) - 1, max(len(self.documents) - 1, 0))
        payload = self.documents[index] if self.documents else {}
        return SimpleNamespace(document=SimpleNamespace(model_dump=lambda: payload))


class _FakeAuditorAgent:
    outputs: list[dict] = []
    instances: list["_FakeAuditorAgent"] = []

    def __init__(self) -> None:
        self.calls: list[object] = []
        _FakeAuditorAgent.instances.append(self)

    async def run(self, input_data: object) -> object:
        self.calls.append(input_data)
        index = min(len(self.calls) - 1, max(len(self.outputs) - 1, 0))
        payload = self.outputs[index] if self.outputs else {}
        return SimpleNamespace(model_dump=lambda: payload)


class _FakeHTTPResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeLinkClient:
    def __init__(self, status_by_method_and_url: dict[tuple[str, str], int]) -> None:
        self.status_by_method_and_url = status_by_method_and_url
        self.calls: list[tuple[str, str]] = []

    async def head(self, url: str) -> _FakeHTTPResponse:
        self.calls.append(("HEAD", url))
        status_code = self.status_by_method_and_url.get(("HEAD", url), 404)
        return _FakeHTTPResponse(status_code)

    async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeHTTPResponse:
        del headers
        self.calls.append(("GET", url))
        status_code = self.status_by_method_and_url.get(("GET", url), 404)
        return _FakeHTTPResponse(status_code)


def _brief_payload() -> dict:
    return {
        "primary_keyword": "helpdesk automation software",
        "search_intent": "informational",
        "page_type": "guide",
        "funnel_stage": "tofu",
        "working_titles": ["Helpdesk Automation Software Guide"],
        "target_audience": "Support leaders",
        "reader_job_to_be_done": "Evaluate tooling",
        "must_include_sections": ["Overview"],
        "target_word_count_min": 20,
        "target_word_count_max": 2000,
        "money_page_links": [],
    }


def _writer_instructions_payload() -> dict:
    return {
        "forbidden_claims": [],
        "compliance_notes": [],
        "internal_linking_minimums": {"min_internal": 0, "min_external": 0},
        "pass_fail_thresholds": {
            "seo_score_target": 75,
            "keyword_density_soft_min": 0.2,
            "keyword_density_soft_max": 2.5,
            "max_auto_revisions": 1,
        },
    }


def _delta_payload() -> dict:
    return {"must_include_sections": ["Overview"]}


@pytest.mark.asyncio
async def test_filter_invalid_document_links_drops_broken_links() -> None:
    service = ArticleGenerationService("donkey.support")
    document = {
        "schema_version": "1.0",
        "seo_meta": {},
        "conversion_plan": {},
        "blocks": [
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Links",
                "links": [
                    {"anchor": "Valid", "href": "/docs"},
                    {"anchor": "Invalid", "href": "/demo"},
                ],
                "cta": {"label": "Try it", "href": "https://donkey.support/pricing"},
            }
        ],
    }
    client = _FakeLinkClient(
        {
            ("HEAD", "https://donkey.support/docs"): 200,
            ("HEAD", "https://donkey.support/demo"): 404,
            ("HEAD", "https://donkey.support/pricing"): 404,
        }
    )

    filtered = await service._filter_invalid_document_links(
        document=document,
        link_cache={},
        client=client,
    )
    block = filtered["blocks"][0]

    assert block["links"] == [{"anchor": "Valid", "href": "/docs"}]
    assert block["cta"]["href"] == "#"


@pytest.mark.asyncio
async def test_filter_invalid_document_links_keeps_deferred_internal_links() -> None:
    service = ArticleGenerationService("donkey.support")
    document = {
        "schema_version": "1.0",
        "seo_meta": {},
        "conversion_plan": {},
        "blocks": [
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Links",
                "links": [
                    {
                        "anchor": "Upcoming article",
                        "href": "",
                        "target_brief_id": "brief-1",
                    }
                ],
            }
        ],
    }
    client = _FakeLinkClient({})

    filtered = await service._filter_invalid_document_links(
        document=document,
        link_cache={},
        client=client,
    )

    assert len(filtered["blocks"][0]["links"]) == 1
    assert filtered["blocks"][0]["links"][0]["target_brief_id"] == "brief-1"
    assert filtered["blocks"][0]["links"][0]["href"] == ""
    assert client.calls == []


@pytest.mark.asyncio
async def test_filter_invalid_document_links_uses_head_get_fallback_and_cache() -> None:
    service = ArticleGenerationService("donkey.support")
    document = {
        "schema_version": "1.0",
        "seo_meta": {},
        "conversion_plan": {},
        "blocks": [
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Links",
                "links": [
                    {"anchor": "One", "href": "/head-get-fallback"},
                    {"anchor": "Two", "href": "/head-get-fallback"},
                ],
                "cta": None,
            }
        ],
    }
    target_url = "https://donkey.support/head-get-fallback"
    client = _FakeLinkClient(
        {
            ("HEAD", target_url): 405,
            ("GET", target_url): 200,
        }
    )

    filtered = await service._filter_invalid_document_links(
        document=document,
        link_cache={},
        client=client,
    )

    assert len(filtered["blocks"][0]["links"]) == 2
    assert client.calls.count(("HEAD", target_url)) == 1
    assert client.calls.count(("GET", target_url)) == 1


def test_apply_brief_internal_link_plan_injects_batch_brief_link() -> None:
    service = ArticleGenerationService("donkey.support")
    document = {
        "schema_version": "1.0",
        "seo_meta": {},
        "conversion_plan": {},
        "blocks": [
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Overview",
                "links": [],
            }
        ],
    }
    brief = {
        "internal_links_out": [
            {
                "target_type": "batch_brief",
                "target_brief_id": "brief-123",
                "target_url": "/blog-1",
                "anchor_text": "Read Blog 1",
                "placement_section": "Overview",
            }
        ]
    }

    updated = service._apply_brief_internal_link_plan(document=document, brief=brief)

    links = updated["blocks"][0]["links"]
    assert len(links) == 1
    assert links[0]["href"] == "/blog-1"
    assert links[0]["anchor"] == "Read Blog 1"
    assert links[0]["target_brief_id"] == "brief-123"


@pytest.mark.asyncio
async def test_generate_with_repair_revises_once_and_returns_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeWriterAgent.instances = []
    _FakeAuditorAgent.instances = []
    _FakeWriterAgent.documents = [
        {
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
                    "body": " ".join(["intro"] * 200),
                },
                {
                    "block_type": "section",
                    "semantic_tag": "section",
                    "heading": "Overview",
                    "level": 2,
                    "body": "Core context for teams.",
                },
            ],
        },
        {
            "schema_version": "1.0",
            "seo_meta": {
                "h1": "Helpdesk Automation Software Guide",
                "meta_title": "Helpdesk Automation Software Guide",
                "meta_description": "Guide",
                "slug": "helpdesk-automation-software-guide",
                "primary_keyword": "helpdesk automation software",
            },
            "conversion_plan": {"primary_intent": "informational", "cta_strategy": []},
            "blocks": [
                {
                    "block_type": "hero",
                    "semantic_tag": "header",
                    "heading": "Helpdesk Automation Software Guide",
                    "body": " ".join(["helpdesk automation software"] + ["context"] * 180),
                },
                {
                    "block_type": "section",
                    "semantic_tag": "section",
                    "heading": "Helpdesk Automation Software Overview",
                    "level": 2,
                    "body": " ".join(["details"] * 100),
                },
            ],
        },
    ]
    _FakeAuditorAgent.outputs = [
        {
            "checklist_items": [],
            "claim_integrity": [],
            "overall_score": 80,
            "hard_failures": [],
            "soft_warnings": [],
            "revision_instructions": [],
        },
        {
            "checklist_items": [],
            "claim_integrity": [],
            "overall_score": 92,
            "hard_failures": [],
            "soft_warnings": [],
            "revision_instructions": [],
        },
    ]
    monkeypatch.setattr("app.services.article_generation.ArticleWriterAgent", _FakeWriterAgent)
    monkeypatch.setattr("app.services.article_generation.ArticleSEOAuditorAgent", _FakeAuditorAgent)

    service = ArticleGenerationService("donkey.support")
    artifact = await service.generate_with_repair(
        brief=_brief_payload(),
        writer_instructions=_writer_instructions_payload(),
        brief_delta=_delta_payload(),
        brand_context="",
        conversion_intents=[],
    )

    assert artifact.status == "draft"
    assert artifact.qa_report["seo_audit"]["overall_score"] >= 0
    assert artifact.qa_report["seo_audit"]["content_type_module"] == "A"
    writer_instance = _FakeWriterAgent.instances[0]
    assert len(writer_instance.calls) == 2
    assert writer_instance.calls[1].existing_document is not None


@pytest.mark.asyncio
async def test_generate_with_repair_soft_warnings_still_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeWriterAgent.instances = []
    _FakeAuditorAgent.instances = []
    _FakeWriterAgent.documents = [
        {
            "schema_version": "1.0",
            "seo_meta": {
                "h1": "Helpdesk Automation Software Guide",
                "meta_title": "Helpdesk Automation Software Guide",
                "meta_description": "Guide",
                "slug": "helpdesk-automation-software-guide",
                "primary_keyword": "helpdesk automation software",
            },
            "conversion_plan": {"primary_intent": "informational", "cta_strategy": []},
            "blocks": [
                {
                    "block_type": "hero",
                    "semantic_tag": "header",
                    "heading": "Helpdesk Automation Software Guide",
                    "body": "helpdesk automation software for support teams.",
                },
                {
                    "block_type": "section",
                    "semantic_tag": "section",
                    "heading": "Helpdesk Automation Software Overview",
                    "level": 2,
                    "body": "Short section for context.",
                },
            ],
        }
    ]
    _FakeAuditorAgent.outputs = [
        {
            "checklist_items": [],
            "claim_integrity": [],
            "overall_score": 88,
            "hard_failures": [],
            "soft_warnings": ["style_tightening"],
            "revision_instructions": [],
        }
    ]
    monkeypatch.setattr("app.services.article_generation.ArticleWriterAgent", _FakeWriterAgent)
    monkeypatch.setattr("app.services.article_generation.ArticleSEOAuditorAgent", _FakeAuditorAgent)

    service = ArticleGenerationService("donkey.support")
    artifact = await service.generate_with_repair(
        brief=_brief_payload(),
        writer_instructions=_writer_instructions_payload(),
        brief_delta=_delta_payload(),
        brand_context="",
        conversion_intents=[],
    )

    assert artifact.status == "draft"
    assert artifact.qa_report["seo_audit"]["soft_warnings"]


@pytest.mark.asyncio
async def test_generate_with_repair_persistent_hard_failures_need_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeWriterAgent.instances = []
    _FakeAuditorAgent.instances = []
    failing_document = {
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
                "body": " ".join(["intro"] * 200),
            },
            {
                "block_type": "section",
                "semantic_tag": "section",
                "heading": "Overview",
                "level": 2,
                "body": "Context for teams.",
            },
        ],
    }
    _FakeWriterAgent.documents = [failing_document]
    _FakeAuditorAgent.outputs = [
        {
            "checklist_items": [],
            "claim_integrity": [],
            "overall_score": 82,
            "hard_failures": [],
            "soft_warnings": [],
            "revision_instructions": [],
        }
    ]
    monkeypatch.setattr("app.services.article_generation.ArticleWriterAgent", _FakeWriterAgent)
    monkeypatch.setattr("app.services.article_generation.ArticleSEOAuditorAgent", _FakeAuditorAgent)

    service = ArticleGenerationService("donkey.support")
    artifact = await service.generate_with_repair(
        brief=_brief_payload(),
        writer_instructions=_writer_instructions_payload(),
        brief_delta=_delta_payload(),
        brand_context="",
        conversion_intents=[],
    )

    assert artifact.status == "needs_review"
    assert artifact.qa_report["seo_audit"]["hard_failures"]
    writer_instance = _FakeWriterAgent.instances[0]
    assert len(writer_instance.calls) == 2
