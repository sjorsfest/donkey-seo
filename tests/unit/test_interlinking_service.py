"""Tests for publication-order aware interlinking helpers."""

from datetime import date
from types import SimpleNamespace

import pytest

from app.agents.sitemap_link_selector import SitemapLinkDecision, SitemapLinkSelectorOutput
from app.services.interlinking_service import InterlinkingService


def _brief(
    *,
    brief_id: str,
    primary_keyword: str,
    proposed_publication_date: date | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=brief_id,
        primary_keyword=primary_keyword,
        working_titles=[],
        proposed_publication_date=proposed_publication_date,
    )


def test_is_valid_publication_predecessor_requires_earlier_date() -> None:
    service = InterlinkingService(
        session=SimpleNamespace(),
        embeddings_client=SimpleNamespace(),
    )
    source = _brief(
        brief_id="b2",
        primary_keyword="Blog 2",
        proposed_publication_date=date(2026, 9, 3),
    )
    target_before = _brief(
        brief_id="b1",
        primary_keyword="Blog 1",
        proposed_publication_date=date(2026, 9, 1),
    )
    target_after = _brief(
        brief_id="b3",
        primary_keyword="Blog 3",
        proposed_publication_date=date(2026, 9, 5),
    )

    assert service._is_valid_publication_predecessor(source, target_before) is True
    assert service._is_valid_publication_predecessor(source, target_after) is False


@pytest.mark.asyncio
async def test_sitemap_selector_agent_order_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.interlinking_service.settings.openrouter_api_key", "test-key")

    class _FakeSelectorAgent:
        async def run(self, _input: object) -> SitemapLinkSelectorOutput:
            return SitemapLinkSelectorOutput(
                selected_urls=[
                    SitemapLinkDecision(url="https://example.com/c", rationale="best fit"),
                    SitemapLinkDecision(url="https://example.com/a", rationale="second fit"),
                ],
                notes="ranked by contextual fit",
            )

    monkeypatch.setattr(
        "app.services.interlinking_service.SitemapLinkSelectorAgent",
        lambda: _FakeSelectorAgent(),
    )

    service = InterlinkingService(
        session=SimpleNamespace(),
        embeddings_client=SimpleNamespace(),
        use_llm_sitemap_selector=True,
        max_sitemap_links=2,
    )

    brief = SimpleNamespace(
        id="brief-1",
        primary_keyword="internal linking strategy",
        working_titles=["Internal linking playbook"],
        supporting_keywords=["seo", "site architecture"],
        target_audience="SEO manager",
        reader_job_to_be_done="Pick relevant internal links",
    )
    topic = SimpleNamespace(
        name="Internal linking",
        dominant_intent="informational",
        funnel_stage="mofu",
    )
    candidates = [
        SimpleNamespace(
            target_type="sitemap_page",
            target_url="https://example.com/a",
            target_brief_id=None,
            anchor_text="learn about a",
            placement_section="middle",
            relevance_score=0.91,
            intent_alignment="assumed_complementary",
            funnel_relationship="assumed_progression",
        ),
        SimpleNamespace(
            target_type="sitemap_page",
            target_url="https://example.com/b",
            target_brief_id=None,
            anchor_text="learn about b",
            placement_section="middle",
            relevance_score=0.89,
            intent_alignment="assumed_complementary",
            funnel_relationship="assumed_progression",
        ),
        SimpleNamespace(
            target_type="sitemap_page",
            target_url="https://example.com/c",
            target_brief_id=None,
            anchor_text="learn about c",
            placement_section="middle",
            relevance_score=0.84,
            intent_alignment="assumed_complementary",
            funnel_relationship="assumed_progression",
        ),
    ]

    selected = await service._select_sitemap_links_with_agent(  # type: ignore[arg-type]
        brief=brief,
        topic=topic,
        candidates=candidates,
    )

    assert [candidate.target_url for candidate in selected] == [
        "https://example.com/c",
        "https://example.com/a",
    ]


@pytest.mark.asyncio
async def test_sitemap_selector_falls_back_to_heuristics_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.interlinking_service.settings.openrouter_api_key", "test-key")

    class _FailingSelectorAgent:
        async def run(self, _input: object) -> SitemapLinkSelectorOutput:
            raise RuntimeError("selector unavailable")

    monkeypatch.setattr(
        "app.services.interlinking_service.SitemapLinkSelectorAgent",
        lambda: _FailingSelectorAgent(),
    )

    service = InterlinkingService(
        session=SimpleNamespace(),
        embeddings_client=SimpleNamespace(),
        use_llm_sitemap_selector=True,
        max_sitemap_links=2,
    )

    brief = SimpleNamespace(
        id="brief-1",
        primary_keyword="internal linking strategy",
        working_titles=[],
        supporting_keywords=[],
        target_audience="",
        reader_job_to_be_done="",
    )
    topic = SimpleNamespace(name="Internal linking", dominant_intent=None, funnel_stage=None)
    candidates = [
        SimpleNamespace(
            target_type="sitemap_page",
            target_url="https://example.com/high",
            target_brief_id=None,
            anchor_text="high",
            placement_section="middle",
            relevance_score=0.95,
            intent_alignment="assumed_complementary",
            funnel_relationship="assumed_progression",
        ),
        SimpleNamespace(
            target_type="sitemap_page",
            target_url="https://example.com/mid",
            target_brief_id=None,
            anchor_text="mid",
            placement_section="middle",
            relevance_score=0.85,
            intent_alignment="assumed_complementary",
            funnel_relationship="assumed_progression",
        ),
        SimpleNamespace(
            target_type="sitemap_page",
            target_url="https://example.com/low",
            target_brief_id=None,
            anchor_text="low",
            placement_section="middle",
            relevance_score=0.75,
            intent_alignment="assumed_complementary",
            funnel_relationship="assumed_progression",
        ),
    ]

    selected = await service._select_sitemap_links_with_agent(  # type: ignore[arg-type]
        brief=brief,
        topic=topic,
        candidates=candidates,
    )

    assert [candidate.target_url for candidate in selected] == [
        "https://example.com/high",
        "https://example.com/mid",
    ]
