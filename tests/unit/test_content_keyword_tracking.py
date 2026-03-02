"""Tests for content keyword traceability and coverage scoring helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import content_keyword_tracking as tracking


class _BriefKeywordRow:
    def __init__(
        self,
        *,
        row_id: str,
        keyword_role: str,
        keyword_text: str,
        keyword_text_normalized: str,
        position: int,
        keyword_id: str | None = None,
    ) -> None:
        self.id = row_id
        self.keyword_role = keyword_role
        self.keyword_text = keyword_text
        self.keyword_text_normalized = keyword_text_normalized
        self.position = position
        self.keyword_id = keyword_id
        self.patch_calls: list[dict[str, object]] = []
        self.delete_calls = 0

    def patch(self, _session: object, dto: object) -> "_BriefKeywordRow":
        payload = dto.to_patch_dict()
        self.patch_calls.append(payload)
        for key, value in payload.items():
            setattr(self, key, value)
        return self

    async def delete(self, _session: object) -> None:
        self.delete_calls += 1


class _SyncSession:
    def __init__(self) -> None:
        self.flush_calls = 0

    async def flush(self) -> None:
        self.flush_calls += 1


@pytest.mark.asyncio
async def test_sync_brief_keywords_is_idempotent_when_targets_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief = SimpleNamespace(
        id="brief-1",
        project_id="project-1",
        topic_id="topic-1",
        primary_keyword="helpdesk automation software",
        supporting_keywords=["ticket routing"],
    )
    existing_rows = [
        _BriefKeywordRow(
            row_id="bk-1",
            keyword_role="primary",
            keyword_text="helpdesk automation software",
            keyword_text_normalized="helpdesk automation software",
            position=0,
        ),
        _BriefKeywordRow(
            row_id="bk-2",
            keyword_role="supporting",
            keyword_text="ticket routing",
            keyword_text_normalized="ticket routing",
            position=0,
        ),
    ]

    async def _load_keywords(_session: object, _brief_id: str) -> list[_BriefKeywordRow]:
        return existing_rows

    async def _resolve_maps(
        _session: object,
        *,
        brief: object,
    ) -> tuple[dict[str, str], dict[str, str]]:
        del brief
        return {}, {}

    monkeypatch.setattr(tracking, "_load_brief_keywords", _load_keywords)
    monkeypatch.setattr(tracking, "_resolve_keyword_lookup_maps", _resolve_maps)
    monkeypatch.setattr(
        tracking.ContentBriefKeyword,
        "create",
        lambda *_args, **_kwargs: pytest.fail("create should not be called"),
    )

    session = _SyncSession()
    rows = await tracking.sync_brief_keywords(session, brief=brief)

    assert rows == existing_rows
    assert session.flush_calls == 1
    assert all(len(row.patch_calls) == 0 for row in existing_rows)
    assert all(row.delete_calls == 0 for row in existing_rows)


@pytest.mark.asyncio
async def test_sync_brief_keywords_replaces_stale_supporting_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief = SimpleNamespace(
        id="brief-1",
        project_id="project-1",
        topic_id="topic-1",
        primary_keyword="helpdesk automation software",
        supporting_keywords=["ticket routing"],
    )
    primary_row = _BriefKeywordRow(
        row_id="bk-1",
        keyword_role="primary",
        keyword_text="helpdesk automation software",
        keyword_text_normalized="helpdesk automation software",
        position=0,
    )
    stale_supporting_row = _BriefKeywordRow(
        row_id="bk-2",
        keyword_role="supporting",
        keyword_text="legacy keyword",
        keyword_text_normalized="legacy keyword",
        position=0,
    )
    current_rows = [primary_row, stale_supporting_row]
    created_rows: list[_BriefKeywordRow] = []

    async def _load_keywords(_session: object, _brief_id: str) -> list[_BriefKeywordRow]:
        return [*([row for row in current_rows if row.delete_calls == 0]), *created_rows]

    async def _resolve_maps(
        _session: object,
        *,
        brief: object,
    ) -> tuple[dict[str, str], dict[str, str]]:
        del brief
        return {}, {}

    def _create_row(_session: object, dto: object) -> _BriefKeywordRow:
        row = _BriefKeywordRow(
            row_id="bk-new",
            keyword_role=dto.keyword_role,  # type: ignore[attr-defined]
            keyword_text=dto.keyword_text,  # type: ignore[attr-defined]
            keyword_text_normalized=dto.keyword_text_normalized,  # type: ignore[attr-defined]
            position=dto.position,  # type: ignore[attr-defined]
            keyword_id=dto.keyword_id,  # type: ignore[attr-defined]
        )
        created_rows.append(row)
        return row

    monkeypatch.setattr(tracking, "_load_brief_keywords", _load_keywords)
    monkeypatch.setattr(tracking, "_resolve_keyword_lookup_maps", _resolve_maps)
    monkeypatch.setattr(tracking.ContentBriefKeyword, "create", _create_row)

    session = _SyncSession()
    rows = await tracking.sync_brief_keywords(
        session,
        brief=brief,
        supporting_keywords=["ticket routing"],
    )

    assert session.flush_calls == 1
    assert stale_supporting_row.delete_calls == 1
    assert len(created_rows) == 1
    assert created_rows[0].keyword_text == "ticket routing"
    assert any(row.keyword_text == "ticket routing" for row in rows)


@pytest.mark.asyncio
async def test_analyze_keyword_usage_reports_signals_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief = SimpleNamespace(
        id="brief-1",
        project_id="project-1",
        topic_id="topic-1",
        search_intent="informational",
        primary_keyword="helpdesk automation software",
        supporting_keywords=[],
    )
    brief_keywords = [
        SimpleNamespace(
            id="bk-1",
            keyword_id=None,
            keyword_text="helpdesk automation software",
            keyword_text_normalized="helpdesk automation software",
            keyword_role="primary",
            position=0,
        ),
        SimpleNamespace(
            id="bk-2",
            keyword_id=None,
            keyword_text="ticket routing",
            keyword_text_normalized="ticket routing",
            keyword_role="supporting",
            position=1,
        ),
    ]
    async def _load_keywords(_session: object, _brief_id: str) -> list[SimpleNamespace]:
        return brief_keywords

    monkeypatch.setattr(tracking, "_load_brief_keywords", _load_keywords)

    document = {
        "seo_meta": {
            "h1": "Helpdesk Automation Software Guide",
        },
        "blocks": [
            {
                "block_type": "hero",
                "heading": "Helpdesk Automation Software Guide",
                "body": (
                    "Helpdesk automation software helps teams reduce manual work. "
                    "This guide covers setup and outcomes."
                ),
            },
            {
                "block_type": "section",
                "level": 2,
                "heading": "Ticket Routing Best Practices",
                "body": "Ticket routing improves response quality when ownership is clear.",
            },
        ],
    }

    report, usages = await tracking.analyze_keyword_usage(
        session=SimpleNamespace(),
        brief=brief,
        document=document,
        article_version_number=1,
    )

    assert report["framework_version"] == tracking.KEYWORD_COVERAGE_FRAMEWORK_VERSION
    assert report["article_version"] == 1
    assert report["summary"]["suggested_keywords_total"] == 2
    assert report["summary"]["used_keywords_total"] == 2
    assert report["summary"]["primary_keyword_used"] is True
    assert len(usages) == 2
    assert usages[0].in_h1 is True
    assert usages[0].in_first_150_words is True
    assert usages[0].seo_incorporation_score > 0
    assert usages[1].in_h2_h3 is True
    assert usages[1].seo_incorporation_score > 0


@pytest.mark.asyncio
async def test_analyze_keyword_usage_avoids_partial_token_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief = SimpleNamespace(
        id="brief-2",
        project_id="project-1",
        topic_id="topic-1",
        search_intent="informational",
        primary_keyword="cat",
        supporting_keywords=[],
    )
    brief_keywords = [
        SimpleNamespace(
            id="bk-3",
            keyword_id=None,
            keyword_text="cat",
            keyword_text_normalized="cat",
            keyword_role="primary",
            position=0,
        )
    ]
    async def _load_keywords(_session: object, _brief_id: str) -> list[SimpleNamespace]:
        return brief_keywords

    monkeypatch.setattr(tracking, "_load_brief_keywords", _load_keywords)

    document = {
        "seo_meta": {"h1": "Catalog Operations"},
        "blocks": [
            {
                "block_type": "hero",
                "heading": "Catalog Operations",
                "body": "Concatenate values in large catalogs to speed reconciliation.",
            }
        ],
    }

    report, usages = await tracking.analyze_keyword_usage(
        session=SimpleNamespace(),
        brief=brief,
        document=document,
        article_version_number=1,
    )

    assert report["summary"]["used_keywords_total"] == 0
    assert usages[0].used is False
    assert usages[0].usage_count == 0
    assert usages[0].seo_incorporation_score == 0


def test_with_keyword_coverage_report_merges_into_qa_report() -> None:
    payload = tracking.with_keyword_coverage_report(
        {"passed": True},
        {"framework_version": "keyword-coverage-v1"},
    )

    assert payload["passed"] is True
    assert payload["keyword_coverage"]["framework_version"] == "keyword-coverage-v1"
