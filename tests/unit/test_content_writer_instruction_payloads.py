"""Tests for writer instruction payload wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from app.api.v1.content.routes import _writer_instructions_payload
from app.services.steps.content.step_14_article_writer import Step14ArticleWriterService


class _RowsResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> list[object]:
        return self._rows


class _FakeSession:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    async def execute(self, _query: object) -> _RowsResult:
        return _RowsResult(self._rows)


def test_routes_writer_payload_includes_pass_fail_thresholds() -> None:
    instructions = SimpleNamespace(
        voice_tone_constraints={},
        forbidden_claims=[],
        compliance_notes=[],
        formatting_requirements={},
        h1_h2_usage={},
        internal_linking_minimums={},
        schema_guidance="",
        qa_checklist=[],
        pass_fail_thresholds={"seo_score_target": 80},
        common_failure_modes=[],
    )

    payload = _writer_instructions_payload(instructions)

    assert payload["pass_fail_thresholds"] == {"seo_score_target": 80}


@pytest.mark.asyncio
async def test_step14_load_writer_instructions_includes_pass_fail_thresholds() -> None:
    session = _FakeSession(
        [
            SimpleNamespace(
                brief_id="brief-1",
                voice_tone_constraints={},
                forbidden_claims=[],
                compliance_notes=[],
                formatting_requirements={},
                h1_h2_usage={},
                internal_linking_minimums={},
                schema_guidance="",
                qa_checklist=[],
                pass_fail_thresholds={"seo_score_target": 70},
                common_failure_modes=[],
            )
        ]
    )
    service = Step14ArticleWriterService(
        session=session,
        project_id="project-1",
        execution=SimpleNamespace(),
    )

    payload = await service._load_writer_instructions([SimpleNamespace(id="brief-1")])

    assert payload["brief-1"]["pass_fail_thresholds"] == {"seo_score_target": 70}


def test_step14_duplicate_article_error_detection() -> None:
    duplicate_error = IntegrityError(
        statement="INSERT INTO content_articles ...",
        params={},
        orig=Exception(
            'duplicate key value violates unique constraint "ix_content_articles_brief_id"'
        ),
    )
    non_duplicate_error = IntegrityError(
        statement="INSERT INTO content_article_versions ...",
        params={},
        orig=Exception(
            'duplicate key value violates unique constraint "uq_content_article_versions_article_version"'
        ),
    )

    assert Step14ArticleWriterService._is_duplicate_article_for_brief_error(duplicate_error)
    assert not Step14ArticleWriterService._is_duplicate_article_for_brief_error(non_duplicate_error)
