"""Tests for publication-order aware interlinking helpers."""

from datetime import date
from types import SimpleNamespace

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
