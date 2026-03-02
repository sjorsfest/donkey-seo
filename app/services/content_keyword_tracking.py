"""Keyword traceability and usage scoring helpers for content artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentArticleKeywordUsage, ContentBrief, ContentBriefKeyword
from app.models.generated_dtos import (
    ContentArticleKeywordUsageCreateDTO,
    ContentBriefKeywordCreateDTO,
    ContentBriefKeywordPatchDTO,
)
from app.models.keyword import Keyword

KEYWORD_COVERAGE_FRAMEWORK_VERSION = "keyword-coverage-v1"
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")


@dataclass(slots=True)
class KeywordUsageComputation:
    """Computed keyword usage and scoring signals for one brief keyword target."""

    brief_keyword_id: str | None
    keyword_id: str | None
    keyword_text: str
    keyword_role: str
    keyword_intent: str | None
    search_volume: int | None
    adjusted_volume: int | None
    used: bool
    usage_count: int
    usage_density_pct: float
    in_h1: bool
    in_first_150_words: bool
    in_h2_h3: bool
    section_hits: int
    seo_incorporation_score: int


def _clean_keyword_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_keyword_text(value: str) -> str:
    tokens = _WORD_PATTERN.findall(str(value or "").lower())
    return " ".join(tokens).strip()


def _keyword_pattern(keyword_text: str) -> re.Pattern[str] | None:
    tokens = _WORD_PATTERN.findall(keyword_text.lower())
    if not tokens:
        return None
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in tokens) + r"\b"
    return re.compile(pattern)


def _extract_section_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    heading = block.get("heading")
    body = block.get("body")
    if isinstance(heading, str):
        parts.append(heading)
    if isinstance(body, str):
        parts.append(body)

    for item in block.get("items") or []:
        parts.append(str(item))
    for item in block.get("table_columns") or []:
        parts.append(str(item))
    for row in block.get("table_rows") or []:
        if isinstance(row, list):
            for cell in row:
                parts.append(str(cell))
    for faq_item in block.get("faq_items") or []:
        if not isinstance(faq_item, dict):
            continue
        question = faq_item.get("question")
        answer = faq_item.get("answer")
        if isinstance(question, str):
            parts.append(question)
        if isinstance(answer, str):
            parts.append(answer)
    return " ".join(parts).strip()


def _collect_document_usage_fields(
    document: dict[str, Any],
) -> tuple[str, str, str, list[str], list[str], int]:
    seo_meta = document.get("seo_meta")
    seo_meta = seo_meta if isinstance(seo_meta, dict) else {}
    h1_text = str(seo_meta.get("h1") or "")

    parts: list[str] = []
    if h1_text:
        parts.append(h1_text)

    h2_h3_headings: list[str] = []
    section_texts: list[str] = []
    blocks = document.get("blocks")
    blocks = blocks if isinstance(blocks, list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        section_text = _extract_section_text(block)
        if section_text:
            section_texts.append(section_text.lower())
            parts.append(section_text)
        heading = block.get("heading")
        if not isinstance(heading, str):
            continue
        block_type = str(block.get("block_type") or "").strip().lower()
        level = block.get("level")
        heading_level: int | None = None
        if block_type == "hero":
            heading_level = 1
        elif isinstance(level, int):
            heading_level = max(2, min(4, int(level)))
        else:
            heading_level = 2
        if heading_level in {2, 3}:
            h2_h3_headings.append(heading.lower())

    full_text = "\n".join(parts)
    words = _WORD_PATTERN.findall(full_text.lower())
    first_150_words = " ".join(words[:150])
    return full_text.lower(), h1_text.lower(), first_150_words, h2_h3_headings, section_texts, len(words)


def _score_keyword_incorporation(
    *,
    keyword_role: str,
    used: bool,
    in_h1: bool,
    in_first_150_words: bool,
    in_h2_h3: bool,
    section_hits: int,
    usage_density_pct: float,
) -> int:
    if not used:
        return 0

    if keyword_role == "primary":
        score = 30  # Used
        if in_h1:
            score += 20
        if in_first_150_words:
            score += 20
        if in_h2_h3:
            score += 15
        if 0.2 <= usage_density_pct <= 2.5:
            score += 15
        return max(0, min(100, score))

    score = 40  # Used
    if in_h2_h3:
        score += 20
    if in_first_150_words:
        score += 10
    if section_hits >= 2:
        score += 15
    if 0.05 <= usage_density_pct <= 1.5:
        score += 15
    return max(0, min(100, score))


def with_keyword_coverage_report(
    qa_report: dict[str, Any] | None,
    keyword_coverage_report: dict[str, Any],
) -> dict[str, Any]:
    """Attach keyword coverage details to the canonical qa_report payload."""
    payload: dict[str, Any] = dict(qa_report) if isinstance(qa_report, dict) else {}
    payload["keyword_coverage"] = keyword_coverage_report
    return payload


async def _load_brief_keywords(session: AsyncSession, brief_id: str) -> list[ContentBriefKeyword]:
    result = await session.execute(
        select(ContentBriefKeyword).where(ContentBriefKeyword.brief_id == brief_id)
    )
    rows = list(result.scalars().all())
    return sorted(
        rows,
        key=lambda item: (
            0 if item.keyword_role == "primary" else 1,
            int(item.position or 0),
            item.keyword_text_normalized,
        ),
    )


async def _resolve_keyword_lookup_maps(
    session: AsyncSession,
    *,
    brief: ContentBrief,
) -> tuple[dict[str, str], dict[str, str]]:
    topic_keywords: dict[str, str] = {}
    if brief.topic_id:
        topic_result = await session.execute(
            select(Keyword).where(
                Keyword.topic_id == brief.topic_id,
            )
        )
        for keyword in topic_result.scalars().all():
            normalized = _normalize_keyword_text(str(keyword.keyword_normalized or keyword.keyword or ""))
            if normalized and normalized not in topic_keywords:
                topic_keywords[normalized] = str(keyword.id)

    project_keywords: dict[str, str] = {}
    project_result = await session.execute(
        select(Keyword).where(Keyword.project_id == brief.project_id)
    )
    for keyword in project_result.scalars().all():
        normalized = _normalize_keyword_text(str(keyword.keyword_normalized or keyword.keyword or ""))
        if normalized and normalized not in project_keywords:
            project_keywords[normalized] = str(keyword.id)

    return topic_keywords, project_keywords


async def sync_brief_keywords(
    session: AsyncSession,
    *,
    brief: ContentBrief,
    primary_keyword: str | None = None,
    supporting_keywords: list[str] | None = None,
    primary_keyword_id: str | None = None,
    supporting_keyword_ids: list[str] | None = None,
) -> list[ContentBriefKeyword]:
    """Sync relational brief keyword targets from brief keyword strings."""
    primary_source = primary_keyword if primary_keyword is not None else str(brief.primary_keyword or "")
    primary_text = _clean_keyword_text(primary_source)
    supporting_source = (
        supporting_keywords
        if supporting_keywords is not None
        else (brief.supporting_keywords or [])
    )
    supporting_texts = [_clean_keyword_text(item) for item in supporting_source]
    supporting_ids = supporting_keyword_ids or []

    desired: list[dict[str, Any]] = []
    if primary_text:
        desired.append(
            {
                "keyword_role": "primary",
                "keyword_text": primary_text,
                "keyword_text_normalized": _normalize_keyword_text(primary_text),
                "position": 0,
                "keyword_id": primary_keyword_id,
            }
        )

    for index, text in enumerate(supporting_texts):
        if not text:
            continue
        normalized = _normalize_keyword_text(text)
        if not normalized:
            continue
        if primary_text and normalized == _normalize_keyword_text(primary_text):
            continue
        explicit_keyword_id = supporting_ids[index] if index < len(supporting_ids) else None
        desired.append(
            {
                "keyword_role": "supporting",
                "keyword_text": text,
                "keyword_text_normalized": normalized,
                "position": index,
                "keyword_id": explicit_keyword_id,
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in desired:
        key = (str(item["keyword_role"]), str(item["keyword_text_normalized"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    desired = deduped

    topic_keywords, project_keywords = await _resolve_keyword_lookup_maps(session, brief=brief)
    existing_rows = await _load_brief_keywords(session, str(brief.id))
    existing_by_key = {
        (row.keyword_role, row.keyword_text_normalized): row
        for row in existing_rows
    }

    seen_existing_ids: set[str] = set()
    for item in desired:
        key = (str(item["keyword_role"]), str(item["keyword_text_normalized"]))
        existing = existing_by_key.get(key)
        resolved_keyword_id = item.get("keyword_id")
        if not resolved_keyword_id:
            normalized = str(item["keyword_text_normalized"])
            resolved_keyword_id = topic_keywords.get(normalized) or project_keywords.get(normalized)

        if existing is not None:
            seen_existing_ids.add(str(existing.id))
            patch_payload: dict[str, Any] = {}
            if existing.keyword_text != item["keyword_text"]:
                patch_payload["keyword_text"] = item["keyword_text"]
            if existing.position != item["position"]:
                patch_payload["position"] = item["position"]
            if str(existing.keyword_id or "") != str(resolved_keyword_id or ""):
                patch_payload["keyword_id"] = resolved_keyword_id
            if patch_payload:
                existing.patch(session, ContentBriefKeywordPatchDTO.from_partial(patch_payload))
            continue

        created = ContentBriefKeyword.create(
            session,
            ContentBriefKeywordCreateDTO(
                brief_id=str(brief.id),
                keyword_id=str(resolved_keyword_id) if resolved_keyword_id else None,
                keyword_text=str(item["keyword_text"]),
                keyword_text_normalized=str(item["keyword_text_normalized"]),
                keyword_role=str(item["keyword_role"]),
                position=int(item["position"]),
            ),
        )
        seen_existing_ids.add(str(created.id))

    for existing in existing_rows:
        if str(existing.id) in seen_existing_ids:
            continue
        await existing.delete(session)

    await session.flush()
    return await _load_brief_keywords(session, str(brief.id))


async def analyze_keyword_usage(
    session: AsyncSession,
    *,
    brief: ContentBrief,
    document: dict[str, Any],
    article_version_number: int,
) -> tuple[dict[str, Any], list[KeywordUsageComputation]]:
    """Compute keyword usage report + row payloads for one article version."""
    brief_keywords = await _load_brief_keywords(session, str(brief.id))
    if not brief_keywords:
        brief_keywords = await sync_brief_keywords(session, brief=brief)

    keyword_ids = [row.keyword_id for row in brief_keywords if row.keyword_id]
    keyword_map: dict[str, Keyword] = {}
    if keyword_ids:
        keyword_result = await session.execute(
            select(Keyword).where(Keyword.id.in_(keyword_ids))
        )
        keyword_map = {
            str(keyword.id): keyword
            for keyword in keyword_result.scalars().all()
        }

    content_text, h1_text, first_150_words, h2_h3_headings, section_texts, word_count = (
        _collect_document_usage_fields(document)
    )

    usage_rows: list[KeywordUsageComputation] = []
    intent_coverage: dict[str, dict[str, int]] = {}
    covered_search_volume_total = 0
    primary_keyword_used = False

    for brief_keyword in brief_keywords:
        pattern = _keyword_pattern(brief_keyword.keyword_text)
        usage_count = len(pattern.findall(content_text)) if pattern else 0
        used = usage_count > 0
        keyword_word_count = len(_WORD_PATTERN.findall(brief_keyword.keyword_text))
        usage_density_pct = 0.0
        if word_count > 0 and keyword_word_count > 0:
            usage_density_pct = (usage_count * keyword_word_count / word_count) * 100

        in_h1 = bool(pattern.search(h1_text)) if pattern else False
        in_first_150_words = bool(pattern.search(first_150_words)) if pattern else False
        in_h2_h3 = (
            any(pattern.search(heading) for heading in h2_h3_headings)
            if pattern
            else False
        )
        section_hits = (
            sum(1 for section_text in section_texts if pattern.search(section_text))
            if pattern
            else 0
        )

        keyword_model = keyword_map.get(str(brief_keyword.keyword_id))
        keyword_intent = (
            str(keyword_model.validated_intent or keyword_model.intent or brief.search_intent or "").strip()
            if keyword_model is not None
            else str(brief.search_intent or "").strip()
        )
        if not keyword_intent:
            keyword_intent = "unknown"

        search_volume = keyword_model.search_volume if keyword_model is not None else None
        adjusted_volume = keyword_model.adjusted_volume if keyword_model is not None else None

        score = _score_keyword_incorporation(
            keyword_role=brief_keyword.keyword_role,
            used=used,
            in_h1=in_h1,
            in_first_150_words=in_first_150_words,
            in_h2_h3=in_h2_h3,
            section_hits=section_hits,
            usage_density_pct=usage_density_pct,
        )

        usage = KeywordUsageComputation(
            brief_keyword_id=str(brief_keyword.id),
            keyword_id=str(brief_keyword.keyword_id) if brief_keyword.keyword_id else None,
            keyword_text=brief_keyword.keyword_text,
            keyword_role=brief_keyword.keyword_role,
            keyword_intent=keyword_intent,
            search_volume=search_volume,
            adjusted_volume=adjusted_volume,
            used=used,
            usage_count=usage_count,
            usage_density_pct=round(usage_density_pct, 3),
            in_h1=in_h1,
            in_first_150_words=in_first_150_words,
            in_h2_h3=in_h2_h3,
            section_hits=section_hits,
            seo_incorporation_score=score,
        )
        usage_rows.append(usage)

        intent_bucket = intent_coverage.setdefault(
            keyword_intent,
            {
                "suggested_keywords": 0,
                "used_keywords": 0,
                "covered_search_volume": 0,
            },
        )
        intent_bucket["suggested_keywords"] += 1
        if used:
            intent_bucket["used_keywords"] += 1
            covered_volume = (
                usage.search_volume
                if usage.search_volume is not None
                else (usage.adjusted_volume or 0)
            )
            intent_bucket["covered_search_volume"] += int(covered_volume or 0)
            covered_search_volume_total += int(covered_volume or 0)
            if usage.keyword_role == "primary":
                primary_keyword_used = True

    suggested_keywords_total = len(usage_rows)
    used_keywords_total = sum(1 for row in usage_rows if row.used)
    coverage_rate = (
        round(used_keywords_total / suggested_keywords_total, 4)
        if suggested_keywords_total > 0
        else 0.0
    )

    report = {
        "framework_version": KEYWORD_COVERAGE_FRAMEWORK_VERSION,
        "article_version": article_version_number,
        "summary": {
            "suggested_keywords_total": suggested_keywords_total,
            "used_keywords_total": used_keywords_total,
            "coverage_rate": coverage_rate,
            "covered_search_volume_total": covered_search_volume_total,
            "primary_keyword_used": primary_keyword_used,
            "intent_coverage": {
                key: intent_coverage[key]
                for key in sorted(intent_coverage.keys())
            },
        },
        "keywords": [
            {
                "brief_keyword_id": row.brief_keyword_id,
                "keyword_id": row.keyword_id,
                "keyword_text": row.keyword_text,
                "keyword_role": row.keyword_role,
                "keyword_intent": row.keyword_intent,
                "search_volume": row.search_volume,
                "adjusted_volume": row.adjusted_volume,
                "used": row.used,
                "usage_count": row.usage_count,
                "usage_density_pct": row.usage_density_pct,
                "in_h1": row.in_h1,
                "in_first_150_words": row.in_first_150_words,
                "in_h2_h3": row.in_h2_h3,
                "section_hits": row.section_hits,
                "seo_incorporation_score": row.seo_incorporation_score,
            }
            for row in usage_rows
        ],
    }
    return report, usage_rows


async def persist_article_keyword_usages(
    session: AsyncSession,
    *,
    article_id: str,
    article_version_number: int,
    brief_id: str,
    keyword_usages: list[KeywordUsageComputation],
) -> None:
    """Persist per-version keyword usage rows for one article snapshot."""
    existing_result = await session.execute(
        select(ContentArticleKeywordUsage).where(
            ContentArticleKeywordUsage.article_id == article_id,
            ContentArticleKeywordUsage.article_version_number == article_version_number,
        )
    )
    for existing in existing_result.scalars().all():
        await existing.delete(session)

    for usage in keyword_usages:
        ContentArticleKeywordUsage.create(
            session,
            ContentArticleKeywordUsageCreateDTO(
                article_id=article_id,
                article_version_number=article_version_number,
                brief_id=brief_id,
                brief_keyword_id=usage.brief_keyword_id,
                keyword_id=usage.keyword_id,
                keyword_text=usage.keyword_text,
                keyword_role=usage.keyword_role,
                keyword_intent=usage.keyword_intent,
                search_volume=usage.search_volume,
                adjusted_volume=usage.adjusted_volume,
                used=usage.used,
                usage_count=usage.usage_count,
                usage_density_pct=usage.usage_density_pct,
                in_h1=usage.in_h1,
                in_first_150_words=usage.in_first_150_words,
                in_h2_h3=usage.in_h2_h3,
                section_hits=usage.section_hits,
                seo_incorporation_score=usage.seo_incorporation_score,
            ),
        )
    await session.flush()
