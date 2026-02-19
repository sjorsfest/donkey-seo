"""Quality checks for generated article artifacts."""

from __future__ import annotations

import re
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _collect_heading_values(document: dict[str, Any]) -> list[str]:
    blocks = _as_list(document.get("blocks"))
    headings: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        heading = block.get("heading")
        if isinstance(heading, str) and heading.strip():
            headings.append(heading.strip().lower())
    return headings


def _collect_text(document: dict[str, Any]) -> str:
    parts: list[str] = []
    seo_meta = _as_dict(document.get("seo_meta"))
    for key in ("h1", "meta_title", "meta_description", "primary_keyword"):
        value = seo_meta.get(key)
        if isinstance(value, str):
            parts.append(value)

    blocks = _as_list(document.get("blocks"))
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for key in ("heading", "body"):
            value = block.get(key)
            if isinstance(value, str):
                parts.append(value)
        for key in ("items", "table_columns"):
            value = block.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
        rows = block.get("table_rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    parts.extend(str(cell) for cell in row)
        faq_items = block.get("faq_items")
        if isinstance(faq_items, list):
            for item in faq_items:
                if isinstance(item, dict):
                    question = item.get("question")
                    answer = item.get("answer")
                    if isinstance(question, str):
                        parts.append(question)
                    if isinstance(answer, str):
                        parts.append(answer)

    return "\n".join(parts)


def _count_links(rendered_html: str, first_party_domain: str | None) -> tuple[int, int]:
    links = re.findall(r'href=\"([^\"]+)\"', rendered_html)
    internal = 0
    external = 0

    normalized_domain = (first_party_domain or "").lower().strip()
    for href in links:
        value = href.strip().lower()
        if not value:
            continue
        if value.startswith("/"):
            internal += 1
            continue
        if normalized_domain and normalized_domain in value:
            internal += 1
            continue
        if value.startswith("http://") or value.startswith("https://"):
            external += 1

    return internal, external


def evaluate_article_quality(
    document: dict[str, Any],
    rendered_html: str,
    *,
    required_sections: list[str] | None,
    forbidden_claims: list[str] | None,
    target_word_count_min: int | None,
    target_word_count_max: int | None,
    min_internal_links: int,
    min_external_links: int,
    require_cta: bool,
    first_party_domain: str | None = None,
) -> dict[str, Any]:
    """Evaluate pass/fail checks for generated content."""
    checks: list[dict[str, Any]] = []

    h1_count = rendered_html.lower().count("<h1")
    checks.append({
        "name": "single_h1",
        "required": True,
        "passed": h1_count == 1,
        "details": {"count": h1_count},
    })

    heading_values = _collect_heading_values(document)
    normalized_headings = {heading.lower() for heading in heading_values}
    missing_sections: list[str] = []
    for section in required_sections or []:
        normalized = section.strip().lower()
        if not normalized:
            continue
        if not any(normalized in heading for heading in normalized_headings):
            missing_sections.append(section)

    checks.append({
        "name": "required_sections",
        "required": True,
        "passed": len(missing_sections) == 0,
        "details": {"missing": missing_sections},
    })

    content_text = _collect_text(document).lower()
    violating_claims = [
        claim for claim in (forbidden_claims or [])
        if claim and claim.lower() in content_text
    ]
    checks.append({
        "name": "forbidden_claims",
        "required": True,
        "passed": len(violating_claims) == 0,
        "details": {"violations": violating_claims},
    })

    words = re.findall(r"\b\w+\b", _collect_text(document))
    word_count = len(words)
    in_range = True
    if target_word_count_min is not None and word_count < target_word_count_min:
        in_range = False
    if target_word_count_max is not None and word_count > target_word_count_max:
        in_range = False
    checks.append({
        "name": "word_count",
        "required": True,
        "passed": in_range,
        "details": {
            "count": word_count,
            "min": target_word_count_min,
            "max": target_word_count_max,
        },
    })

    internal_links, external_links = _count_links(rendered_html, first_party_domain)
    checks.append({
        "name": "internal_links",
        "required": True,
        "passed": internal_links >= min_internal_links,
        "details": {"count": internal_links, "minimum": min_internal_links},
    })
    checks.append({
        "name": "external_links",
        "required": True,
        "passed": external_links >= min_external_links,
        "details": {"count": external_links, "minimum": min_external_links},
    })

    blocks = _as_list(document.get("blocks"))
    has_cta = any(
        isinstance(block, dict) and block.get("block_type") == "cta"
        for block in blocks
    )
    checks.append({
        "name": "cta_presence",
        "required": require_cta,
        "passed": (not require_cta) or has_cta,
        "details": {"required": require_cta, "has_cta": has_cta},
    })

    failed_required = [
        check["name"]
        for check in checks
        if check.get("required") and not check.get("passed")
    ]

    return {
        "passed": len(failed_required) == 0,
        "required_failures": failed_required,
        "checks": checks,
        "summary": {
            "word_count": word_count,
            "internal_links": internal_links,
            "external_links": external_links,
        },
    }
