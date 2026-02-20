"""Deterministic SEO checklist checks for generated articles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

RISK_TERM_PATTERNS: tuple[str, ...] = (
    r"\bhealth\b",
    r"\bmedical\b",
    r"\bfinance\b",
    r"\bfinancial\b",
    r"\blegal\b",
    r"\blaw\b",
    r"\bsecurity\b",
    r"\bcybersecurity\b",
    r"\bsafety\b",
    r"\bai risk\b",
    r"\bpersonal data\b",
    r"\bpii\b",
    r"\bgdpr\b",
    r"\bhipaa\b",
)

ABSOLUTE_CLAIM_PATTERNS: tuple[str, ...] = (
    r"\bguarantee(?:d)?\b",
    r"\b100%\b",
    r"\balways\b",
    r"\bnever\b",
    r"\brisk[- ]free\b",
)

MODULE_A_PAGE_TYPES = {"guide", "how-to", "glossary"}
MODULE_B_PAGE_TYPES = {"comparison", "alternatives", "list"}
MODULE_C_PAGE_TYPES = {"landing", "tool", "calculator", "template"}
MODULE_D_PAGE_TYPES = {"opinion", "thought-leadership", "editorial"}


@dataclass(slots=True)
class DeterministicAuditReport:
    """Deterministic checklist output used by article QA."""

    framework_version: str
    content_type_module: str
    risk_module_applied: bool
    overall_score: int
    hard_failures: list[str]
    soft_warnings: list[str]
    checks: list[dict[str, Any]]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def select_content_type_module(page_type: str | None, search_intent: str | None) -> str:
    """Select content module A/B/C/D based on page type."""
    normalized_page_type = (page_type or "").strip().lower()
    if normalized_page_type in MODULE_A_PAGE_TYPES:
        return "A"
    if normalized_page_type in MODULE_B_PAGE_TYPES:
        return "B"
    if normalized_page_type in MODULE_C_PAGE_TYPES:
        return "C"
    if normalized_page_type in MODULE_D_PAGE_TYPES:
        return "D"
    _ = search_intent
    return "A"


def should_apply_risk_module(
    compliance_notes: list[str] | None,
    brief_text_fields: list[str] | None,
) -> bool:
    """Detect whether risk-sensitive controls should be applied."""
    if compliance_notes and any(
        isinstance(note, str) and note.strip()
        for note in compliance_notes
    ):
        return True

    content = "\n".join(
        item.strip().lower()
        for item in (brief_text_fields or [])
        if isinstance(item, str) and item.strip()
    )
    return any(re.search(pattern, content) for pattern in RISK_TERM_PATTERNS)


def _collect_text(document: dict[str, Any]) -> str:
    parts: list[str] = []
    seo_meta = _as_dict(document.get("seo_meta"))
    h1 = seo_meta.get("h1")
    if isinstance(h1, str):
        parts.append(h1)

    for block in _as_list(document.get("blocks")):
        if not isinstance(block, dict):
            continue
        heading = block.get("heading")
        body = block.get("body")
        if isinstance(heading, str):
            parts.append(heading)
        if isinstance(body, str):
            parts.append(body)
        for item in _as_list(block.get("items")):
            parts.append(str(item))
        for item in _as_list(block.get("table_columns")):
            parts.append(str(item))
        for row in _as_list(block.get("table_rows")):
            if isinstance(row, list):
                for cell in row:
                    parts.append(str(cell))
        for faq_item in _as_list(block.get("faq_items")):
            if isinstance(faq_item, dict):
                question = faq_item.get("question")
                answer = faq_item.get("answer")
                if isinstance(question, str):
                    parts.append(question)
                if isinstance(answer, str):
                    parts.append(answer)

    return "\n".join(parts)


def _collect_words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower())


def _block_heading_level(block: dict[str, Any]) -> int | None:
    block_type = str(block.get("block_type") or "section")
    if block_type == "hero":
        return 1
    if block_type in {"summary", "section", "conclusion", "sources"}:
        level = block.get("level")
        if isinstance(level, int):
            return max(2, min(4, level))
        return 2
    if block_type in {"list", "steps", "comparison_table", "faq", "cta"}:
        return 2
    return None


def _collect_heading_blocks(document: dict[str, Any]) -> list[tuple[str, int]]:
    headings: list[tuple[str, int]] = []
    for block in _as_list(document.get("blocks")):
        if not isinstance(block, dict):
            continue
        heading = block.get("heading")
        if not isinstance(heading, str) or not heading.strip():
            continue
        level = _block_heading_level(block)
        if level is None:
            continue
        headings.append((heading.strip(), level))
    return headings


def _count_links(rendered_html: str, first_party_domain: str | None) -> tuple[int, int]:
    links = re.findall(r'href="([^"]+)"', rendered_html)
    internal = 0
    external = 0
    normalized_domain = (first_party_domain or "").lower().strip()

    for href in links:
        normalized_href = href.strip().lower()
        if not normalized_href:
            continue
        if normalized_href.startswith("/"):
            internal += 1
            continue
        if normalized_domain and normalized_domain in normalized_href:
            internal += 1
            continue
        if normalized_href.startswith("http://") or normalized_href.startswith("https://"):
            external += 1
    return internal, external


def _section_word_count(block: dict[str, Any]) -> int:
    values: list[str] = []
    for key in ("body",):
        value = block.get(key)
        if isinstance(value, str):
            values.append(value)
    for item in _as_list(block.get("items")):
        values.append(str(item))
    for row in _as_list(block.get("table_rows")):
        if isinstance(row, list):
            for cell in row:
                values.append(str(cell))
    for faq_item in _as_list(block.get("faq_items")):
        if isinstance(faq_item, dict):
            question = faq_item.get("question")
            answer = faq_item.get("answer")
            if isinstance(question, str):
                values.append(question)
            if isinstance(answer, str):
                values.append(answer)
    return len(_collect_words(" ".join(values)))


def _keyword_occurrences(text: str, primary_keyword: str) -> int:
    keyword_words = _collect_words(primary_keyword)
    if not keyword_words:
        return 0
    pattern = r"\b" + r"\s+".join(re.escape(word) for word in keyword_words) + r"\b"
    return len(re.findall(pattern, text.lower()))


def run_deterministic_checklist(
    document: dict[str, Any],
    rendered_html: str,
    *,
    primary_keyword: str,
    page_type: str | None,
    search_intent: str | None,
    required_sections: list[str] | None,
    forbidden_claims: list[str] | None,
    target_word_count_min: int | None,
    target_word_count_max: int | None,
    min_internal_links: int,
    min_external_links: int,
    require_cta: bool,
    first_party_domain: str | None,
    compliance_notes: list[str] | None,
    brief_text_fields: list[str] | None,
    keyword_density_soft_min: float = 0.2,
    keyword_density_soft_max: float = 2.5,
) -> DeterministicAuditReport:
    """Run deterministic checks aligned to the SEO checklist modules."""
    checks: list[dict[str, Any]] = []

    content_type_module = select_content_type_module(page_type, search_intent)
    risk_module_applied = should_apply_risk_module(compliance_notes, brief_text_fields)

    content_text = _collect_text(document)
    content_text_lower = content_text.lower()
    words = _collect_words(content_text)
    word_count = len(words)

    h1_count = rendered_html.lower().count("<h1")
    checks.append({
        "name": "single_h1",
        "required": True,
        "passed": h1_count == 1,
        "details": {"count": h1_count},
    })

    heading_blocks = _collect_heading_blocks(document)
    hierarchy_violations: list[dict[str, Any]] = []
    previous_level = 1
    for heading, level in heading_blocks:
        if level > previous_level + 1:
            hierarchy_violations.append(
                {"heading": heading, "level": level, "previous_level": previous_level}
            )
        previous_level = level
    checks.append({
        "name": "heading_hierarchy",
        "required": True,
        "passed": len(hierarchy_violations) == 0 and len(heading_blocks) > 0,
        "details": {"violations": hierarchy_violations, "heading_count": len(heading_blocks)},
    })

    normalized_headings = [heading.strip().lower() for heading, _ in heading_blocks]
    missing_sections: list[str] = []
    for section in required_sections or []:
        normalized_section = section.strip().lower()
        if not normalized_section:
            continue
        if not any(normalized_section in heading for heading in normalized_headings):
            missing_sections.append(section)
    checks.append({
        "name": "required_sections",
        "required": True,
        "passed": len(missing_sections) == 0,
        "details": {"missing": missing_sections},
    })

    normalized_keyword = " ".join(_collect_words(primary_keyword))
    seo_meta = _as_dict(document.get("seo_meta"))
    h1_value = str(seo_meta.get("h1") or "")
    keyword_in_h1 = bool(normalized_keyword) and normalized_keyword in h1_value.lower()
    checks.append({
        "name": "primary_keyword_in_h1",
        "required": True,
        "passed": keyword_in_h1,
        "details": {"keyword": primary_keyword, "h1": h1_value},
    })

    first_150_words = " ".join(words[:150])
    keyword_in_first_150 = bool(normalized_keyword) and normalized_keyword in first_150_words
    checks.append({
        "name": "primary_keyword_first_150_words",
        "required": True,
        "passed": keyword_in_first_150,
        "details": {"keyword": primary_keyword},
    })

    keyword_in_h2 = any(
        level == 2 and normalized_keyword and normalized_keyword in heading.lower()
        for heading, level in heading_blocks
    )
    checks.append({
        "name": "primary_keyword_in_h2",
        "required": True,
        "passed": keyword_in_h2,
        "details": {"keyword": primary_keyword},
    })

    violating_claims = [
        claim for claim in (forbidden_claims or [])
        if claim and claim.lower() in content_text_lower
    ]
    checks.append({
        "name": "forbidden_claims",
        "required": True,
        "passed": len(violating_claims) == 0,
        "details": {"violations": violating_claims},
    })

    in_word_count_range = True
    if target_word_count_min is not None and word_count < target_word_count_min:
        in_word_count_range = False
    if target_word_count_max is not None and word_count > target_word_count_max:
        in_word_count_range = False
    checks.append({
        "name": "word_count",
        "required": True,
        "passed": in_word_count_range,
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

    blocks = [block for block in _as_list(document.get("blocks")) if isinstance(block, dict)]
    has_cta = any(block.get("block_type") == "cta" for block in blocks)
    checks.append({
        "name": "cta_presence",
        "required": require_cta,
        "passed": (not require_cta) or has_cta,
        "details": {"required": require_cta, "has_cta": has_cta},
    })

    keyword_occurrences = _keyword_occurrences(content_text, primary_keyword)
    keyword_word_count = len(_collect_words(primary_keyword))
    keyword_density = 0.0
    if word_count > 0 and keyword_word_count > 0:
        keyword_density = (keyword_occurrences * keyword_word_count / word_count) * 100
    density_passed = keyword_density_soft_min <= keyword_density <= keyword_density_soft_max
    checks.append({
        "name": "keyword_density_soft_band",
        "required": False,
        "passed": density_passed,
        "details": {
            "density": round(keyword_density, 3),
            "minimum": keyword_density_soft_min,
            "maximum": keyword_density_soft_max,
        },
    })

    thin_sections: list[str] = []
    for block in blocks:
        heading = block.get("heading")
        if not isinstance(heading, str) or not heading.strip():
            continue
        if block.get("block_type") == "hero":
            continue
        if _section_word_count(block) < 60:
            thin_sections.append(heading.strip())
    checks.append({
        "name": "thin_sections",
        "required": False,
        "passed": len(thin_sections) == 0,
        "details": {"headings": thin_sections},
    })

    seen_headings: set[str] = set()
    repeated_headings: list[str] = []
    for heading, _ in heading_blocks:
        normalized_heading = heading.strip().lower()
        if normalized_heading in seen_headings:
            repeated_headings.append(heading.strip())
        else:
            seen_headings.add(normalized_heading)
    checks.append({
        "name": "repeated_headings",
        "required": False,
        "passed": len(repeated_headings) == 0,
        "details": {"headings": repeated_headings},
    })

    if content_type_module == "B":
        has_comparison_block = any(
            block.get("block_type") == "comparison_table"
            for block in blocks
        )
        checks.append({
            "name": "module_b_comparison_block",
            "required": False,
            "passed": has_comparison_block,
            "details": {"has_comparison_table": has_comparison_block},
        })

    if risk_module_applied:
        absolute_claim_hits: list[str] = []
        for pattern in ABSOLUTE_CLAIM_PATTERNS:
            absolute_claim_hits.extend(re.findall(pattern, content_text_lower))
        checks.append({
            "name": "risk_sensitive_absolute_claims",
            "required": True,
            "passed": len(absolute_claim_hits) == 0,
            "details": {"hits": absolute_claim_hits},
        })

    hard_failures = [
        check["name"]
        for check in checks
        if check.get("required") and not check.get("passed")
    ]
    soft_warnings = [
        check["name"]
        for check in checks
        if not check.get("required") and not check.get("passed")
    ]

    overall_score = max(0, min(100, 100 - (len(hard_failures) * 12) - (len(soft_warnings) * 4)))

    return DeterministicAuditReport(
        framework_version="seo-checklist-v1",
        content_type_module=content_type_module,
        risk_module_applied=risk_module_applied,
        overall_score=overall_score,
        hard_failures=hard_failures,
        soft_warnings=soft_warnings,
        checks=checks,
    )
