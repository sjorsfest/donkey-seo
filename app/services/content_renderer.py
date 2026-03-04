"""Deterministic renderer for CMS-agnostic article blocks."""

from __future__ import annotations

import json
from html import escape
from typing import Any

BLOCK_TYPES = {
    "hero",
    "summary",
    "section",
    "list",
    "comparison_table",
    "steps",
    "faq",
    "cta",
    "conclusion",
    "sources",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_link_list(value: Any) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            links.append(item)
    return links


def _as_schema_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _render_links(links: list[dict[str, Any]]) -> str:
    if not links:
        return ""

    parts = ["<ul class=\"article-links\">"]
    for link in links:
        href = _safe_text(link.get("href") or "")
        anchor = _safe_text(link.get("anchor") or link.get("label") or href)
        if not href:
            continue
        parts.append(f"<li><a href=\"{href}\">{anchor}</a></li>")
    parts.append("</ul>")
    return "".join(parts)


def _heading_tag(level: int) -> str:
    if level < 2:
        return "h2"
    if level > 4:
        return "h4"
    return f"h{level}"


def _fallback_items_from_table_rows(value: Any) -> list[str]:
    items: list[str] = []
    for row in _as_list(value):
        if not isinstance(row, list):
            continue
        cells = [str(cell).strip() for cell in row if str(cell).strip()]
        if not cells:
            continue
        if len(cells) >= 2:
            items.append(f"{cells[0]}: {'; '.join(cells[1:])}")
        else:
            items.append(cells[0])
    return items


def _render_block(block: dict[str, Any], fallback_h1: str, h1_used: bool) -> tuple[str, bool]:
    block_type = str(block.get("block_type") or "section")
    if block_type not in BLOCK_TYPES:
        block_type = "section"

    heading = _safe_text(block.get("heading") or "")
    body = _safe_text(block.get("body") or "")
    links = _as_link_list(block.get("links"))

    if block_type == "hero":
        h1_value = heading or _safe_text(fallback_h1)
        h1_html = "" if h1_used else f"<h1>{h1_value}</h1>"
        body_html = f"<p>{body}</p>" if body else ""
        return (
            f"<header data-block-type=\"hero\">{h1_html}{body_html}{_render_links(links)}</header>",
            h1_used or bool(h1_html),
        )

    if block_type in {"summary", "section", "conclusion", "sources"}:
        tag = "footer" if block_type == "conclusion" else "section"
        heading_level = int(block.get("level") or 2)
        heading_tag = _heading_tag(heading_level)
        heading_html = f"<{heading_tag}>{heading}</{heading_tag}>" if heading else ""
        body_html = f"<p>{body}</p>" if body else ""
        items = _as_list(block.get("items"))
        list_html = ""
        if items:
            list_tag = "ol" if block.get("ordered") else "ul"
            item_html = "".join(f"<li>{_safe_text(item)}</li>" for item in items)
            list_html = f"<{list_tag}>{item_html}</{list_tag}>"
        return (
            (
                f"<{tag} data-block-type=\"{block_type}\">"
                f"{heading_html}{body_html}{list_html}{_render_links(links)}</{tag}>"
            ),
            h1_used,
        )

    if block_type in {"list", "steps"}:
        items = _as_list(block.get("items"))
        if not items:
            items = _fallback_items_from_table_rows(block.get("table_rows"))
        list_tag = "ol" if block_type == "steps" or block.get("ordered") else "ul"
        heading_html = f"<h2>{heading}</h2>" if heading else ""
        item_html = "".join(f"<li>{_safe_text(item)}</li>" for item in items)
        return (
            f"<section data-block-type=\"{block_type}\">{heading_html}"
            f"<{list_tag}>{item_html}</{list_tag}>{_render_links(links)}</section>",
            h1_used,
        )

    if block_type == "comparison_table":
        columns = _as_list(block.get("table_columns"))
        rows = _as_list(block.get("table_rows"))
        heading_html = f"<h2>{heading}</h2>" if heading else ""
        header_cells = "".join(f"<th>{_safe_text(col)}</th>" for col in columns)
        body_rows: list[str] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            cells = "".join(f"<td>{_safe_text(cell)}</td>" for cell in row)
            body_rows.append(f"<tr>{cells}</tr>")
        table_html = (
            f"<table><thead><tr>{header_cells}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )
        return (
            f"<section data-block-type=\"comparison_table\">{heading_html}{table_html}{_render_links(links)}</section>",
            h1_used,
        )

    if block_type == "faq":
        faq_items = _as_list(block.get("faq_items"))
        heading_html = f"<h2>{heading}</h2>" if heading else ""
        details_html: list[str] = []
        for item in faq_items:
            if not isinstance(item, dict):
                continue
            question = _safe_text(item.get("question") or "")
            answer = _safe_text(item.get("answer") or "")
            if not question:
                continue
            details_html.append(
                f"<details><summary>{question}</summary><p>{answer}</p></details>"
            )
        return (
            f"<section data-block-type=\"faq\">{heading_html}{''.join(details_html)}{_render_links(links)}</section>",
            h1_used,
        )

    cta = _as_dict(block.get("cta"))
    cta_label = _safe_text(cta.get("label") or "Learn more")
    cta_href = _safe_text(cta.get("href") or "#")
    heading_html = f"<h2>{heading}</h2>" if heading else ""
    body_html = f"<p>{body}</p>" if body else ""
    button_html = f"<a class=\"cta-button\" href=\"{cta_href}\">{cta_label}</a>"
    return (
        f"<aside data-block-type=\"cta\">{heading_html}{body_html}{button_html}{_render_links(links)}</aside>",
        h1_used,
    )


def _render_json_ld_scripts(document: dict[str, Any]) -> str:
    scripts: list[str] = []
    for schema_item in _as_schema_list(document.get("structured_data")):
        payload = json.dumps(schema_item, ensure_ascii=True, separators=(",", ":"))
        payload = payload.replace("</", "<\\/")
        scripts.append(f"<script type=\"application/ld+json\">{payload}</script>")
    return "".join(scripts)


def render_modular_document(document: dict[str, Any]) -> str:
    """Render semantic article HTML from the modular JSON contract."""
    seo_meta = _as_dict(document.get("seo_meta"))
    fallback_h1 = str(seo_meta.get("h1") or "Untitled")

    blocks = _as_list(document.get("blocks"))
    article_parts: list[str] = []
    h1_used = False

    for raw_block in blocks:
        if not isinstance(raw_block, dict):
            continue
        rendered, h1_used = _render_block(raw_block, fallback_h1, h1_used)
        article_parts.append(rendered)

    if not h1_used:
        article_parts.insert(0, f"<header data-block-type=\"hero\"><h1>{_safe_text(fallback_h1)}</h1></header>")

    article_html = "<article>" + "".join(article_parts) + "</article>"
    return article_html + _render_json_ld_scripts(document)
