"""Deterministic renderer for CMS-agnostic article blocks."""

from __future__ import annotations

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


def _render_block(block: dict[str, Any], fallback_h1: str, h1_used: bool) -> tuple[str, bool]:
    block_type = str(block.get("block_type") or "section")
    if block_type not in BLOCK_TYPES:
        block_type = "section"

    heading = _safe_text(block.get("heading") or "")
    body = _safe_text(block.get("body") or "")
    links = block.get("links") if isinstance(block.get("links"), list) else []

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
        return (
            f"<{tag} data-block-type=\"{block_type}\">{heading_html}{body_html}{_render_links(links)}</{tag}>",
            h1_used,
        )

    if block_type in {"list", "steps"}:
        items = block.get("items") if isinstance(block.get("items"), list) else []
        list_tag = "ol" if block_type == "steps" or block.get("ordered") else "ul"
        heading_html = f"<h2>{heading}</h2>" if heading else ""
        item_html = "".join(f"<li>{_safe_text(item)}</li>" for item in items)
        return (
            f"<section data-block-type=\"{block_type}\">{heading_html}"
            f"<{list_tag}>{item_html}</{list_tag}>{_render_links(links)}</section>",
            h1_used,
        )

    if block_type == "comparison_table":
        columns = block.get("table_columns") if isinstance(block.get("table_columns"), list) else []
        rows = block.get("table_rows") if isinstance(block.get("table_rows"), list) else []
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
        faq_items = block.get("faq_items") if isinstance(block.get("faq_items"), list) else []
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

    cta = block.get("cta") if isinstance(block.get("cta"), dict) else {}
    cta_label = _safe_text(cta.get("label") or "Learn more")
    cta_href = _safe_text(cta.get("href") or "#")
    heading_html = f"<h2>{heading}</h2>" if heading else ""
    body_html = f"<p>{body}</p>" if body else ""
    button_html = f"<a class=\"cta-button\" href=\"{cta_href}\">{cta_label}</a>"
    return (
        f"<aside data-block-type=\"cta\">{heading_html}{body_html}{button_html}{_render_links(links)}</aside>",
        h1_used,
    )


def render_modular_document(document: dict[str, Any]) -> str:
    """Render semantic article HTML from the modular JSON contract."""
    seo_meta = document.get("seo_meta") if isinstance(document.get("seo_meta"), dict) else {}
    fallback_h1 = str(seo_meta.get("h1") or "Untitled")

    blocks = document.get("blocks") if isinstance(document.get("blocks"), list) else []
    article_parts: list[str] = []
    h1_used = False

    for raw_block in blocks:
        if not isinstance(raw_block, dict):
            continue
        rendered, h1_used = _render_block(raw_block, fallback_h1, h1_used)
        article_parts.append(rendered)

    if not h1_used:
        article_parts.insert(0, f"<header data-block-type=\"hero\"><h1>{_safe_text(fallback_h1)}</h1></header>")

    return "<article>" + "".join(article_parts) + "</article>"
