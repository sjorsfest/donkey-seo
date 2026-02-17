"""Shared article generation service used by step execution and API endpoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from app.agents.article_writer import ArticleWriterAgent, ArticleWriterInput
from app.services.content_quality import evaluate_article_quality
from app.services.content_renderer import render_modular_document

_ALLOWED_BLOCK_TYPES = {
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

_SEMANTIC_BY_BLOCK = {
    "hero": "header",
    "summary": "section",
    "section": "section",
    "list": "section",
    "comparison_table": "table",
    "steps": "section",
    "faq": "section",
    "cta": "aside",
    "conclusion": "footer",
    "sources": "section",
}


@dataclass(slots=True)
class GeneratedArticleArtifact:
    """Final generated article payload."""

    title: str
    slug: str
    primary_keyword: str
    modular_document: dict[str, Any]
    rendered_html: str
    qa_report: dict[str, Any]
    status: str
    generation_model: str | None
    generation_temperature: float


class ArticleGenerationService:
    """Generate article blocks, render HTML, run QA, and repair once if needed."""

    def __init__(self, target_domain: str) -> None:
        self.target_domain = target_domain

    async def generate_with_repair(
        self,
        *,
        brief: dict[str, Any],
        writer_instructions: dict[str, Any],
        brief_delta: dict[str, Any],
        brand_context: str,
        conversion_intents: list[str],
    ) -> GeneratedArticleArtifact:
        """Generate article and perform one repair pass on QA failure."""
        qa_feedback: list[str] = []
        agent = ArticleWriterAgent()

        for attempt in range(2):
            output = await agent.run(
                input_data=ArticleWriterInput(
                    brief=brief,
                    writer_instructions=writer_instructions,
                    brief_delta=brief_delta,
                    brand_context=brand_context,
                    conversion_intents=conversion_intents,
                    target_domain=self.target_domain,
                    qa_feedback=qa_feedback,
                )
            )
            document = self._normalize_document(
                output.document.model_dump(),
                brief,
                writer_instructions,
                conversion_intents,
            )
            rendered_html = render_modular_document(document)
            qa_report = evaluate_article_quality(
                document,
                rendered_html,
                required_sections=self._required_sections(brief, brief_delta),
                forbidden_claims=writer_instructions.get("forbidden_claims") or [],
                target_word_count_min=brief.get("target_word_count_min"),
                target_word_count_max=brief.get("target_word_count_max"),
                min_internal_links=self._min_link_count(writer_instructions, "min_internal"),
                min_external_links=self._min_link_count(writer_instructions, "min_external"),
                require_cta=self._cta_required(brief, conversion_intents),
                first_party_domain=self._normalize_domain(self.target_domain),
            )

            if qa_report.get("passed") or attempt == 1:
                status = "draft" if qa_report.get("passed") else "needs_review"
                seo_meta = document.get("seo_meta") if isinstance(document.get("seo_meta"), dict) else {}
                return GeneratedArticleArtifact(
                    title=str(seo_meta.get("h1") or brief.get("primary_keyword") or "Untitled"),
                    slug=str(seo_meta.get("slug") or self._slugify(str(brief.get("primary_keyword") or "article"))),
                    primary_keyword=str(
                        seo_meta.get("primary_keyword")
                        or brief.get("primary_keyword")
                        or ""
                    ),
                    modular_document=document,
                    rendered_html=rendered_html,
                    qa_report=qa_report,
                    status=status,
                    generation_model=getattr(agent, "_model", None),
                    generation_temperature=agent.temperature,
                )

            qa_feedback = [
                f"Fix QA failure '{item}': {self._check_details(qa_report, item)}"
                for item in qa_report.get("required_failures", [])
            ]

        raise RuntimeError("Unreachable article generation state")

    def _check_details(self, qa_report: dict[str, Any], check_name: str) -> str:
        checks = qa_report.get("checks") if isinstance(qa_report.get("checks"), list) else []
        for check in checks:
            if isinstance(check, dict) and check.get("name") == check_name:
                return str(check.get("details"))
        return "missing details"

    def _normalize_document(
        self,
        document: dict[str, Any],
        brief: dict[str, Any],
        writer_instructions: dict[str, Any],
        conversion_intents: list[str],
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "schema_version": "1.0",
            "seo_meta": document.get("seo_meta") if isinstance(document.get("seo_meta"), dict) else {},
            "conversion_plan": document.get("conversion_plan") if isinstance(document.get("conversion_plan"), dict) else {},
            "blocks": document.get("blocks") if isinstance(document.get("blocks"), list) else [],
        }

        seo_meta = normalized["seo_meta"]
        primary_keyword = str(brief.get("primary_keyword") or "")
        if not seo_meta.get("primary_keyword"):
            seo_meta["primary_keyword"] = primary_keyword

        if not seo_meta.get("h1"):
            working_titles = brief.get("working_titles") if isinstance(brief.get("working_titles"), list) else []
            seo_meta["h1"] = str(working_titles[0] if working_titles else primary_keyword or "Untitled")

        if not seo_meta.get("meta_title"):
            seo_meta["meta_title"] = str(seo_meta.get("h1"))

        if not seo_meta.get("meta_description"):
            audience = str(brief.get("target_audience") or "")
            seo_meta["meta_description"] = (
                f"{seo_meta.get('h1')} for {audience}"[:155] if audience else str(seo_meta.get("h1"))[:155]
            )

        if not seo_meta.get("slug"):
            seo_meta["slug"] = self._slugify(primary_keyword or str(seo_meta.get("h1")))

        normalized_blocks = self._normalize_blocks(normalized["blocks"], seo_meta)

        if self._cta_required(brief, conversion_intents) and not any(
            block.get("block_type") == "cta" for block in normalized_blocks
        ):
            money_pages = brief.get("money_page_links") if isinstance(brief.get("money_page_links"), list) else []
            cta_href = "#"
            if money_pages and isinstance(money_pages[0], dict):
                cta_href = str(money_pages[0].get("url") or "#")
            normalized_blocks.append(
                {
                    "block_type": "cta",
                    "semantic_tag": "aside",
                    "heading": "Ready to take the next step?",
                    "body": "Explore the product and see if it fits your workflow.",
                    "cta": {"label": "Learn more", "href": cta_href},
                    "links": [],
                }
            )

        normalized["blocks"] = normalized_blocks

        if not normalized["conversion_plan"].get("primary_intent"):
            normalized["conversion_plan"]["primary_intent"] = str(brief.get("funnel_stage") or "informational")
        if "cta_strategy" not in normalized["conversion_plan"]:
            normalized["conversion_plan"]["cta_strategy"] = []

        if not normalized["conversion_plan"]["cta_strategy"] and normalized["blocks"]:
            normalized["conversion_plan"]["cta_strategy"] = [
                "CTA near conclusion",
                "Contextual CTA in product-relevant sections",
            ]

        return normalized

    def _normalize_blocks(self, blocks: list[Any], seo_meta: dict[str, Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        hero_count = 0

        for raw_block in blocks:
            if not isinstance(raw_block, dict):
                continue

            block_type = str(raw_block.get("block_type") or "section")
            if block_type not in _ALLOWED_BLOCK_TYPES:
                block_type = "section"

            if block_type == "hero":
                hero_count += 1
                if hero_count > 1:
                    block_type = "section"

            block: dict[str, Any] = {
                "block_type": block_type,
                "semantic_tag": _SEMANTIC_BY_BLOCK[block_type],
                "heading": raw_block.get("heading"),
                "level": raw_block.get("level"),
                "body": raw_block.get("body"),
                "items": raw_block.get("items") if isinstance(raw_block.get("items"), list) else [],
                "ordered": bool(raw_block.get("ordered", False)),
                "table_columns": raw_block.get("table_columns") if isinstance(raw_block.get("table_columns"), list) else [],
                "table_rows": raw_block.get("table_rows") if isinstance(raw_block.get("table_rows"), list) else [],
                "faq_items": raw_block.get("faq_items") if isinstance(raw_block.get("faq_items"), list) else [],
                "cta": raw_block.get("cta") if isinstance(raw_block.get("cta"), dict) else None,
                "links": raw_block.get("links") if isinstance(raw_block.get("links"), list) else [],
            }
            normalized.append(block)

        if hero_count == 0:
            normalized.insert(
                0,
                {
                    "block_type": "hero",
                    "semantic_tag": "header",
                    "heading": seo_meta.get("h1"),
                    "level": None,
                    "body": "",
                    "items": [],
                    "ordered": False,
                    "table_columns": [],
                    "table_rows": [],
                    "faq_items": [],
                    "cta": None,
                    "links": [],
                },
            )

        return normalized

    def _required_sections(self, brief: dict[str, Any], brief_delta: dict[str, Any]) -> list[str]:
        required: list[str] = []
        for source in (
            brief.get("must_include_sections"),
            brief_delta.get("must_include_sections"),
        ):
            if isinstance(source, list):
                for item in source:
                    if not isinstance(item, str):
                        continue
                    stripped = item.strip()
                    if stripped and stripped.lower() not in {x.lower() for x in required}:
                        required.append(stripped)
        return required

    def _min_link_count(self, writer_instructions: dict[str, Any], key: str) -> int:
        mins = writer_instructions.get("internal_linking_minimums")
        if isinstance(mins, dict):
            value = mins.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
        return 0

    def _cta_required(self, brief: dict[str, Any], conversion_intents: list[str]) -> bool:
        funnel_stage = str(brief.get("funnel_stage") or "").lower()
        return funnel_stage in {"mofu", "bofu", "transactional", "commercial"} or bool(conversion_intents)

    def _slugify(self, value: str) -> str:
        slug = value.lower().strip()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "article"

    def _normalize_domain(self, domain: str) -> str:
        parsed = urlparse(domain if domain.startswith(("http://", "https://")) else f"https://{domain}")
        return parsed.netloc.lower()
