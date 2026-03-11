"""Shared article generation service used by step execution and API endpoints."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.agents.article_seo_auditor import ArticleSEOAuditorAgent, ArticleSEOAuditorInput
from app.agents.article_writer import ArticleWriterAgent, ArticleWriterInput
from app.services.content_quality import evaluate_article_quality
from app.services.content_renderer import render_modular_document
from app.services.seo_checklist import DeterministicAuditReport, run_deterministic_checklist

logger = logging.getLogger(__name__)

_LINK_VALIDATION_TIMEOUT_SECONDS = 4.0
_TOPIC_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
}

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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
    """Generate article blocks, render HTML, and run a single SEO audit pass."""

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
        """Generate article and run one hybrid QA audit pass."""
        writer_agent = ArticleWriterAgent()
        seo_auditor = ArticleSEOAuditorAgent()

        pass_fail_thresholds = _as_dict(writer_instructions.get("pass_fail_thresholds"))
        seo_score_target = self._int_threshold(pass_fail_thresholds, "seo_score_target", default=75)
        density_soft_min = self._float_threshold(
            pass_fail_thresholds,
            "keyword_density_soft_min",
            default=0.2,
        )
        density_soft_max = self._float_threshold(
            pass_fail_thresholds,
            "keyword_density_soft_max",
            default=2.5,
        )

        required_sections = self._required_sections(brief, brief_delta)
        forbidden_claims = writer_instructions.get("forbidden_claims") or []
        min_internal_links = self._min_link_count(writer_instructions, "min_internal")
        min_external_links = self._min_link_count(writer_instructions, "min_external")
        require_cta = self._cta_required(brief, conversion_intents)
        first_party_domain = self._normalize_domain(self.target_domain)
        link_validation_cache: dict[str, bool] = {}

        async with httpx.AsyncClient(
            timeout=_LINK_VALIDATION_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as link_client:
            output = await writer_agent.run(
                input_data=ArticleWriterInput(
                    brief=brief,
                    writer_instructions=writer_instructions,
                    brief_delta=brief_delta,
                    brand_context=brand_context,
                    conversion_intents=conversion_intents,
                    target_domain=self.target_domain,
                    qa_feedback=[],
                    existing_document=None,
                )
            )
            document = self._normalize_document(
                output.document.model_dump(),
                brief,
                writer_instructions,
                conversion_intents,
            )
            document = self._apply_brief_internal_link_plan(document=document, brief=brief)
            document = await self._filter_invalid_document_links(
                document=document,
                link_cache=link_validation_cache,
                client=link_client,
            )
            rendered_html = render_modular_document(document)
            base_qa_report = evaluate_article_quality(
                document,
                rendered_html,
                required_sections=required_sections,
                forbidden_claims=forbidden_claims,
                target_word_count_min=brief.get("target_word_count_min"),
                target_word_count_max=brief.get("target_word_count_max"),
                min_internal_links=min_internal_links,
                min_external_links=min_external_links,
                require_cta=require_cta,
                first_party_domain=first_party_domain,
            )
            deterministic_report = run_deterministic_checklist(
                document,
                rendered_html,
                primary_keyword=str(brief.get("primary_keyword") or ""),
                topic_anchor=self._topic_anchor(brief=brief, document=document),
                page_type=brief.get("page_type"),
                search_intent=brief.get("search_intent"),
                blueprint_key=brief.get("blueprint_key"),
                required_sections=required_sections,
                forbidden_claims=forbidden_claims,
                target_word_count_min=brief.get("target_word_count_min"),
                target_word_count_max=brief.get("target_word_count_max"),
                min_internal_links=min_internal_links,
                min_external_links=min_external_links,
                require_cta=require_cta,
                first_party_domain=first_party_domain,
                compliance_notes=writer_instructions.get("compliance_notes") or [],
                brief_text_fields=[
                    str(brief.get("primary_keyword") or ""),
                    str(brief.get("search_intent") or ""),
                    str(brief.get("page_type") or ""),
                    str(brief.get("target_audience") or ""),
                    str(brief.get("reader_job_to_be_done") or ""),
                    str(brand_context or ""),
                    str(_as_dict(document.get("seo_meta")).get("h1") or ""),
                ],
                keyword_density_soft_min=density_soft_min,
                keyword_density_soft_max=density_soft_max,
            )
            llm_audit = await self._run_llm_seo_audit(
                auditor=seo_auditor,
                brief=brief,
                writer_instructions=writer_instructions,
                document=document,
                deterministic_report=deterministic_report,
            )
            qa_report = self._merge_qa_reports(
                base_qa_report=base_qa_report,
                deterministic_report=deterministic_report,
                llm_audit=llm_audit,
                seo_score_target=seo_score_target,
            )
            revision_feedback = self._build_revision_feedback(qa_report, brief)
            seo_audit = _as_dict(qa_report.get("seo_audit"))
            seo_audit["revision_feedback"] = revision_feedback
            qa_report["seo_audit"] = seo_audit

            seo_meta = _as_dict(document.get("seo_meta"))
            return GeneratedArticleArtifact(
                title=str(seo_meta.get("h1") or brief.get("primary_keyword") or "Untitled"),
                slug=str(
                    seo_meta.get("slug")
                    or self._slugify(str(brief.get("primary_keyword") or "article"))
                ),
                primary_keyword=str(
                    seo_meta.get("primary_keyword")
                    or brief.get("primary_keyword")
                    or ""
                ),
                modular_document=document,
                rendered_html=rendered_html,
                qa_report=qa_report,
                status="draft",
                generation_model=getattr(writer_agent, "_model", None),
                generation_temperature=writer_agent.temperature,
            )

    async def _run_llm_seo_audit(
        self,
        *,
        auditor: ArticleSEOAuditorAgent,
        brief: dict[str, Any],
        writer_instructions: dict[str, Any],
        document: dict[str, Any],
        deterministic_report: DeterministicAuditReport,
    ) -> dict[str, Any]:
        try:
            audit_output = await auditor.run(
                input_data=ArticleSEOAuditorInput(
                    brief=brief,
                    writer_instructions=writer_instructions,
                    document=document,
                    deterministic_audit={
                        "framework_version": deterministic_report.framework_version,
                        "overall_score": deterministic_report.overall_score,
                        "hard_failures": deterministic_report.hard_failures,
                        "soft_warnings": deterministic_report.soft_warnings,
                        "checks": deterministic_report.checks,
                    },
                    content_type_module=deterministic_report.content_type_module,
                    risk_module_applied=deterministic_report.risk_module_applied,
                    blueprint_key=str(brief.get("blueprint_key") or ""),
                )
            )
            return audit_output.model_dump()
        except Exception as exc:
            logger.warning(
                "LLM SEO audit failed, continuing with deterministic QA",
                extra={"error": str(exc)},
            )
            return {
                "checklist_items": [],
                "claim_integrity": [],
                "overall_score": deterministic_report.overall_score,
                "hard_failures": [],
                "soft_warnings": [],
                "revision_instructions": [],
            }

    def _merge_qa_reports(
        self,
        *,
        base_qa_report: dict[str, Any],
        deterministic_report: DeterministicAuditReport,
        llm_audit: dict[str, Any],
        seo_score_target: int,
    ) -> dict[str, Any]:
        base_checks = [
            check
            for check in _as_list(base_qa_report.get("checks"))
            if isinstance(check, dict)
        ]
        llm_checklist = [
            check for check in _as_list(llm_audit.get("checklist_items"))
            if isinstance(check, dict)
        ]
        llm_checks: list[dict[str, Any]] = []
        for item in llm_checklist:
            severity = str(item.get("severity") or "soft")
            status = str(item.get("status") or "warning")
            llm_checks.append(
                {
                    "name": str(item.get("id") or "llm_check"),
                    "required": severity == "hard",
                    "passed": status == "pass",
                    "details": {
                        "status": status,
                        "severity": severity,
                        "evidence": str(item.get("evidence") or ""),
                        "fix_instruction": str(item.get("fix_instruction") or ""),
                    },
                }
            )

        checks = [*base_checks, *deterministic_report.checks, *llm_checks]

        base_required_failures = _as_list(base_qa_report.get("required_failures"))
        hard_failures = self._dedupe_strings(
            [
                *[str(item) for item in base_required_failures if item],
                *deterministic_report.hard_failures,
                *[str(item) for item in _as_list(llm_audit.get("hard_failures")) if item],
            ]
        )
        soft_warnings = self._dedupe_strings(
            [
                *deterministic_report.soft_warnings,
                *[str(item) for item in _as_list(llm_audit.get("soft_warnings")) if item],
            ]
        )

        llm_score = llm_audit.get("overall_score")
        if isinstance(llm_score, int):
            overall_score = int((deterministic_report.overall_score + llm_score) / 2)
        else:
            overall_score = deterministic_report.overall_score

        summary = dict(_as_dict(base_qa_report.get("summary")))
        summary["overall_score"] = overall_score
        summary["hard_failure_count"] = len(hard_failures)
        summary["soft_warning_count"] = len(soft_warnings)

        llm_findings = {
            "checklist_items": llm_checklist,
            "claim_integrity": [
                item
                for item in _as_list(llm_audit.get("claim_integrity"))
                if isinstance(item, dict)
            ],
        }
        revision_feedback = [
            str(item)
            for item in _as_list(llm_audit.get("revision_instructions"))
            if isinstance(item, str) and item.strip()
        ]

        return {
            "passed": len(hard_failures) == 0,
            "required_failures": hard_failures,
            "checks": checks,
            "summary": summary,
            "seo_audit": {
                "framework_version": deterministic_report.framework_version,
                "content_type_module": deterministic_report.content_type_module,
                "risk_module_applied": deterministic_report.risk_module_applied,
                "overall_score": overall_score,
                "score_target": seo_score_target,
                "hard_failures": hard_failures,
                "soft_warnings": soft_warnings,
                "llm_findings": llm_findings,
                "revision_feedback": revision_feedback,
            },
        }

    def _build_revision_feedback(
        self,
        qa_report: dict[str, Any],
        brief: dict[str, Any],
    ) -> list[str]:
        seo_audit = _as_dict(qa_report.get("seo_audit"))
        hard_failures = [str(item) for item in _as_list(seo_audit.get("hard_failures")) if item]
        feedback: list[str] = []

        for check_name in hard_failures:
            details = self._check_details(qa_report, check_name)
            feedback.append(f"Fix QA failure '{check_name}': {details}")

        for item in _as_list(seo_audit.get("revision_feedback")):
            if isinstance(item, str) and item.strip():
                feedback.append(item.strip())

        keyword = str(brief.get("primary_keyword") or "").strip()
        topic_anchor = self._topic_anchor(brief=brief, document=None)
        search_intent = str(brief.get("search_intent") or "informational").strip()
        target_audience = str(brief.get("target_audience") or "").strip()
        if keyword:
            overlap_ratio = self._keyword_topic_overlap_ratio(keyword, topic_anchor)
            if topic_anchor and overlap_ratio < 0.34:
                feedback.append(
                    f"Prioritize the working-title topic '{topic_anchor}'. "
                    f"Use primary keyword '{keyword}' naturally where relevant."
                )
            else:
                feedback.append(
                    f"Preserve primary keyword strategy for '{keyword}' without keyword stuffing."
                )
        feedback.append(f"Preserve search intent: '{search_intent}'.")
        if target_audience:
            feedback.append(f"Preserve ICP hook for audience: '{target_audience}'.")
        feedback.append(
            "Apply minimal targeted edits; keep existing structure unless "
            "a failing check requires change."
        )
        feedback.append(
            "Never use em dashes (—); use commas, periods, or parentheses instead."
        )

        return self._dedupe_strings(feedback)

    def _dedupe_strings(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _int_threshold(self, payload: dict[str, Any], key: str, *, default: int) -> int:
        value = payload.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float_threshold(self, payload: dict[str, Any], key: str, *, default: float) -> float:
        value = payload.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _check_details(self, qa_report: dict[str, Any], check_name: str) -> str:
        checks = _as_list(qa_report.get("checks"))
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
        seo_meta: dict[str, Any] = _as_dict(document.get("seo_meta"))
        conversion_plan: dict[str, Any] = _as_dict(document.get("conversion_plan"))
        blocks: list[Any] = _as_list(document.get("blocks"))
        structured_data: Any = document.get("structured_data")
        normalized: dict[str, Any] = {
            "schema_version": "1.0",
            "seo_meta": seo_meta,
            "conversion_plan": conversion_plan,
            "blocks": blocks,
        }
        if isinstance(structured_data, (dict, list)):
            normalized["structured_data"] = structured_data

        primary_keyword = str(brief.get("primary_keyword") or "")
        locked_title = str(brief.get("locked_title") or "").strip()
        if not seo_meta.get("primary_keyword"):
            seo_meta["primary_keyword"] = primary_keyword

        if locked_title:
            seo_meta["h1"] = locked_title
        elif not seo_meta.get("h1"):
            working_titles = _as_list(brief.get("working_titles"))
            seo_meta["h1"] = str(
                working_titles[0] if working_titles else primary_keyword or "Untitled"
            )

        if not seo_meta.get("meta_title"):
            seo_meta["meta_title"] = str(seo_meta.get("h1"))

        if not seo_meta.get("meta_description"):
            audience = str(brief.get("target_audience") or "")
            seo_meta["meta_description"] = (
                f"{seo_meta.get('h1')} for {audience}"[:155]
                if audience
                else str(seo_meta.get("h1"))[:155]
            )

        if not seo_meta.get("slug"):
            seo_meta["slug"] = self._slugify(primary_keyword or str(seo_meta.get("h1")))

        normalized_blocks = self._normalize_blocks(blocks, seo_meta)
        if locked_title:
            for block in normalized_blocks:
                if not isinstance(block, dict):
                    continue
                if str(block.get("block_type") or "").strip().lower() != "hero":
                    continue
                block["heading"] = locked_title
                break

        if self._cta_required(brief, conversion_intents) and not any(
            block.get("block_type") == "cta" for block in normalized_blocks
        ):
            money_pages = _as_list(brief.get("money_page_links"))
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
        self._sync_faqpage_structured_data(document=normalized)

        if not conversion_plan.get("primary_intent"):
            conversion_plan["primary_intent"] = str(brief.get("funnel_stage") or "informational")
        if "cta_strategy" not in conversion_plan or not isinstance(
            conversion_plan.get("cta_strategy"),
            list,
        ):
            conversion_plan["cta_strategy"] = []

        cta_strategy = _as_list(conversion_plan.get("cta_strategy"))
        if not cta_strategy and normalized["blocks"]:
            conversion_plan["cta_strategy"] = [
                "CTA near conclusion",
                "Contextual CTA in product-relevant sections",
            ]

        normalized["conversion_plan"] = conversion_plan
        sanitized = self._sanitize_em_dashes(normalized)
        return sanitized if isinstance(sanitized, dict) else normalized

    def _sync_faqpage_structured_data(
        self,
        *,
        document: dict[str, Any],
    ) -> None:
        structured_data = self._coerce_structured_data(document.get("structured_data"))
        retained_items = [
            item for item in structured_data if not self._is_faqpage_schema(item)
        ]

        faq_entities = self._faq_main_entities_from_blocks(_as_list(document.get("blocks")))
        if faq_entities:
            retained_items.append(
                {
                    "@context": "https://schema.org",
                    "@type": "FAQPage",
                    "mainEntity": faq_entities,
                }
            )

        if retained_items:
            document["structured_data"] = retained_items
            return
        document.pop("structured_data", None)

    def _coerce_structured_data(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    def _is_faqpage_schema(self, schema_item: dict[str, Any]) -> bool:
        schema_type = schema_item.get("@type")
        if isinstance(schema_type, str):
            return schema_type.strip().lower() == "faqpage"
        if isinstance(schema_type, list):
            return any(
                isinstance(item, str) and item.strip().lower() == "faqpage"
                for item in schema_type
            )
        return False

    def _faq_main_entities_from_blocks(
        self,
        blocks: list[Any],
    ) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        seen_questions: set[str] = set()

        for block in blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("block_type") or "").strip().lower() != "faq":
                continue

            for raw_item in _as_list(block.get("faq_items")):
                if not isinstance(raw_item, dict):
                    continue
                question = str(raw_item.get("question") or "").strip()
                answer = str(raw_item.get("answer") or "").strip()
                if not question or not answer:
                    continue
                question_key = question.lower()
                if question_key in seen_questions:
                    continue
                seen_questions.add(question_key)
                entities.append(
                    {
                        "@type": "Question",
                        "name": question,
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": answer,
                        },
                    }
                )

        return entities

    def _sanitize_em_dashes(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize_em_dash_text(value)
        if isinstance(value, list):
            return [self._sanitize_em_dashes(item) for item in value]
        if isinstance(value, dict):
            return {key: self._sanitize_em_dashes(item) for key, item in value.items()}
        return value

    def _sanitize_em_dash_text(self, text: str) -> str:
        if "—" not in text:
            return text
        sanitized = re.sub(r"\s*—\s*", ", ", text)
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        return sanitized

    async def _filter_invalid_document_links(
        self,
        *,
        document: dict[str, Any],
        link_cache: dict[str, bool],
        client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        blocks = _as_list(document.get("blocks"))
        if not blocks:
            return document

        dropped_links = 0
        neutralized_ctas = 0
        for block in blocks:
            if not isinstance(block, dict):
                continue

            valid_links: list[dict[str, Any]] = []
            for raw_link in _as_list(block.get("links")):
                if not isinstance(raw_link, dict):
                    continue
                target_brief_id = str(raw_link.get("target_brief_id") or "").strip()
                href = str(raw_link.get("href") or "").strip()
                if target_brief_id:
                    # Deferred internal link: keep metadata and resolve href later.
                    valid_links.append(raw_link)
                    continue
                if not href:
                    continue
                if await self._href_exists(href=href, link_cache=link_cache, client=client):
                    valid_links.append(raw_link)
                else:
                    dropped_links += 1
            block["links"] = valid_links

            cta = block.get("cta")
            if not isinstance(cta, dict):
                continue
            cta_href = str(cta.get("href") or "").strip()
            if not cta_href:
                continue
            if await self._href_exists(href=cta_href, link_cache=link_cache, client=client):
                continue
            cta["href"] = "#"
            block["cta"] = cta
            neutralized_ctas += 1

        if dropped_links > 0 or neutralized_ctas > 0:
            logger.info(
                "Filtered invalid links from generated article",
                extra={
                    "dropped_links": dropped_links,
                    "neutralized_ctas": neutralized_ctas,
                    "target_domain": self.target_domain,
                },
            )

        return document

    def _apply_brief_internal_link_plan(
        self,
        *,
        document: dict[str, Any],
        brief: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject deterministic internal links from brief recommendations.

        This guarantees planned interlinks (including unpublished predecessors)
        are present even when the writer model omits them.
        """
        recommendations = [
            rec
            for rec in _as_list(brief.get("internal_links_out"))
            if isinstance(rec, dict)
        ]
        if not recommendations:
            return document

        blocks = _as_list(document.get("blocks"))
        if not blocks:
            return document

        existing_keys: set[str] = set()
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for link in _as_list(block.get("links")):
                if not isinstance(link, dict):
                    continue
                existing_keys.add(
                    self._link_key(
                        href=str(link.get("href") or ""),
                        anchor=str(link.get("anchor") or ""),
                    )
                )

        for rec in recommendations[:6]:
            target_type = str(rec.get("target_type") or "").strip().lower()
            href = str(rec.get("target_url") or "").strip()
            target_brief_id = str(rec.get("target_brief_id") or "").strip()

            # Deferred batch links intentionally keep empty href until publish_url exists.
            if target_type == "batch_brief" and target_brief_id and not href:
                href = ""
            if not href:
                if not target_brief_id:
                    continue

            anchor = str(rec.get("anchor_text") or "").strip() or self._fallback_anchor_from_href(href)
            link_key = self._link_key(href=href, anchor=anchor)
            if link_key in existing_keys:
                continue

            placement = str(rec.get("placement_section") or "").strip()
            block_idx = self._select_internal_link_block_index(blocks=blocks, placement_section=placement)

            block = blocks[block_idx]
            if not isinstance(block, dict):
                continue
            block_links = _as_list(block.get("links"))
            link_payload: dict[str, Any] = {"anchor": anchor, "href": href}
            if target_brief_id:
                link_payload["target_brief_id"] = target_brief_id
            block_links.append(link_payload)
            block["links"] = block_links
            existing_keys.add(link_key)

        document["blocks"] = blocks
        return document

    def _select_internal_link_block_index(
        self,
        *,
        blocks: list[Any],
        placement_section: str,
    ) -> int:
        normalized_placement = placement_section.strip().lower()
        if normalized_placement:
            for idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                heading = str(block.get("heading") or "").strip().lower()
                if not heading:
                    continue
                if heading == normalized_placement or normalized_placement in heading:
                    return idx

        preferred_types = {
            "section",
            "summary",
            "list",
            "steps",
            "comparison_table",
            "conclusion",
        }
        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if str(block.get("block_type") or "").strip().lower() in preferred_types:
                return idx

        return 0

    def _fallback_anchor_from_href(self, href: str) -> str:
        if not href:
            return "Related article"
        path = href.split("?", 1)[0].rstrip("/")
        slug = path.split("/")[-1]
        words = re.sub(r"[-_]+", " ", slug).strip()
        return words or "Related article"

    def _link_key(self, *, href: str, anchor: str) -> str:
        return "|".join([
            href.strip().lower(),
            anchor.strip().lower(),
        ])

    async def _href_exists(
        self,
        *,
        href: str,
        link_cache: dict[str, bool],
        client: httpx.AsyncClient,
    ) -> bool:
        resolved_url = self._resolve_href_to_url(href)
        if not resolved_url:
            return True

        cached = link_cache.get(resolved_url)
        if cached is not None:
            return cached

        exists = await self._check_remote_url_exists(url=resolved_url, client=client)
        link_cache[resolved_url] = exists
        return exists

    def _resolve_href_to_url(self, href: str) -> str | None:
        normalized_href = href.strip()
        if not normalized_href:
            return None

        lowered = normalized_href.lower()
        if lowered.startswith(("#", "mailto:", "tel:", "javascript:")):
            return None

        parsed = urlparse(normalized_href)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            return None

        return urljoin(f"{self._target_base_url()}/", normalized_href)

    async def _check_remote_url_exists(
        self,
        *,
        url: str,
        client: httpx.AsyncClient,
    ) -> bool:
        head_status = await self._request_status(url=url, client=client, method="HEAD")
        if head_status is not None:
            if head_status in {405, 501}:
                get_status = await self._request_status(url=url, client=client, method="GET")
                if get_status is not None:
                    return self._status_indicates_existing_page(get_status)
            return self._status_indicates_existing_page(head_status)

        get_status = await self._request_status(url=url, client=client, method="GET")
        if get_status is not None:
            return self._status_indicates_existing_page(get_status)

        return False

    async def _request_status(
        self,
        *,
        url: str,
        client: httpx.AsyncClient,
        method: str,
    ) -> int | None:
        try:
            if method == "HEAD":
                response = await client.head(url)
            else:
                response = await client.get(url, headers={"Range": "bytes=0-0"})
            return int(response.status_code)
        except httpx.HTTPError:
            return None

    def _status_indicates_existing_page(self, status_code: int) -> bool:
        if 200 <= status_code < 400:
            return True
        if status_code in {401, 403, 429}:
            return True
        if 500 <= status_code < 600:
            return True
        if status_code in {404, 410}:
            return False
        return False

    def _target_base_url(self) -> str:
        parsed = urlparse(
            self.target_domain
            if self.target_domain.startswith(("http://", "https://"))
            else f"https://{self.target_domain}"
        )
        scheme = parsed.scheme if parsed.scheme in {"http", "https"} else "https"
        netloc = parsed.netloc or parsed.path
        return f"{scheme}://{netloc}".rstrip("/")

    def _normalize_blocks(
        self,
        blocks: list[Any],
        seo_meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
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
                "table_columns": (
                    raw_block.get("table_columns")
                    if isinstance(raw_block.get("table_columns"), list)
                    else []
                ),
                "table_rows": (
                    raw_block.get("table_rows")
                    if isinstance(raw_block.get("table_rows"), list)
                    else []
                ),
                "faq_items": (
                    raw_block.get("faq_items")
                    if isinstance(raw_block.get("faq_items"), list)
                    else []
                ),
                "cta": raw_block.get("cta") if isinstance(raw_block.get("cta"), dict) else None,
                "links": raw_block.get("links") if isinstance(raw_block.get("links"), list) else [],
            }
            if block_type in {"list", "steps"} and not block["items"]:
                block["items"] = self._items_from_table_rows(block["table_rows"])
            if block_type == "faq" and not block["faq_items"]:
                block["faq_items"] = self._faq_items_from_table_rows(block["table_rows"])
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
        seen: set[str] = set()

        def _append(section: str) -> None:
            normalized = section.strip()
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            required.append(normalized)

        for source in (
            brief.get("must_include_sections"),
            brief_delta.get("must_include_sections"),
        ):
            if isinstance(source, list):
                for item in source:
                    if not isinstance(item, str):
                        continue
                    _append(item)

        for section in _as_list(brief.get("outline")):
            if not isinstance(section, dict):
                continue
            level = section.get("level")
            if isinstance(level, int) and level != 2:
                continue
            heading = section.get("heading")
            if isinstance(heading, str):
                _append(heading)

        return required

    def _topic_anchor(self, *, brief: dict[str, Any], document: dict[str, Any] | None) -> str:
        locked_title = str(brief.get("locked_title") or "").strip()
        if locked_title:
            return locked_title

        for candidate in _as_list(brief.get("working_titles")):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        if isinstance(document, dict):
            seo_meta = _as_dict(document.get("seo_meta"))
            h1 = str(seo_meta.get("h1") or "").strip()
            if h1:
                return h1
        return ""

    def _keyword_topic_overlap_ratio(self, primary_keyword: str, topic_anchor: str) -> float:
        keyword_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", primary_keyword.lower())
            if token and token not in _TOPIC_TOKEN_STOPWORDS
        }
        if not keyword_tokens:
            return 1.0

        topic_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", topic_anchor.lower())
            if token and token not in _TOPIC_TOKEN_STOPWORDS
        }
        if not topic_tokens:
            return 1.0

        overlap = keyword_tokens.intersection(topic_tokens)
        return len(overlap) / len(keyword_tokens)

    def _items_from_table_rows(self, table_rows: list[Any]) -> list[str]:
        items: list[str] = []
        for row in table_rows:
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

    def _faq_items_from_table_rows(self, table_rows: list[Any]) -> list[dict[str, str]]:
        faq_items: list[dict[str, str]] = []
        for row in table_rows:
            if not isinstance(row, list) or len(row) < 2:
                continue
            question = str(row[0]).strip()
            answer = str(row[1]).strip()
            if not question:
                continue
            faq_items.append({"question": question, "answer": answer})
        return faq_items

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
        return (
            funnel_stage in {"mofu", "bofu", "transactional", "commercial"}
            or bool(conversion_intents)
        )

    def _slugify(self, value: str) -> str:
        slug = value.lower().strip()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "article"

    def _normalize_domain(self, domain: str) -> str:
        parsed = urlparse(domain if domain.startswith(("http://", "https://")) else f"https://{domain}")
        return parsed.netloc.lower()
