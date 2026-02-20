"""Shared article generation service used by step execution and API endpoints."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from app.agents.article_seo_auditor import ArticleSEOAuditorAgent, ArticleSEOAuditorInput
from app.agents.article_writer import ArticleWriterAgent, ArticleWriterInput
from app.services.content_quality import evaluate_article_quality
from app.services.content_renderer import render_modular_document
from app.services.seo_checklist import DeterministicAuditReport, run_deterministic_checklist

logger = logging.getLogger(__name__)

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
    """Generate article blocks, render HTML, run QA, and apply one SEO revision."""

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
        """Generate article and run hybrid QA/revision workflow."""
        qa_feedback: list[str] = []
        previous_document: dict[str, Any] | None = None
        writer_agent = ArticleWriterAgent()
        seo_auditor = ArticleSEOAuditorAgent()

        pass_fail_thresholds = _as_dict(writer_instructions.get("pass_fail_thresholds"))
        seo_score_target = self._int_threshold(pass_fail_thresholds, "seo_score_target", default=75)
        max_auto_revisions = self._int_threshold(
            pass_fail_thresholds,
            "max_auto_revisions",
            default=1,
        )
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
        total_attempts = max(1, max_auto_revisions + 1)

        required_sections = self._required_sections(brief, brief_delta)
        forbidden_claims = writer_instructions.get("forbidden_claims") or []
        min_internal_links = self._min_link_count(writer_instructions, "min_internal")
        min_external_links = self._min_link_count(writer_instructions, "min_external")
        require_cta = self._cta_required(brief, conversion_intents)
        first_party_domain = self._normalize_domain(self.target_domain)

        for attempt in range(total_attempts):
            output = await writer_agent.run(
                input_data=ArticleWriterInput(
                    brief=brief,
                    writer_instructions=writer_instructions,
                    brief_delta=brief_delta,
                    brand_context=brand_context,
                    conversion_intents=conversion_intents,
                    target_domain=self.target_domain,
                    qa_feedback=qa_feedback,
                    existing_document=previous_document,
                )
            )
            document = self._normalize_document(
                output.document.model_dump(),
                brief,
                writer_instructions,
                conversion_intents,
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
                page_type=brief.get("page_type"),
                search_intent=brief.get("search_intent"),
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

            hard_failures = _as_list(seo_audit.get("hard_failures"))
            overall_score = int(seo_audit.get("overall_score") or 0)
            should_retry = (
                attempt < total_attempts - 1
                and (len(hard_failures) > 0 or overall_score < seo_score_target)
            )

            if not should_retry:
                status = "needs_review" if hard_failures else "draft"
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
                    status=status,
                    generation_model=getattr(writer_agent, "_model", None),
                    generation_temperature=writer_agent.temperature,
                )

            qa_feedback = revision_feedback
            previous_document = document

        raise RuntimeError("Unreachable article generation state")

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
        search_intent = str(brief.get("search_intent") or "informational").strip()
        target_audience = str(brief.get("target_audience") or "").strip()
        if keyword:
            feedback.append(f"Preserve primary keyword strategy for '{keyword}'.")
        feedback.append(f"Preserve search intent: '{search_intent}'.")
        if target_audience:
            feedback.append(f"Preserve ICP hook for audience: '{target_audience}'.")
        feedback.append(
            "Apply minimal targeted edits; keep existing structure unless "
            "a failing check requires change."
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
        normalized: dict[str, Any] = {
            "schema_version": "1.0",
            "seo_meta": seo_meta,
            "conversion_plan": conversion_plan,
            "blocks": blocks,
        }

        primary_keyword = str(brief.get("primary_keyword") or "")
        if not seo_meta.get("primary_keyword"):
            seo_meta["primary_keyword"] = primary_keyword

        if not seo_meta.get("h1"):
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
        return normalized

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
