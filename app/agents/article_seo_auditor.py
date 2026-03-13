"""LLM SEO auditor for checklist-based article QA."""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


ChecklistStatus = Literal["pass", "fail", "warning"]
ChecklistSeverity = Literal["hard", "soft"]
ClaimClassification = Literal["fact", "consensus", "logical", "opinion", "speculation"]

# Canonical LLM check IDs grouped by scope — used to build per-audit allowed ID list
_BASE_CHECK_IDS: tuple[str, ...] = (
    "base.intent_alignment",
    "base.structure",
    "base.intro_quality",
    "base.depth_completeness",
    "base.clarity_readability",
    "base.logical_flow",
    "base.keyword_integration",
    "base.claim_integrity",
    "base.conclusion_quality",
)

_MODULE_CHECK_IDS: dict[str, tuple[str, ...]] = {
    "A": (
        "module_a.query_answered",
        "module_a.sub_topics_covered",
        "module_a.appropriate_reading_level",
    ),
    "B": (
        "module_b.decision_support",
        "module_b.alternatives_fair",
        "module_b.product_claim_accuracy",
        "module_b.cta_buyer_intent",
    ),
    "C": (
        "module_c.icp_resonance",
        "module_c.value_proposition_early",
        "module_c.social_proof_present",
        "module_c.cta_strength",
        "module_c.feature_benefit_framing",
    ),
    "D": (
        "module_d.thesis_stated",
        "module_d.originality",
        "module_d.argumentation_supported",
        "module_d.counterargument_acknowledged",
    ),
}

_RISK_CHECK_IDS: tuple[str, ...] = (
    "risk.absolute_claims",
    "risk.hedging_language",
    "risk.professional_advice_disclaimer",
)

_MODULE_RUBRICS: dict[str, str] = {
    "A": """Module A — Informational (guide, how-to, glossary, use-case, checklist)

Check IDs to evaluate (use these exact IDs):
- module_a.query_answered: Does the article fully answer the stated search query without assuming prior knowledge?
- module_a.sub_topics_covered: Are all key sub-topics the reader would expect covered at adequate depth?
- module_a.appropriate_reading_level: Is the language clear and accessible for the target audience?

Also evaluate all BASE checks (base.*) with attention to:
- Completeness: no major gaps a reader searching this query would notice
- Intro quality: states what the reader will learn within the first paragraph
- Conclusion: summarises value and provides a clear next step""",

    "B": """Module B — Commercial Comparison (comparison, alternatives, list, best-x-for-y, statistics)

Check IDs to evaluate (use these exact IDs):
- module_b.decision_support: Does the article help the reader make a buying or adoption decision?
- module_b.alternatives_fair: Are competing options represented fairly, without misleading omissions?
- module_b.product_claim_accuracy: Are product feature and pricing claims specific and verifiable (no vague superlatives)?
- module_b.cta_buyer_intent: Does the CTA match the buyer-intent stage (e.g. trial, demo, comparison)?

Also evaluate all BASE checks (base.*) with attention to:
- Structure: comparison tables or structured pros/cons are present and accurate
- Recency: features, prices, and rankings are described as current""",

    "C": """Module C — SaaS Product-Led (landing, tool, calculator, template)

Check IDs to evaluate (use these exact IDs):
- module_c.icp_resonance: Does the headline and opening speak directly to the ICP's pain point from the brief?
- module_c.value_proposition_early: Is the product's core benefit clear within the first 100 words?
- module_c.social_proof_present: Are testimonials, case studies, or usage numbers present?
- module_c.cta_strength: Is the primary CTA specific, prominent, and action-oriented?
- module_c.feature_benefit_framing: Are features framed as outcomes/benefits rather than just listed?

Also evaluate all BASE checks (base.*) with attention to:
- ICP hook from the brief: the opening must address the ICP's stated pain, not generic claims
- Trust signals: security, compliance, or credibility markers appropriate to the ICP""",

    "D": """Module D — Thought Leadership (opinion, thought-leadership, editorial)

Check IDs to evaluate (use these exact IDs):
- module_d.thesis_stated: Is a clear, defensible thesis stated early in the article?
- module_d.originality: Does the piece bring a new angle, proprietary data, or insight not widely available?
- module_d.argumentation_supported: Are claims backed by evidence, data, or sound reasoning?
- module_d.counterargument_acknowledged: Does the article acknowledge and address the strongest counterargument?

Also evaluate all BASE checks (base.*) with attention to:
- Author voice: consistent, authoritative tone throughout
- Actionability: reader leaves with an insight or framing they can apply""",
}

_RISK_MODULE_RULES = """Risk Sensitivity Module — Active

This article covers a risk-sensitive topic (health, finance, legal, security, AI risk, personal data, or compliance).

Check IDs to evaluate (use these exact IDs):
- risk.absolute_claims: Flag any use of absolute language: "guarantee/guaranteed", "100%", "always", "never", "risk-free".
  These are hard failures when used to describe product outcomes or safety.
- risk.hedging_language: Verify that uncertain or predictive claims use appropriate hedging ("may", "can", "in some cases",
  "results vary", etc.). Missing hedging on speculative claims is a soft warning.
- risk.professional_advice_disclaimer: Check that the article does not present content as a substitute for professional
  advice (medical, legal, financial) without an appropriate disclaimer. Missing disclaimer is a hard failure."""

_BASE_FRAMEWORK = """BASE Framework — apply to every audit

B — Brand & Intent alignment
  Check id: base.intent_alignment
  Does the article topic, search intent, and primary keyword match the brief? Does it address the ICP described in the brief?

A — Architecture
  Check id: base.structure
  Is heading hierarchy logical (H1 → H2 → H3 without skips)? Does the article have a clear intro, body, and conclusion?
  Check id: base.intro_quality
  Does the intro hook the reader and state what they will learn? Is the primary keyword present naturally in the opening?
  Check id: base.conclusion_quality
  Does the conclusion summarise the core value and provide a clear next step or call to action?

S — Substance
  Check id: base.depth_completeness
  Does the article cover the topic with sufficient depth? Are sections padded with filler or genuinely informative?
  Check id: base.claim_integrity
  Are factual claims accurate and attributed? Are speculative or opinion claims labelled as such?
  Check id: base.logical_flow
  Do sections connect coherently? Are there abrupt topic shifts or non-sequiturs?

E — Expression
  Check id: base.clarity_readability
  Are sentences clear and concise? Is the reading level appropriate for the audience in the brief?
  Check id: base.keyword_integration
  Is the primary keyword used naturally in the text? Flag stuffing (forced repetition) or avoidance (keyword absent from key positions)."""


class SEOChecklistItem(BaseModel):
    """Single checklist finding from the SEO auditor."""

    id: str
    status: ChecklistStatus
    severity: ChecklistSeverity
    evidence: str = ""
    fix_instruction: str = ""


class ClaimIntegrityItem(BaseModel):
    """Claim classification and rationale."""

    claim: str
    classification: ClaimClassification
    rationale: str = ""


class ArticleSEOAuditorInput(BaseModel):
    """Input payload for article SEO auditing."""

    brief: dict
    writer_instructions: dict
    document: dict
    deterministic_audit: dict
    content_type_module: str
    risk_module_applied: bool
    blueprint_key: str = ""


class ArticleSEOAuditorOutput(BaseModel):
    """Structured SEO audit output."""

    checklist_items: list[SEOChecklistItem] = Field(default_factory=list)
    claim_integrity: list[ClaimIntegrityItem] = Field(default_factory=list)
    overall_score: int = Field(ge=0, le=100, default=75)
    hard_failures: list[str] = Field(default_factory=list)
    soft_warnings: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)


class ArticleSEOAuditorAgent(BaseAgent[ArticleSEOAuditorInput, ArticleSEOAuditorOutput]):
    """Evaluate generated article quality against SEO checklist modules."""

    model_tier = "reasoning"
    model = "openrouter:anthropic/claude-4.5-sonnet"
    temperature = 0.2

    @property
    def system_prompt(self) -> str:
        return f"""You are an SEO content quality controller.

Audit the article using the BASE framework and the selected content-type module rubric,
both provided in the user message. If risk_module_applied is true, also apply the risk
sensitivity module rules provided.

Rules:
- Use only the canonical check IDs defined in the rubrics. Do not invent new IDs.
- Preserve topic, search intent, primary keyword strategy, and ICP hook from the brief.
- Do not suggest fluff or broad rewrites unrelated to audit findings.
- Do not fabricate sources or claims.
- Mark only high-confidence blocking issues as severity=hard.
- Keep revision instructions concise and directly actionable.
- ICP and brand voice context is in the Content Brief — use it when evaluating resonance checks.
- Only use the canonical check IDs listed in the prompt. Do not invent new IDs.

Output structured JSON only."""

    @property
    def output_type(self) -> type[ArticleSEOAuditorOutput]:
        return ArticleSEOAuditorOutput

    def _build_prompt(self, input_data: ArticleSEOAuditorInput) -> str:
        logger.info(
            "Building article SEO auditor prompt",
            extra={
                "content_type_module": input_data.content_type_module,
                "risk_module_applied": input_data.risk_module_applied,
                "blueprint_key": input_data.blueprint_key,
            },
        )

        module_rubric = _MODULE_RUBRICS.get(
            input_data.content_type_module, _MODULE_RUBRICS["A"]
        )

        risk_block = ""
        if input_data.risk_module_applied:
            risk_block = f"## Risk Module Rules\n{_RISK_MODULE_RULES}\n\n"

        blueprint_block = ""
        if input_data.blueprint_key:
            from app.services.blueprints import BLUEPRINT_REGISTRY

            blueprint = BLUEPRINT_REGISTRY.get(input_data.blueprint_key)
            if blueprint:
                quality_rules = "\n".join(f"- {r}" for r in blueprint.quality_rules)
                common_mistakes = "\n".join(f"- {m}" for m in blueprint.common_mistakes)
                blueprint_block = (
                    f"## Blueprint Quality Rules ({input_data.blueprint_key})\n"
                    f"{quality_rules}\n\n"
                    f"## Blueprint Common Mistakes to Flag\n"
                    f"{common_mistakes}\n\n"
                )

        active_ids = (
            _BASE_CHECK_IDS
            + _MODULE_CHECK_IDS.get(input_data.content_type_module, _MODULE_CHECK_IDS["A"])
            + (_RISK_CHECK_IDS if input_data.risk_module_applied else ())
        )
        allowed_ids_block = "## Allowed Check IDs\n" + "\n".join(
            f"- {cid}" for cid in active_ids
        ) + "\n\n"

        return (
            "Audit this generated article and return structured findings.\n\n"
            f"## BASE Framework\n{_BASE_FRAMEWORK}\n\n"
            f"## Content-Type Module Rubric\n{module_rubric}\n\n"
            f"{risk_block}"
            f"{allowed_ids_block}"
            f"{blueprint_block}"
            "## Content Brief\n"
            f"{json.dumps(input_data.brief, indent=2, ensure_ascii=True)}\n\n"
            "## Writer Instructions\n"
            f"{json.dumps(input_data.writer_instructions, indent=2, ensure_ascii=True)}\n\n"
            "## Deterministic Audit Report\n"
            f"{json.dumps(input_data.deterministic_audit, indent=2, ensure_ascii=True)}\n\n"
            "## Article Document\n"
            f"{json.dumps(input_data.document, indent=2, ensure_ascii=True)}\n\n"
            "Return checklist findings, claim integrity classifications, a 0-100 score, "
            "hard/soft issue lists, and short revision instructions."
        )
