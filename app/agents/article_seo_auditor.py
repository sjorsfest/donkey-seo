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
    temperature = 0.2

    @property
    def system_prompt(self) -> str:
        return """You are an SEO content quality controller.

Audit the article against:
1. BASE framework (intent/focus, structure, intro quality, depth/completeness,
   clarity/readability, logical flow, natural keyword integration, claim integrity,
   conclusion quality).
2. Selected content-type module (A informational, B commercial comparison,
   C SaaS product-led, D thought leadership).
3. Risk sensitivity module if risk_module_applied is true.

Rules:
- Preserve topic, search intent, primary keyword strategy, and ICP hook.
- Do not suggest fluff or broad rewrites unrelated to audit findings.
- Do not fabricate sources or claims.
- Mark only high-confidence blocking issues as severity=hard.
- Keep revision instructions concise and directly actionable.

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
            },
        )
        return (
            "Audit this generated article and return structured findings.\n\n"
            "## Content-Type Module\n"
            f"{input_data.content_type_module}\n\n"
            "## Risk Module Applied\n"
            f"{input_data.risk_module_applied}\n\n"
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
