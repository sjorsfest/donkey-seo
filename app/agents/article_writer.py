"""Article writer agent for Step 14 modular content generation."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

_SOURCE_REQUIRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[$€£]\s?\d", re.IGNORECASE),
    re.compile(r"\b(?:usd|eur|gbp)\s?\d", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s?%", re.IGNORECASE),
    re.compile(r"\b(?:according to|study|research|survey|report|benchmark|dataset|data)\b", re.IGNORECASE),
)

BlockType = Literal[
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
]
SemanticTag = Literal["header", "section", "aside", "footer", "table"]


class BlockLink(BaseModel):
    """Hyperlink attached to a content block."""

    anchor: str = Field(description="Anchor text")
    href: str = Field(description="URL or path")


class FAQItem(BaseModel):
    """FAQ question and answer."""

    question: str
    answer: str


class CTAData(BaseModel):
    """Call-to-action payload."""

    label: str
    href: str


class ArticleBlockResult(BaseModel):
    """Single semantic block in the modular article document."""

    block_type: BlockType
    semantic_tag: SemanticTag
    heading: str | None = None
    level: int | None = Field(default=None, ge=2, le=4)
    body: str | None = None
    items: list[str] = Field(default_factory=list)
    ordered: bool = False
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    faq_items: list[FAQItem] = Field(default_factory=list)
    cta: CTAData | None = None
    links: list[BlockLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_summary_body(self) -> "ArticleBlockResult":
        if self.block_type != "summary":
            return self
        if not (self.body or "").strip():
            raise ValueError(
                "summary blocks must include a short non-empty body so it can be reused as an excerpt"
            )
        return self


class ArticleSEOMeta(BaseModel):
    """Top-level SEO metadata for the generated article."""

    h1: str
    meta_title: str
    meta_description: str
    slug: str
    primary_keyword: str


class ConversionPlan(BaseModel):
    """High-level conversion plan for CTAs and narrative intent."""

    primary_intent: str
    cta_strategy: list[str] = Field(default_factory=list)


class ArticleDocumentResult(BaseModel):
    """CMS-agnostic modular content contract."""

    schema_version: str = "1.0"
    seo_meta: ArticleSEOMeta
    conversion_plan: ConversionPlan
    blocks: list[ArticleBlockResult] = Field(default_factory=list)

    @staticmethod
    def _block_text(block: ArticleBlockResult) -> str:
        values: list[str] = []
        if block.heading:
            values.append(block.heading)
        if block.body:
            values.append(block.body)
        values.extend(str(item) for item in block.items)
        values.extend(str(item) for item in block.table_columns)
        for row in block.table_rows:
            values.extend(str(cell) for cell in row)
        for item in block.faq_items:
            values.append(item.question)
            values.append(item.answer)
        return "\n".join(values)

    @staticmethod
    def _needs_sources(text: str) -> bool:
        return any(pattern.search(text) for pattern in _SOURCE_REQUIRED_PATTERNS)

    @staticmethod
    def _has_source_entries(block: ArticleBlockResult) -> bool:
        if (block.body or "").strip():
            return True
        if any(str(item).strip() for item in block.items):
            return True
        if any((link.href or "").strip() for link in block.links):
            return True
        return False

    @model_validator(mode="after")
    def _require_sources_for_data_claims(self) -> "ArticleDocumentResult":
        sources_blocks = [block for block in self.blocks if block.block_type == "sources"]
        needs_sources = any(
            self._needs_sources(self._block_text(block))
            for block in self.blocks
            if block.block_type != "sources"
        )

        if needs_sources and not sources_blocks:
            raise ValueError(
                "include a sources block when using source-dependent claims (for example prices, percentages, or attributed data claims)"
            )
        if needs_sources and not any(self._has_source_entries(block) for block in sources_blocks):
            raise ValueError(
                "sources block cannot be empty when source-dependent claims are present"
            )
        return self


class ArticleWriterInput(BaseModel):
    """Input for article generation."""

    brief: dict
    writer_instructions: dict
    brief_delta: dict
    brand_context: str = ""
    conversion_intents: list[str] = Field(default_factory=list)
    target_domain: str = ""
    qa_feedback: list[str] = Field(default_factory=list)
    existing_document: dict | None = None


class ArticleWriterOutput(BaseModel):
    """Output from article generator."""

    document: ArticleDocumentResult


class ArticleWriterAgent(BaseAgent[ArticleWriterInput, ArticleWriterOutput]):
    """Generates structured modular article blocks from brief artifacts."""

    model_tier = "reasoning"
    model = "openrouter:anthropic/claude-sonnet-4.6"
    temperature = 0.4

    @property
    def system_prompt(self) -> str:
        return """You are an expert SEO content writer producing CMS-agnostic modular content.

Return ONLY structured JSON matching the schema.

Hard requirements:
1. Use block types from the allowed enum only.
2. Use semantic_tag values that match the block intent.
3. Include exactly one hero block and one H1.
4. Include exactly one summary block with a short, non-empty body (1-2 sentences) that can be reused as an excerpt.
5. Respect forbidden claims and compliance notes.
6. Follow must-include sections from brief and delta, and cover the outline headings with substantive content.
7. Keep the output conversion-oriented for funnel stage and conversion intents.
8. Add meaningful internal/external links inside block.links where appropriate.
9. Never output raw HTML in body text.
10. If QA feedback is provided, revise the draft to specifically fix those failures.
11. If an existing document is provided, apply minimal targeted edits
    instead of rewriting from scratch.
12. Preserve topic, search intent, primary keyword strategy, and ICP hook.
13. Prioritize the working title/topic; use the primary keyword naturally without forcing off-topic sections.
14. Never use em dashes (—); use commas, periods, or parentheses instead.
15. If you include source-dependent claims (for example prices, percentages, statistics, or "according to" statements), add a non-empty sources block that lists the supporting sources.

Writing quality:
- Clear, practical, and audience-aligned.
- Covers outline and key points from the brief.
- Uses concise paragraphs and scannable sections.
- Keeps claims honest and grounded in provided brand context.
- Avoid heading-only placeholder sections; each section should include useful body content, list items, table rows, or FAQ entries.
"""

    @property
    def output_type(self) -> type[ArticleWriterOutput]:
        return ArticleWriterOutput

    def _build_prompt(self, input_data: ArticleWriterInput) -> str:
        logger.info(
            "Building article writer prompt",
            extra={
                "primary_keyword": input_data.brief.get("primary_keyword"),
                "funnel_stage": input_data.brief.get("funnel_stage"),
                "qa_feedback_count": len(input_data.qa_feedback),
            },
        )
        conversion_intents = ", ".join(input_data.conversion_intents)
        conversion_intents_text = conversion_intents if conversion_intents else "Not specified"

        return (
            "Generate a complete modular article document from these inputs.\n\n"
            "## Content Brief\n"
            f"{json.dumps(input_data.brief, indent=2, ensure_ascii=True)}\n\n"
            "## Writer Instructions\n"
            f"{json.dumps(input_data.writer_instructions, indent=2, ensure_ascii=True)}\n\n"
            "## Brief Delta\n"
            f"{json.dumps(input_data.brief_delta, indent=2, ensure_ascii=True)}\n\n"
            "## Brand Context\n"
            f"{input_data.brand_context or 'Not provided'}\n\n"
            "## Conversion Intents\n"
            f"{conversion_intents_text}\n\n"
            "## Target Domain\n"
            f"{input_data.target_domain or 'Not specified'}\n\n"
            "## Non-Negotiable Style Rule\n"
            "Never use em dashes (—). Use commas, periods, or parentheses instead.\n\n"
            "## QA Feedback To Fix\n"
            f"{json.dumps(input_data.qa_feedback, ensure_ascii=True)}\n\n"
            "## Existing Document (for targeted revision)\n"
            f"{json.dumps(input_data.existing_document, indent=2, ensure_ascii=True)}\n\n"
            "Build the final modular article now."
        )
