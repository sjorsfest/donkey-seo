"""Article writer agent for Step 14 modular content generation."""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

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


class ArticleWriterInput(BaseModel):
    """Input for article generation."""

    brief: dict
    writer_instructions: dict
    brief_delta: dict
    brand_context: str = ""
    conversion_intents: list[str] = Field(default_factory=list)
    target_domain: str = ""
    qa_feedback: list[str] = Field(default_factory=list)


class ArticleWriterOutput(BaseModel):
    """Output from article generator."""

    document: ArticleDocumentResult


class ArticleWriterAgent(BaseAgent[ArticleWriterInput, ArticleWriterOutput]):
    """Generates structured modular article blocks from brief artifacts."""

    model_tier = "reasoning"
    temperature = 0.4

    @property
    def system_prompt(self) -> str:
        return """You are an expert SEO content writer producing CMS-agnostic modular content.

Return ONLY structured JSON matching the schema.

Hard requirements:
1. Use block types from the allowed enum only.
2. Use semantic_tag values that match the block intent.
3. Include exactly one hero block and one H1.
4. Respect forbidden claims and compliance notes.
5. Follow must-include sections from brief and delta.
6. Keep the output conversion-oriented for funnel stage and conversion intents.
7. Add meaningful internal/external links inside block.links where appropriate.
8. Never output raw HTML in body text.
9. If QA feedback is provided, revise the draft to specifically fix those failures.

Writing quality:
- Clear, practical, and audience-aligned.
- Covers outline and key points from the brief.
- Uses concise paragraphs and scannable sections.
- Keeps claims honest and grounded in provided brand context.
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
            f"{', '.join(input_data.conversion_intents) if input_data.conversion_intents else 'Not specified'}\n\n"
            "## Target Domain\n"
            f"{input_data.target_domain or 'Not specified'}\n\n"
            "## QA Feedback To Fix\n"
            f"{json.dumps(input_data.qa_feedback, ensure_ascii=True)}\n\n"
            "Build the final modular article now."
        )
