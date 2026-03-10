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
    """Orchestrates article generation using ContentBlocksAgent and ArticleMetadataAgent.

    This agent coordinates two specialized agents:
    1. ContentBlocksAgent - generates article content blocks
    2. ArticleMetadataAgent - generates SEO metadata and conversion plan
    """

    model_tier = "reasoning"
    temperature = 0.4

    def __init__(self, model_override: str | None = None) -> None:
        super().__init__(model_override)
        # Import here to avoid circular dependencies
        from app.agents.article_metadata_agent import ArticleMetadataAgent, ArticleMetadataInput
        from app.agents.content_blocks_agent import ContentBlocksAgent, ContentBlocksInput

        self._content_blocks_agent = ContentBlocksAgent(model_override)
        self._metadata_agent = ArticleMetadataAgent(model_override)
        self._ContentBlocksInput = ContentBlocksInput
        self._ArticleMetadataInput = ArticleMetadataInput

    @property
    def system_prompt(self) -> str:
        # Not used since we orchestrate sub-agents, but required by base class
        return ""

    @property
    def output_type(self) -> type[ArticleWriterOutput]:
        return ArticleWriterOutput

    def _build_prompt(self, input_data: ArticleWriterInput) -> str:
        # Not used since we orchestrate sub-agents, but required by base class
        return ""

    async def run(
        self,
        input_data: ArticleWriterInput,
        context: dict | None = None,
    ) -> ArticleWriterOutput:
        """Run the orchestrated article generation.

        Steps:
        1. Generate content blocks using ContentBlocksAgent
        2. Generate metadata using ArticleMetadataAgent
        3. Combine into final ArticleDocumentResult
        """
        logger.info(
            "Starting orchestrated article generation",
            extra={
                "primary_keyword": input_data.brief.get("primary_keyword"),
                "funnel_stage": input_data.brief.get("funnel_stage"),
                "qa_feedback_count": len(input_data.qa_feedback),
            },
        )

        # Step 1: Generate content blocks
        logger.info("Step 1: Generating content blocks")
        content_input = self._ContentBlocksInput(
            brief=input_data.brief,
            writer_instructions=input_data.writer_instructions,
            brief_delta=input_data.brief_delta,
            brand_context=input_data.brand_context,
            conversion_intents=input_data.conversion_intents,
            target_domain=input_data.target_domain,
            qa_feedback=input_data.qa_feedback,
            existing_document=input_data.existing_document,
        )
        content_output = await self._content_blocks_agent.run(content_input, context)

        logger.info(
            "Content blocks generated",
            extra={"blocks_count": len(content_output.blocks)},
        )

        # Step 2: Generate metadata from blocks
        logger.info("Step 2: Generating SEO metadata and conversion plan")
        metadata_input = self._ArticleMetadataInput(
            blocks=content_output.blocks,
            brief=input_data.brief,
            conversion_intents=input_data.conversion_intents,
            target_domain=input_data.target_domain,
        )
        metadata_output = await self._metadata_agent.run(metadata_input, context)

        logger.info(
            "Metadata generated",
            extra={
                "h1": metadata_output.seo_meta.h1,
                "primary_keyword": metadata_output.seo_meta.primary_keyword,
            },
        )

        # Step 3: Combine into final document
        document = ArticleDocumentResult(
            schema_version="1.0",
            seo_meta=metadata_output.seo_meta,
            conversion_plan=metadata_output.conversion_plan,
            blocks=content_output.blocks,
        )

        logger.info("Article generation orchestration complete")
        return ArticleWriterOutput(document=document)
