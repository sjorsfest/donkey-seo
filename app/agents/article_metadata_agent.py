"""Article metadata generation agent - generates SEO and conversion metadata from content blocks."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from app.agents.article_writer import ArticleBlockResult, ArticleSEOMeta, ConversionPlan
from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ArticleMetadataInput(BaseModel):
    """Input for metadata generation."""

    blocks: list[ArticleBlockResult] = Field(default_factory=list)
    brief: dict
    conversion_intents: list[str] = Field(default_factory=list)
    target_domain: str = ""


class ArticleMetadataOutput(BaseModel):
    """Output from metadata generator."""

    seo_meta: ArticleSEOMeta
    conversion_plan: ConversionPlan


class ArticleMetadataAgent(BaseAgent[ArticleMetadataInput, ArticleMetadataOutput]):
    """Generates SEO metadata and conversion plan from article content blocks."""

    model_tier = "standard"
    model = "openrouter:minimax/minimax-m2.5"
    temperature = 0.3

    @property
    def system_prompt(self) -> str:
        return """You are an SEO metadata specialist. Generate optimal SEO metadata and conversion strategy from article content.

Return ONLY structured JSON matching the schema.

Requirements:

SEO Metadata (seo_meta):
1. h1: Extract the main H1 heading from the hero or first major block. Should match the article's primary topic.
2. meta_title: Craft a compelling, SEO-optimized title (50-60 characters). Include the primary keyword naturally.
3. meta_description: Write a compelling meta description (150-160 characters) that summarizes the article and encourages clicks. Include primary keyword.
4. slug: Generate a clean, keyword-rich URL slug (lowercase, hyphens, no special characters). Based on the h1/topic.
5. primary_keyword: Identify the primary keyword target from the brief and content.

Conversion Plan (conversion_plan):
1. primary_intent: Identify the main conversion goal (e.g., "lead generation", "product awareness", "comparison", "tutorial completion").
2. cta_strategy: List 2-4 strategic CTA placements or approaches based on the funnel stage and conversion intents (e.g., ["early soft CTA after introduction", "comparison table with sign-up link", "final strong CTA in conclusion"]).

Quality guidelines:
- Ensure meta_title and meta_description are compelling and click-worthy
- Keep slug concise and readable
- Match tone and intent to the funnel stage
- Align conversion plan with the content structure and flow
"""

    @property
    def output_type(self) -> type[ArticleMetadataOutput]:
        return ArticleMetadataOutput

    def _build_prompt(self, input_data: ArticleMetadataInput) -> str:
        logger.info(
            "Building article metadata prompt",
            extra={
                "blocks_count": len(input_data.blocks),
                "primary_keyword": input_data.brief.get("primary_keyword"),
            },
        )

        # Extract key content snippets for context
        block_summaries = []
        for i, block in enumerate(input_data.blocks[:10]):  # First 10 blocks for context
            summary = f"Block {i+1} ({block.block_type}): "
            if block.heading:
                summary += f"'{block.heading}'"
            if block.body:
                body_preview = block.body[:100] + "..." if len(block.body) > 100 else block.body
                summary += f" - {body_preview}"
            block_summaries.append(summary)

        blocks_context = "\n".join(block_summaries)
        conversion_intents = ", ".join(input_data.conversion_intents)
        conversion_intents_text = conversion_intents if conversion_intents else "Not specified"

        return (
            "Generate SEO metadata and conversion plan for this article.\n\n"
            "## Content Blocks Summary\n"
            f"{blocks_context}\n\n"
            "## Full Blocks (for detailed analysis)\n"
            f"{json.dumps([block.model_dump() for block in input_data.blocks], indent=2, ensure_ascii=True)}\n\n"
            "## Content Brief\n"
            f"{json.dumps(input_data.brief, indent=2, ensure_ascii=True)}\n\n"
            "## Conversion Intents\n"
            f"{conversion_intents_text}\n\n"
            "## Target Domain\n"
            f"{input_data.target_domain or 'Not specified'}\n\n"
            "Generate the SEO metadata and conversion plan now."
        )
