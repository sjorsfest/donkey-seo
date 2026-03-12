"""Content blocks generation agent - generates article content blocks only."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from app.agents.article_writer import ArticleBlockResult
from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ContentBlocksInput(BaseModel):
    """Input for content blocks generation."""

    brief: dict
    writer_instructions: dict
    brief_delta: dict
    brand_context: str = ""
    conversion_intents: list[str] = Field(default_factory=list)
    target_domain: str = ""
    qa_feedback: list[str] = Field(default_factory=list)
    existing_document: dict | None = None


class ContentBlocksOutput(BaseModel):
    """Output from content blocks generator."""

    blocks: list[ArticleBlockResult] = Field(default_factory=list)


class ContentBlocksAgent(BaseAgent[ContentBlocksInput, ContentBlocksOutput]):
    """Generates structured content blocks for articles."""

    model_tier = "reasoning"
    model = "openrouter:google/gemini-3.1-pro-preview"
    temperature = 0.4

    @property
    def system_prompt(self) -> str:
        return """You are an expert SEO content writer producing modular content blocks.

Return ONLY structured JSON with a "blocks" array matching the schema.

Hard requirements:
1. Use block types from the allowed enum only: hero, summary, section, list, comparison_table, steps, faq, cta, conclusion, sources.
2. Use semantic_tag values that match the block intent (header, section, aside, footer, table).
3. Include exactly one hero block with a descriptive heading.
4. Include exactly one summary block with a short, non-empty body (1-2 sentences) that can be reused as an excerpt.
5. Respect forbidden claims and compliance notes from the brief.
6. Follow must-include sections from brief and delta, and cover the outline headings with substantive content.
7. Keep the output conversion-oriented for the funnel stage and conversion intents.
8. Add meaningful internal/external links inside block.links where appropriate.
9. Never output raw HTML in body text.
10. If QA feedback is provided, revise the draft to specifically fix those failures.
11. If an existing document is provided, apply minimal targeted edits instead of rewriting from scratch.
12. Preserve topic, search intent, and primary keyword strategy from the brief.
13. Prioritize the working title/topic; use the primary keyword naturally without forcing off-topic sections.
14. CRITICAL: Naturally incorporate supporting keywords from the brief throughout the article. Use multiple supporting keywords where contextually relevant - they should appear organically in headings, body text, lists, and tables. Do not force them, but ensure several supporting keywords are used across different sections.
15. Never use em dashes (—); use commas, periods, or parentheses instead.
16. If you include source-dependent claims (prices, percentages, statistics, or "according to" statements), add a non-empty sources block that stores citations only in block.links.
17. In sources blocks, every link object MUST include both non-empty "anchor" and "href" values, and source entries must not be placed in items/body text.

Writing quality:
- Clear, practical, and audience-aligned.
- This is SEO-optimized content; apply SEO best practices throughout (keyword placement, semantic richness, headings structure, readability, user intent matching) to maximize search visibility and ranking potential.
- Covers outline and key points from the brief.
- Uses concise paragraphs and scannable sections.
- Keeps claims honest and grounded in provided brand context.
- Avoid heading-only placeholder sections; each section should include useful body content, list items, table rows, or FAQ entries.
- Include one H1-level heading (level 1 is reserved for the page title, so use level 2 for your main heading in the hero block).
"""

    @property
    def output_type(self) -> type[ContentBlocksOutput]:
        return ContentBlocksOutput

    def _build_prompt(self, input_data: ContentBlocksInput) -> str:
        logger.info(
            "Building content blocks prompt",
            extra={
                "primary_keyword": input_data.brief.get("primary_keyword"),
                "funnel_stage": input_data.brief.get("funnel_stage"),
                "qa_feedback_count": len(input_data.qa_feedback),
            },
        )
        conversion_intents = ", ".join(input_data.conversion_intents)
        conversion_intents_text = conversion_intents if conversion_intents else "Not specified"

        supporting_keywords = input_data.brief.get("supporting_keywords", [])
        supporting_keywords_text = (
            "\n".join(f"  - {kw}" for kw in supporting_keywords[:30])
            if supporting_keywords
            else "None provided"
        )

        return (
            "Generate modular article content blocks from these inputs.\n\n"
            "## Content Brief\n"
            f"{json.dumps(input_data.brief, indent=2, ensure_ascii=True)}\n\n"
            "## Supporting Keywords To Incorporate\n"
            "IMPORTANT: Naturally weave these supporting keywords throughout the article content.\n"
            "Use them in headings, body paragraphs, list items, and table content where contextually appropriate.\n"
            "Aim to use multiple different supporting keywords across various sections.\n\n"
            f"{supporting_keywords_text}\n\n"
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
            "Build the content blocks now."
        )
