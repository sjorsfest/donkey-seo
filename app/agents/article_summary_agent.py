"""Article summary agent — generates a short excerpt from completed content blocks."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ArticleSummaryInput(BaseModel):
    """Input for article summary generation."""

    h1: str
    primary_keyword: str
    target_audience: str = ""
    article_content: str = ""


class ArticleSummaryOutput(BaseModel):
    """Output from article summary generator."""

    body: str = Field(description="1-2 sentence article excerpt for use in previews and cards")


class ArticleSummaryAgent(BaseAgent[ArticleSummaryInput, ArticleSummaryOutput]):
    """Generates a short excerpt from an article's title and opening content."""

    model_tier = "standard"
    temperature = 0.3

    @property
    def system_prompt(self) -> str:
        return """Write a concise 1-2 sentence article summary (excerpt) from the provided content.

Requirements:
- 30-60 words
- Captures the article's core value for the target audience
- Includes the primary keyword naturally if it fits without forcing it
- Suitable as a card preview or RSS excerpt
- No em dashes (—); use commas or parentheses instead

Output structured JSON only."""

    @property
    def output_type(self) -> type[ArticleSummaryOutput]:
        return ArticleSummaryOutput

    def _build_prompt(self, input_data: ArticleSummaryInput) -> str:
        parts = [
            f"Article title: {input_data.h1}",
            f"Primary keyword: {input_data.primary_keyword}",
        ]
        if input_data.target_audience:
            parts.append(f"Target audience: {input_data.target_audience}")
        if input_data.article_content:
            parts.append(f"Article content:\n{input_data.article_content}")
        parts.append("\nWrite the summary now.")
        return "\n".join(parts)
