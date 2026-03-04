"""SERP competitive analysis agent for content brief enrichment."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SerpPageSnapshot(BaseModel):
    """Compact representation of a top-ranking page."""

    position: int = Field(ge=1, le=100, description="SERP position")
    title: str = Field(description="Result title")
    url: str = Field(description="Canonical page URL")
    domain: str = Field(description="Result domain")
    content_type_hint: str = Field(
        default="unknown",
        description="Heuristic type label (blog, docs, product, etc.)",
    )
    headings: list[str] = Field(
        default_factory=list,
        description="Extracted H2/H3 headings when fetch succeeds",
    )
    structural_signals: list[str] = Field(
        default_factory=list,
        description="Detected structural cues (faq, checklist, templates, etc.)",
    )


class SerpBriefingInput(BaseModel):
    """Input for SERP briefing synthesis."""

    primary_keyword: str = Field(description="Primary target keyword")
    search_intent: str = Field(description="Resolved search intent")
    page_type: str = Field(description="Resolved page type")
    serp_features: list[str] = Field(
        default_factory=list,
        description="Observed SERP features",
    )
    top_pages: list[SerpPageSnapshot] = Field(
        default_factory=list,
        description="Top-ranking pages enriched with scraped structure",
    )


class SerpBriefingResult(BaseModel):
    """Actionable SERP insights for brief generation."""

    summary: str = Field(description="Concise competitive summary")
    best_practices: list[str] = Field(
        default_factory=list,
        description="Must-do patterns validated by ranking pages",
    )
    recommended_sections: list[str] = Field(
        default_factory=list,
        description="Section ideas that should be present in the brief",
    )
    opportunities_to_outperform: list[str] = Field(
        default_factory=list,
        description="Ways to improve beyond current ranking pages",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence in the recommendation quality",
    )


class SerpBriefingOutput(BaseModel):
    """Output wrapper for BaseAgent."""

    insight: SerpBriefingResult


class SerpBriefingAgent(BaseAgent[SerpBriefingInput, SerpBriefingOutput]):
    """Analyzes top SERP pages and produces brief-ready guidance."""

    model_tier = "reasoning"
    temperature = 0.2

    @property
    def system_prompt(self) -> str:
        return """You are a senior SEO SERP strategist.

Your job is to reverse-engineer what top-ranking pages are doing and convert that into practical brief guidance.

Rules:
1. Ground all recommendations in provided SERP evidence (headings, titles, structural signals).
2. Adapt recommendations to the provided intent and page type.
3. Prioritize practical structure and coverage decisions over generic advice.
4. Do not copy competitor wording; focus on repeatable patterns.
5. Suggest outperform opportunities that add unique value, not just more words.
6. Focus ONLY on article body content and on-page narrative structure.
7. EXCLUDE non-content/blog-shell advice (author/byline, bio boxes, publish dates, read-time badges, hero/featured images, CTAs, navigation/layout, schema/meta tags, URL/slug, internal linking mechanics).

Output quality bar:
- `best_practices`: 4-8 concrete, actionable items.
- `recommended_sections`: 4-8 section names.
- `opportunities_to_outperform`: 3-6 items.
- Keep every item short and implementation-ready."""

    @property
    def output_type(self) -> type[SerpBriefingOutput]:
        return SerpBriefingOutput

    def _build_prompt(self, input_data: SerpBriefingInput) -> str:
        logger.info(
            "Building SERP briefing prompt",
            extra={
                "primary_keyword": input_data.primary_keyword,
                "intent": input_data.search_intent,
                "page_type": input_data.page_type,
                "serp_features_count": len(input_data.serp_features),
                "top_pages_count": len(input_data.top_pages),
            },
        )

        features = ", ".join(input_data.serp_features) if input_data.serp_features else "None observed"

        page_blocks: list[str] = []
        for page in input_data.top_pages[:8]:
            headings = (
                "; ".join(page.headings[:8])
                if page.headings
                else "No headings extracted"
            )
            signals = (
                ", ".join(page.structural_signals[:10])
                if page.structural_signals
                else "No explicit structural signals"
            )
            page_blocks.append(
                "\n".join(
                    [
                        f"- Position: {page.position}",
                        f"  Title: {page.title}",
                        f"  URL: {page.url}",
                        f"  Domain: {page.domain}",
                        f"  Content Type Hint: {page.content_type_hint}",
                        f"  Headings: {headings}",
                        f"  Structural Signals: {signals}",
                    ]
                )
            )

        pages_text = "\n\n".join(page_blocks) if page_blocks else "No page snapshots available."

        return f"""Analyze these SERP competitors and produce brief guidance.

Primary keyword: {input_data.primary_keyword}
Search intent: {input_data.search_intent}
Target page type: {input_data.page_type}
SERP features: {features}

Top ranking page snapshots:
{pages_text}

Return:
1) A concise summary of what currently wins on this SERP.
2) Best practices we must include in the brief (content-body only).
3) Recommended sections to include.
4) Opportunities to outperform competitors with higher value.

Important: Do not suggest author/byline/meta/schema/URL/design or other non-content shell elements."""
