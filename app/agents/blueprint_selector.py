"""Lightweight agent for selecting page type blueprint, content role, and pillar."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class BlueprintSelectorInput(BaseModel):
    """Input for the blueprint selector agent."""

    topic_name: str = Field(description="Name of the topic cluster")
    primary_keyword: str = Field(description="Primary target keyword")
    search_intent: str = Field(description="Search intent from discovery (informational, commercial, transactional)")
    discovery_page_type: str = Field(description="Page type suggested by discovery pipeline")
    funnel_stage: str = Field(description="Funnel stage: tofu, mofu, or bofu")
    supporting_keywords: list[str] = Field(
        default_factory=list,
        description="Top supporting keywords for context",
    )
    available_blueprints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Available blueprint options with key, label, description",
    )


class BlueprintSelectorOutput(BaseModel):
    """Output from the blueprint selector agent."""

    blueprint_key: str = Field(
        description="The selected blueprint key. Must exactly match one of the available blueprint keys."
    )
    content_role: Literal["pillar", "supporting", "high_intent"] = Field(
        description=(
            "Role in the content hierarchy. "
            "'pillar' for broad authority pages that anchor a topic cluster (rare — only for topics spanning 5+ sub-topics). "
            "'supporting' for educational depth pages that build topical authority. "
            "'high_intent' for decision-stage or conversion-focused pages."
        )
    )
    pillar_slug: Literal["learn", "compare", "guides", "resources"] = Field(
        description=(
            "Content pillar for URL structure. "
            "'learn' for educational/awareness content. "
            "'compare' for decision-support content. "
            "'guides' for implementation/tactical content. "
            "'resources' for utility/asset content."
        )
    )
    rationale: str = Field(
        description="Brief explanation of why this blueprint was selected (1-2 sentences)."
    )


class BlueprintSelectorAgent(BaseAgent[BlueprintSelectorInput, BlueprintSelectorOutput]):
    """Select the optimal page type blueprint for a topic.

    This is a fast, lightweight agent that runs before brief generation.
    It chooses which blueprint template best matches the keyword, intent,
    and audience — ensuring diverse, well-structured content.
    """

    model_tier = "fast"
    temperature = 0.3

    @property
    def system_prompt(self) -> str:
        return """You are an SEO content strategist selecting the optimal page type for a content piece.

## Your Task
Given a topic with its keyword, intent, and funnel stage, select:
1. The best-matching page type blueprint
2. The content role in the site hierarchy
3. The content pillar for URL organization

## Blueprint Selection Rules

Match the blueprint to what the searcher actually wants:
- "best X for Y" or "top 10" patterns → best-x-for-y
- "X vs Y" or direct brand comparisons → comparison
- "X alternatives" or replacement intent → alternatives
- Niche audience + specific workflow → use-case
- Industry-vertical service pages → industry
- "X template" or reusable asset queries → template
- "X statistics" or data/benchmark queries → statistics
- "what is X" or definition queries → glossary
- Interactive utility or tool queries → tool
- "X checklist" or step-by-step process queries → checklist

Do NOT default everything to the same blueprint. Consider the keyword pattern, search intent, and what format would genuinely serve the searcher best.

## Content Role Rules

- **pillar**: RARE. Only for broad authority topics that will serve as the hub linking to 5+ sub-articles. Examples: "SEO Guide", "Content Marketing", "CRM Software".
- **supporting**: Educational depth pages that link to a pillar. Glossaries, statistics, industry overviews, use-case guides.
- **high_intent**: Decision-stage or conversion-focused pages. Comparisons, alternatives, best-of lists, templates, tools, checklists.

The blueprint's default_content_role is a good starting point, but override it if the topic clearly warrants a different role.

## Pillar Slug Rules

- **learn**: Definitions, statistics, industry insights, foundational knowledge
- **compare**: Comparisons, best-of lists, alternatives, evaluation content
- **guides**: How-to guides, use cases, checklists, implementation playbooks
- **resources**: Templates, tools, calculators, downloadable assets

Output structured JSON only."""

    @property
    def output_type(self) -> type[BlueprintSelectorOutput]:
        return BlueprintSelectorOutput

    def _build_prompt(self, input_data: BlueprintSelectorInput) -> str:
        logger.info(
            "Building blueprint selector prompt",
            extra={
                "topic": input_data.topic_name,
                "primary_keyword": input_data.primary_keyword,
                "discovery_page_type": input_data.discovery_page_type,
            },
        )

        supporting = ", ".join(input_data.supporting_keywords[:5]) if input_data.supporting_keywords else "None"

        blueprints_text = "\n".join(
            f"- **{bp['key']}** ({bp['label']}): {bp['description']} "
            f"[default role: {bp.get('default_content_role', 'supporting')}, "
            f"default pillar: {bp.get('default_pillar_slug', 'learn')}]"
            for bp in input_data.available_blueprints
        )

        return f"""Select the best page type blueprint for this topic.

## Topic
- **Name**: {input_data.topic_name}
- **Primary Keyword**: {input_data.primary_keyword}
- **Search Intent**: {input_data.search_intent}
- **Discovery Page Type**: {input_data.discovery_page_type}
- **Funnel Stage**: {input_data.funnel_stage}
- **Supporting Keywords**: {supporting}

## Available Blueprints
{blueprints_text}

Select the blueprint_key, content_role, and pillar_slug that best match this topic's keyword pattern and search intent."""
