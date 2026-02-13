"""Intent classifier agent for Step 5: Intent Labeling."""

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class KeywordIntent(BaseModel):
    """Intent classification for a single keyword."""

    keyword: str
    intent_label: str = Field(
        description="informational, commercial, transactional, or navigational"
    )
    intent_confidence: float = Field(ge=0, le=1, description="Confidence 0-1")
    recommended_page_type: str = Field(
        description="guide, list, comparison, alternatives, tool, glossary, or landing"
    )
    funnel_stage: str = Field(description="tofu, mofu, or bofu")
    intent_rationale: str = Field(description="Brief explanation of classification")
    risk_flags: list[str] = Field(
        default_factory=list,
        description="ambiguous, local, ugc_dominated, etc."
    )


class IntentClassifierInput(BaseModel):
    """Input for intent classifier agent."""

    keywords: list[str]
    context: str = Field(
        default="",
        description="Optional context about the brand/niche"
    )


class IntentClassifierOutput(BaseModel):
    """Output from intent classifier agent."""

    classifications: list[KeywordIntent]


class IntentClassifierAgent(BaseAgent[IntentClassifierInput, IntentClassifierOutput]):
    """Agent for classifying keyword search intent.

    Used in Step 5 to:
    - Classify intent (informational/commercial/transactional/navigational)
    - Recommend page type
    - Assign funnel stage
    - Flag risks

    Note: Deterministic rules are applied first in the step service.
    This agent handles ambiguous cases that rules can't classify.
    """

    model_tier = "standard"
    temperature = 0.3  # Low temperature for consistent classification

    @property
    def system_prompt(self) -> str:
        return """You are a search intent classification expert. For each keyword, determine:

1. **Search Intent** (choose exactly one):
   - informational: User wants to learn or understand (how to, what is, guide, tutorial)
   - commercial: User is researching before purchase (best, alternatives, vs, comparison, review)
   - transactional: User is ready to buy/convert (buy, order, price, discount, deal)
   - navigational: User wants a specific page/site (brand name, login, specific product)

2. **Page Type Recommendation** (choose exactly one):
   - guide: Step-by-step how-to or comprehensive educational content
   - list: Listicle format (top 10, best X, X tools/tips)
   - comparison: X vs Y format
   - alternatives: Alternative to X format
   - tool: Interactive tool, calculator, or template
   - glossary: Definition and explanation of terms
   - landing: Product/service landing page

3. **Funnel Stage** (choose exactly one):
   - tofu: Top of funnel - awareness, education (informational intent)
   - mofu: Middle of funnel - consideration, research (commercial intent)
   - bofu: Bottom of funnel - decision, purchase (transactional intent)

4. **Risk Flags** (include all that apply, or empty list):
   - ambiguous: Intent unclear from keyword alone
   - local: Likely has local search intent ("near me", city names)
   - ugc_dominated: SERPs likely dominated by forums/Reddit/Quora
   - branded: Contains competitor/brand name
   - seasonal: Likely has seasonal search patterns
   - trending: Current events/news topic

Be precise and consistent. Return structured classifications for each keyword."""

    @property
    def output_type(self) -> type[IntentClassifierOutput]:
        return IntentClassifierOutput

    def _build_prompt(self, input_data: IntentClassifierInput) -> str:
        logger.info(
            "Building intent classification prompt",
            extra={
                "keyword_count": len(input_data.keywords),
                "has_context": bool(input_data.context),
            },
        )
        keywords_text = "\n".join(f"- {kw}" for kw in input_data.keywords)

        context_text = ""
        if input_data.context:
            context_text = f"\n\nContext about the brand/niche:\n{input_data.context}\n"

        return f"""Classify the search intent for each of these keywords:{context_text}

Keywords to classify:
{keywords_text}

For each keyword, provide:
1. Intent label (informational/commercial/transactional/navigational)
2. Confidence score (0-1)
3. Recommended page type
4. Funnel stage (tofu/mofu/bofu)
5. Brief rationale (1 sentence)
6. Any risk flags

Return all classifications in a single structured response."""
