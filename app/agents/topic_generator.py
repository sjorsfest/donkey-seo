"""Topic generator agent for Step 2: Seed Keyword Generation."""

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SeedBucket(BaseModel):
    """A seed keyword bucket/category."""

    name: str = Field(description="Bucket name, e.g. 'Core Offer', 'Pain Points', 'Use Cases'")
    description: str = Field(description="What this bucket covers")
    icp_relevance: str = Field(description="Why this bucket matters to the target audience")
    product_tie_in: str = Field(description="How this bucket relates to products/services")


class SeedKeyword(BaseModel):
    """A seed keyword for keyword research."""

    keyword: str = Field(
        description="Short seed keyword (1-4 words). Must be a broad, generic term suitable for keyword research tools.",
    )
    bucket_name: str = Field(description="Which bucket this keyword belongs to")
    relevance_score: float = Field(ge=0, le=1, description="Relevance to brand 0-1")


class TopicGeneratorInput(BaseModel):
    """Input for topic generator agent."""

    company_name: str
    products_services: list[dict]
    offer_categories: list[str] = Field(default_factory=list)
    target_audience: dict
    buyer_jobs: list[str] = Field(default_factory=list)
    conversion_intents: list[str] = Field(default_factory=list)
    unique_value_props: list[str]
    in_scope_topics: list[str]
    out_of_scope_topics: list[str]
    learning_context: str = Field(
        default="",
        description="Optional project memory from previous discovery iterations.",
    )


class TopicGeneratorOutput(BaseModel):
    """Output from topic generator agent."""

    buckets: list[SeedBucket]
    seed_keywords: list[SeedKeyword]
    known_gaps: list[str] = Field(
        default_factory=list,
        description="Areas that couldn't be inferred but might yield good seeds",
    )


class TopicGeneratorAgent(BaseAgent[TopicGeneratorInput, TopicGeneratorOutput]):
    """Agent for generating seed keywords from brand profile.

    Used in Step 2 to create seed keyword buckets and 20-50 seed keywords
    that can be fed into keyword research tools for expansion.
    """

    model_tier = "reasoning"
    temperature = 0.6

    @property
    def system_prompt(self) -> str:
        return """You are an SEO keyword researcher. Your job is to generate SEED KEYWORDS — broad starter terms that can be expanded by keyword tools into many relevant opportunities.

## WHAT SEED KEYWORDS ARE

Seed keywords are SHORT (1-4 words), broad, generic terms. They are NOT article titles, NOT long-tail phrases, NOT full sentences.

GOOD seed keywords:
- live chat software
- customer support
- helpdesk pricing
- ticketing system
- chatbot
- knowledge base
- support automation
- CRM for startups

BAD seed keywords (TOO SPECIFIC / TOO LONG — never generate these):
- "Pricing breakdown for real-time integration features across free and Pro tiers"
- "How to collect visitor email and name without disrupting user flow"
- "Advanced automation scenarios without per-seat pricing"
- "Choosing the right plan for your indie SaaS support"

## SEED KEYWORD BUCKETS

Organize seeds into these buckets (use only the ones that apply):

1) **Core Offer** — Product/service category names. The generic term someone would search before they know brands.
   Examples: project management software, roof repair, meal prep, accounting services

2) **Pain Points** — Problems the audience has. Short "how to" / "fix" / "reduce" roots.
   Examples: reduce churn, fix slow website, improve sleep, stop back pain

3) **Use Cases** — Jobs-to-be-done. "[thing] for [audience]" patterns.
   Examples: CRM for small business, invoicing for freelancers, yoga for beginners

4) **Audience Modifiers** — Who it's for (combine with core terms in expansion).
   Examples: for startups, for dentists, for remote teams, for ecommerce

5) **Features** — Major features people search/shop by.
   Examples: time tracking, automations, online booking, inventory management

6) **Alternatives & Competitors** — "alternative", "vs", adjacent categories.
   Examples: Asana alternative, Mailchimp vs, password manager, email marketing

7) **Location** (only if the business is local) — City/area + service.
   Examples: plumber amsterdam, dentist centrum, wedding photographer near me

8) **Purchase Intent** — Words that signal buying/evaluating.
   Examples: pricing, cost, best, reviews, demo, free trial

## RULES

1. Each seed keyword MUST be 1-4 words. Absolutely no exceptions.
2. Seeds must be terms that real people actually search for on Google.
3. Seeds must be broad enough that a keyword tool can expand them into 10-100+ related keywords.
4. Do NOT generate article titles, questions, or full phrases.
5. Do NOT over-specify — "live chat" is better than "live chat for SaaS startups with 2-person teams".
6. Aim for 20-50 seed keywords total.
7. Stay within the defined topic boundaries.
8. Prefer simple, commonly searched terms over niche jargon."""

    @property
    def output_type(self) -> type[TopicGeneratorOutput]:
        return TopicGeneratorOutput

    def _build_prompt(self, input_data: TopicGeneratorInput) -> str:
        logger.info(
            "Building seed keyword generation prompt",
            extra={
                "company": input_data.company_name,
                "products_count": len(input_data.products_services),
                "offer_categories_count": len(input_data.offer_categories),
                "buyer_jobs_count": len(input_data.buyer_jobs),
                "in_scope_count": len(input_data.in_scope_topics),
                "out_of_scope_count": len(input_data.out_of_scope_topics),
            },
        )
        products_text = "\n".join(
            f"- {p.get('name', 'Unknown')}: {p.get('description', 'No description')}"
            for p in input_data.products_services
        )

        audience_text = f"""
Target Roles: {', '.join(input_data.target_audience.get('target_roles', []))}
Industries: {', '.join(input_data.target_audience.get('target_industries', []))}
Pain Points: {', '.join(input_data.target_audience.get('primary_pains', []))}
Desired Outcomes: {', '.join(input_data.target_audience.get('desired_outcomes', []))}
"""

        offer_categories = ", ".join(input_data.offer_categories) or "Not specified"
        buyer_jobs = ", ".join(input_data.buyer_jobs) or "Not specified"
        conversion_intents = ", ".join(input_data.conversion_intents) or "Not specified"
        learning_context = (
            f"\n## Discovery Memory\n{input_data.learning_context}\n"
            if input_data.learning_context
            else ""
        )

        return f"""Generate seed keywords for {input_data.company_name}.

## Products/Services
{products_text}

## Offer Categories
{offer_categories}

## Target Audience
{audience_text}

## Buyer Jobs
{buyer_jobs}

## Conversion Intents
{conversion_intents}

## Value Propositions
{chr(10).join(f'- {v}' for v in input_data.unique_value_props)}

## Topic Boundaries
In-scope: {', '.join(input_data.in_scope_topics) or 'Not specified'}
Out-of-scope: {', '.join(input_data.out_of_scope_topics) or 'Not specified'}
{learning_context}

## Instructions
1. Create seed keyword buckets (only use buckets that apply to this business)
2. Generate 20-50 short seed keywords (1-4 words each) distributed across buckets
3. Each seed must be a broad term suitable for keyword research tool expansion
4. Score relevance to the brand (0-1)
5. Note any gaps — areas where you couldn't confidently generate seeds

Remember: seed keywords are SHORT. "live chat software" not "How to choose the right live chat software for your business".
"""
