"""Brand extractor agent for Step 1: Brand Profile extraction."""

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ProductService(BaseModel):
    """A product or service offered by the company."""

    name: str
    description: str
    category: str
    target_audience: str | None = None
    core_benefits: list[str] = Field(default_factory=list)


class ICP(BaseModel):
    """Ideal Customer Profile - target audience details."""

    target_roles: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    company_sizes: list[str] = Field(default_factory=list)
    primary_pains: list[str] = Field(default_factory=list)
    desired_outcomes: list[str] = Field(default_factory=list)
    common_objections: list[str] = Field(default_factory=list)


class BrandExtractorInput(BaseModel):
    """Input for the brand extractor agent."""

    domain: str
    scraped_content: str
    additional_context: str | None = None


class BrandProfile(BaseModel):
    """Extracted brand profile output."""

    company_name: str
    tagline: str | None = None
    products_services: list[ProductService] = Field(default_factory=list)
    money_pages: list[str] = Field(
        default_factory=list,
        description="URLs that appear to be conversion/sales focused",
    )
    unique_value_props: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    target_audience: ICP
    tone_attributes: list[str] = Field(
        default_factory=list,
        description="E.g., professional, casual, technical, friendly",
    )
    allowed_claims: list[str] = Field(
        default_factory=list,
        description="Claims that are supported by evidence on the site",
    )
    restricted_claims: list[str] = Field(
        default_factory=list,
        description="Claims that should be avoided (unsubstantiated, regulated)",
    )
    in_scope_topics: list[str] = Field(default_factory=list)
    out_of_scope_topics: list[str] = Field(default_factory=list)
    extraction_confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in the extraction (0-1)",
    )


class BrandExtractorAgent(BaseAgent[BrandExtractorInput, BrandProfile]):
    """Agent for extracting brand profile from scraped website content.

    Used in Step 1 of the pipeline to understand the brand, products,
    target audience, and positioning from website content.
    """

    model_tier = "reasoning"  # Complex extraction needs quality
    temperature = 0.3  # Lower temperature for more factual extraction

    @property
    def system_prompt(self) -> str:
        return """You are an expert brand analyst specializing in B2B and B2C companies.
Your task is to extract comprehensive brand information from website content.

Focus on identifying:
1. **Products/Services**: What does the company sell? Include names, descriptions,
   target users, and key benefits for each.

2. **Target Audience (ICP)**: Who are they selling to? Identify roles, industries,
   company sizes, pain points, desired outcomes, and common objections.

3. **Positioning**: What makes them unique? Extract value propositions,
   differentiators, and positioning statements.

4. **Voice & Tone**: How do they communicate? Is it professional, casual,
   technical, friendly, authoritative?

5. **Claims**: What claims do they make? Separate supported claims from
   potentially risky/unsubstantiated ones.

6. **Topic Boundaries**: What topics are clearly in-scope vs out-of-scope
   for this brand?

Guidelines:
- Only extract information that can be DIRECTLY inferred from the content
- If something is unclear, indicate lower confidence
- Identify money pages (signup, demo, pricing, contact) by URL patterns
- Be thorough but accurate - quality over quantity
- Mark assumptions clearly in your extraction
"""

    @property
    def output_type(self) -> type[BrandProfile]:
        return BrandProfile

    def _build_prompt(self, input_data: BrandExtractorInput) -> str:
        logger.info(
            "Building brand extraction prompt",
            extra={
                "domain": input_data.domain,
                "content_length": len(input_data.scraped_content),
                "has_additional_context": input_data.additional_context is not None,
            },
        )
        # Truncate content if too long (keep first 15k chars)
        content = input_data.scraped_content
        if len(content) > 15000:
            content = content[:15000] + "\n\n[Content truncated...]"

        prompt = f"""Analyze the following website content from {input_data.domain}
and extract a comprehensive brand profile.

## Website Content

{content}

"""
        if input_data.additional_context:
            prompt += f"""
## Additional Context

{input_data.additional_context}

"""

        prompt += """
## Instructions

Extract all relevant brand information and structure it according to the schema.
Be thorough in identifying products/services and their target audiences.
Assess your confidence in the extraction (0-1) based on content quality.
"""
        return prompt
