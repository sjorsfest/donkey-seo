"""ICP recommendation agent for Step 1 brand profiling."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ICPNiche(BaseModel):
    """A recommended ICP niche for go-to-market targeting."""

    niche_name: str = Field(description="Short label for the ICP niche")
    target_roles: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    company_sizes: list[str] = Field(default_factory=list)
    primary_pains: list[str] = Field(default_factory=list)
    desired_outcomes: list[str] = Field(default_factory=list)
    likely_objections: list[str] = Field(default_factory=list)
    why_good_fit: str = Field(
        description="Why this niche is a strong fit based on product capabilities",
    )


class ICPRecommenderInput(BaseModel):
    """Input for ICP recommendation."""

    company_name: str
    tagline: str | None = None
    products_services: list[dict[str, Any]] = Field(default_factory=list)
    unique_value_props: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    current_target_audience: dict[str, list[str]] = Field(default_factory=dict)


class ICPRecommenderOutput(BaseModel):
    """Suggested ICP niches inferred from product fit."""

    suggested_niches: list[ICPNiche] = Field(default_factory=list)
    recommendation_confidence: float = Field(ge=0, le=1, default=0.5)


class ICPRecommenderAgent(BaseAgent[ICPRecommenderInput, ICPRecommenderOutput]):
    """Agent that proposes additional ICP niches beyond website-stated audience."""

    model_tier = "reasoning"
    temperature = 0.7

    @property
    def system_prompt(self) -> str:
        return """You are a B2B/B2C go-to-market strategist focused on ideal customer profile
design. Your job is to propose additional ICP niches that are likely to buy the product,
based on core product capability and differentiation.

Rules:
- Prioritize fit to product capability and value props over generic market guesses.
- Return 3-5 niche ideas with distinct roles/industries where possible.
- Include both obvious and adjacent opportunities when justified.
- Avoid unrealistic, heavily regulated, or weak-fit niches unless strongly supported.
- Keep output practical for SEO/content targeting and campaign planning.
- Reuse existing audience context, but do not just repeat it; expand it."""

    @property
    def output_type(self) -> type[ICPRecommenderOutput]:
        return ICPRecommenderOutput

    def _build_prompt(self, input_data: ICPRecommenderInput) -> str:
        logger.info(
            "Building ICP recommendation prompt",
            extra={
                "company_name": input_data.company_name,
                "products_count": len(input_data.products_services),
                "uvp_count": len(input_data.unique_value_props),
                "differentiators_count": len(input_data.differentiators),
            },
        )
        products_text = "\n".join(
            (
                f"- {product.get('name', 'Unknown')}: "
                f"{product.get('description', 'No description')}"
            )
            for product in input_data.products_services
        )
        if not products_text:
            products_text = "- Not provided"

        current_roles = ", ".join(input_data.current_target_audience.get("target_roles", []))
        current_industries = ", ".join(
            input_data.current_target_audience.get("target_industries", [])
        )
        current_pains = ", ".join(input_data.current_target_audience.get("primary_pains", []))

        return f"""Generate additional ICP niche recommendations for {input_data.company_name}.

## Company
Name: {input_data.company_name}
Tagline: {input_data.tagline or "Not provided"}

## Core Products / Services
{products_text}

## Unique Value Propositions
{chr(10).join(f"- {item}" for item in input_data.unique_value_props) or "- Not provided"}

## Differentiators
{chr(10).join(f"- {item}" for item in input_data.differentiators) or "- Not provided"}

## Existing Audience (from website extraction)
Roles: {current_roles or "Not provided"}
Industries: {current_industries or "Not provided"}
Primary pains: {current_pains or "Not provided"}

## Instructions
1. Propose 3-5 high-fit ICP niches this product can serve well.
2. For each niche, provide roles, industries, company sizes, pains, outcomes, and objections.
3. Explain why each niche is a good fit based on capabilities and differentiators.
4. Keep recommendations realistic and actionable.
5. Avoid copy-pasting the existing audience; broaden where justified.
"""
