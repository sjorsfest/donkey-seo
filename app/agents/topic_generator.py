"""Topic generator agent for Step 2: Seed Topic Generation."""

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent


class Pillar(BaseModel):
    """A content pillar/theme."""

    name: str
    description: str
    icp_relevance: str = Field(description="Why this matters to the target audience")
    product_tie_in: str = Field(description="How this relates to products/services")


class SeedTopic(BaseModel):
    """A seed topic under a pillar."""

    topic_phrase: str
    pillar_name: str
    intended_content_types: list[str] = Field(
        description="E.g., guide, comparison, tutorial, listicle"
    )
    coverage_intent: str = Field(description="educate, compare, or convert")
    funnel_stage: str = Field(description="tofu, mofu, or bofu")
    relevance_score: float = Field(ge=0, le=1, description="Relevance to brand 0-1")


class TopicGeneratorInput(BaseModel):
    """Input for topic generator agent."""

    company_name: str
    products_services: list[dict]
    target_audience: dict
    unique_value_props: list[str]
    in_scope_topics: list[str]
    out_of_scope_topics: list[str]


class TopicGeneratorOutput(BaseModel):
    """Output from topic generator agent."""

    pillars: list[Pillar]
    seed_topics: list[SeedTopic]
    known_gaps: list[str] = Field(
        default_factory=list,
        description="Topics that couldn't be inferred but might be valuable",
    )


class TopicGeneratorAgent(BaseAgent[TopicGeneratorInput, TopicGeneratorOutput]):
    """Agent for generating seed topics from brand profile.

    Used in Step 2 to create 3-8 content pillars and 10-50 seed topics
    organized under those pillars.
    """

    model = "openai:gpt-4-turbo"
    temperature = 0.6  # Moderate creativity for topic ideation

    @property
    def system_prompt(self) -> str:
        return """You are an SEO content strategist specializing in building topical authority.

Given a brand profile, your task is to generate a strategic content map:

## PILLARS (3-8)
Create content pillars that:
1. Align with the company's products/services
2. Address target audience pain points and goals
3. Build topical authority in the brand's domain
4. Stay within the defined topic boundaries

Each pillar should have a clear connection to business goals.

## SEED TOPICS (10-50)
For each pillar, generate seed topics that:
1. Address specific audience questions or needs
2. Have clear search intent
3. Can be developed into standalone content
4. Vary across the marketing funnel (TOFU/MOFU/BOFU)

### Funnel Stage Guidelines:
- **TOFU (Top of Funnel)**: Educational, awareness content
  - "What is X", "How does X work", "Guide to X"
  - Intent: informational

- **MOFU (Middle of Funnel)**: Consideration content
  - "Best X for Y", "X vs Y", "X alternatives", "How to choose X"
  - Intent: commercial investigation

- **BOFU (Bottom of Funnel)**: Decision content
  - "X pricing", "X review", "X demo", "Buy X"
  - Intent: transactional

### Content Types:
- guide: Comprehensive how-to or educational content
- comparison: X vs Y format
- listicle: Top 10, Best X, etc.
- tutorial: Step-by-step instructions
- review: In-depth product/service analysis
- glossary: Definition and explanation
- case_study: Real-world example/story

## KNOWN GAPS
Identify topics that:
- Couldn't be confidently inferred from the brand profile
- Might be valuable but need confirmation
- Are borderline in-scope/out-of-scope

Be strategic and focused. Quality over quantity."""

    @property
    def output_type(self) -> type[TopicGeneratorOutput]:
        return TopicGeneratorOutput

    def _build_prompt(self, input_data: TopicGeneratorInput) -> str:
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

        return f"""Generate content pillars and seed topics for {input_data.company_name}.

## Products/Services
{products_text}

## Target Audience
{audience_text}

## Value Propositions
{chr(10).join(f'- {v}' for v in input_data.unique_value_props)}

## Topic Boundaries
In-scope: {', '.join(input_data.in_scope_topics) or 'Not specified'}
Out-of-scope: {', '.join(input_data.out_of_scope_topics) or 'Not specified'}

## Instructions
1. Create 3-8 strategic content pillars
2. Generate 10-50 seed topics distributed across pillars
3. Ensure a mix of TOFU (40%), MOFU (40%), BOFU (20%) content
4. Assign content types based on topic intent
5. Score relevance to the brand (0-1)
6. Note any gaps or uncertainties
"""
