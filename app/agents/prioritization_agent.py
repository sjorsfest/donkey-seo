"""Prioritization agent for Step 7: Topic Prioritization."""

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent


class TopicPrioritization(BaseModel):
    """Prioritization result for a single topic."""

    topic_index: int
    expected_role: str = Field(
        description="quick_win, authority_builder, or revenue_driver"
    )
    recommended_url_type: str = Field(
        description="blog, comparison, landing, resource, or tool"
    )
    target_money_pages: list[str] = Field(
        default_factory=list,
        description="Money pages this topic should link to",
    )
    score_adjustments: dict = Field(
        default_factory=dict,
        description="Suggested adjustments to calculated scores with reasons",
    )
    validation_notes: str = Field(
        default="",
        description="Any concerns about the scoring or prioritization",
    )


class PrioritizationAgentInput(BaseModel):
    """Input for prioritization agent."""

    topics: list[dict]  # Each has name, keywords, metrics, priority_factors
    brand_context: str = Field(default="", description="Brand profile summary")
    money_pages: list[str] = Field(default_factory=list, description="Known money page URLs")
    primary_goal: str = Field(default="", description="Project's primary business goal")


class PrioritizationAgentOutput(BaseModel):
    """Output from prioritization agent."""

    prioritizations: list[TopicPrioritization]
    overall_strategy_notes: str = Field(
        default="",
        description="High-level recommendations for the content strategy",
    )


class PrioritizationAgent(BaseAgent[PrioritizationAgentInput, PrioritizationAgentOutput]):
    """Agent for validating and enhancing topic prioritization.

    Used in Step 7 after initial scoring to:
    - Validate that scoring makes sense for the brand's goals
    - Assign expected roles (quick_win, authority_builder, revenue_driver)
    - Recommend money page linking targets
    - Flag topics that need SERP validation
    """

    model = "openai:gpt-4-turbo"
    temperature = 0.3

    @property
    def system_prompt(self) -> str:
        return """You are a content prioritization strategist. Given scored topic clusters:

1. **Assign Expected Roles**:
   - quick_win: Low difficulty (<35), decent volume (>200/mo), can rank within 2-3 months
   - authority_builder: Foundational content for topical authority, may not drive traffic directly
   - revenue_driver: High commercial intent, directly ties to conversion/money pages

2. **Recommend URL Types**:
   - blog: Informational content, how-to guides, educational pieces
   - comparison: X vs Y comparisons, alternatives, reviews
   - landing: Product/service pages, commercial intent
   - resource: Tools, templates, calculators, downloadables
   - tool: Interactive tools, calculators

3. **Money Page Linking**:
   - Identify which money pages (pricing, product, demo) each topic should link to
   - Consider the funnel stage and reader intent
   - TOFU content → awareness pages, resources
   - MOFU content → comparison, pricing pages
   - BOFU content → demo, pricing, product pages

4. **Score Validation**:
   - Flag if calculated scores seem misaligned with business goals
   - Suggest adjustments if priority seems off
   - Consider: Is a high-volume TOFU keyword more valuable than a low-volume BOFU keyword?

5. **Strategy Notes**:
   - Provide high-level recommendations
   - Identify any gaps in the topic backlog
   - Flag potential quick wins that could be prioritized

Be practical and actionable. Focus on business impact, not just SEO metrics."""

    @property
    def output_type(self) -> type[PrioritizationAgentOutput]:
        return PrioritizationAgentOutput

    def _build_prompt(self, input_data: PrioritizationAgentInput) -> str:
        topics_text = []
        for i, topic in enumerate(input_data.topics):
            name = topic.get("name", f"Topic {i}")
            primary_kw = topic.get("primary_keyword", "")
            intent = topic.get("dominant_intent", "unknown")
            funnel = topic.get("funnel_stage", "unknown")
            volume = topic.get("total_volume", 0)
            difficulty = topic.get("avg_difficulty", 0)
            keyword_count = topic.get("keyword_count", 0)
            priority_score = topic.get("priority_score", 0)
            factors = topic.get("priority_factors", {})

            factors_text = ", ".join(f"{k}: {v:.2f}" for k, v in factors.items()) if factors else "N/A"

            topics_text.append(
                f"Topic {i}: {name}\n"
                f"  Primary Keyword: {primary_kw}\n"
                f"  Intent: {intent} | Funnel: {funnel}\n"
                f"  Volume: {volume} | Difficulty: {difficulty:.1f} | Keywords: {keyword_count}\n"
                f"  Priority Score: {priority_score:.2f}\n"
                f"  Factors: {factors_text}"
            )

        context_parts = []
        if input_data.brand_context:
            context_parts.append(f"Brand Context:\n{input_data.brand_context}")
        if input_data.primary_goal:
            context_parts.append(f"Primary Goal: {input_data.primary_goal}")
        if input_data.money_pages:
            context_parts.append(f"Money Pages:\n" + "\n".join(f"  - {mp}" for mp in input_data.money_pages[:10]))

        context_text = "\n\n".join(context_parts) if context_parts else "No additional context provided."

        return f"""Prioritize and assign roles to these topics:

{context_text}

Topics (sorted by calculated priority score):
{chr(10).join(topics_text)}

For each topic, provide:
1. Expected role (quick_win / authority_builder / revenue_driver)
2. Recommended URL type (blog / comparison / landing / resource / tool)
3. Target money pages to link to
4. Any score adjustments with reasons
5. Validation notes if the prioritization seems off

Also provide overall strategy notes for the content backlog."""
