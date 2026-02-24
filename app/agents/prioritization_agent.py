"""Prioritization agent for Step 7: Topic Prioritization."""

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TopicPrioritization(BaseModel):
    """Prioritization result for a single topic."""

    topic_id: str | None = Field(
        default=None,
        description="Stable topic identifier for robust mapping",
    )
    topic_index: int

    # Qualitative scores from LLM (0-1 scale)
    llm_business_alignment: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic business relevance: how well this topic relates to the brand's products, services, and goals (0.0-1.0)",
    )
    llm_business_alignment_rationale: str = Field(
        description="Short explanation of why this business alignment score was given",
    )
    llm_authority_value: float = Field(
        ge=0.0,
        le=1.0,
        description="Authority building value: how well this topic contributes to topical authority in context of the full cluster (0.0-1.0)",
    )
    llm_authority_value_rationale: str = Field(
        description="Short explanation of why this authority value score was given",
    )

    llm_tier_recommendation: str = Field(
        description="Final-cut recommendation: primary, secondary, or exclude",
    )
    llm_fit_adjustment: float = Field(
        ge=-0.15,
        le=0.18,
        description="Bounded fit adjustment for borderline topics (-0.15 to 0.18)",
    )
    llm_final_cut_rationale: str = Field(
        description="Short rationale for final-cut tier recommendation",
    )

    # Role and content type
    expected_role: str = Field(
        description="quick_win, authority_builder, or revenue_driver"
    )
    recommended_url_type: str = Field(
        description="blog, comparison, landing, resource, or tool"
    )
    recommended_publish_order: int = Field(
        description="Suggested publishing sequence (1=first) for optimal authority building",
    )

    # Linking
    target_money_pages: list[str] = Field(
        default_factory=list,
        description="Money pages this topic should link to",
    )

    # Notes
    validation_notes: str = Field(
        default="",
        description="Any concerns about the scoring or prioritization",
    )


class PrioritizationAgentInput(BaseModel):
    """Input for prioritization agent."""

    topics: list[dict]  # Each has name, keywords, metrics, scoring_signals
    brand_context: str = Field(default="", description="Brand profile summary")
    money_pages: list[str] = Field(default_factory=list, description="Known money page URLs")
    primary_goal: str = Field(default="", description="Project's primary business goal")
    compact_mode: bool = Field(
        default=False,
        description="Use a compact prompt when retrying after failures",
    )


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
    - Evaluate qualitative factors (business alignment, authority value)
    - Assign expected roles (quick_win, authority_builder, revenue_driver)
    - Recommend money page linking targets
    - Suggest optimal publishing order for authority building
    """

    model_tier = "standard"
    temperature = 0.3

    @property
    def system_prompt(self) -> str:
        return """You are a content prioritization strategist. Given scored topic clusters, evaluate each topic and provide final-cut recommendations.

1. **Score Business Alignment (llm_business_alignment)**: 0.0-1.0
   - Assess how semantically relevant this topic is to the brand's products, services, and goals
   - Go beyond keyword matching — understand conceptual relevance
   - Consider: Does this topic naturally lead to the brand's solutions? Would searchers be good customers?
   - Consider the brand's positioning, value props, and target audience
   - Provide a short rationale explaining your score

2. **Score Authority Building Value (llm_authority_value)**: 0.0-1.0
   - Assess how well this topic contributes to topical authority in context of the full cluster
   - Consider: Is this foundational content that establishes expertise? Does it connect sub-topics (hub value)? Does it demonstrate depth? Is it unique vs overlapping?
   - Provide a short rationale explaining your score

3. **Assign Expected Roles**:
   - quick_win: Low difficulty (<35), decent volume (>200/mo), can rank within 2-3 months
   - authority_builder: Foundational content for topical authority, may not drive traffic directly
   - revenue_driver: High commercial intent, directly ties to conversion/money pages

4. **Recommend URL Types**:
   - blog: Informational content, how-to guides, educational pieces
   - comparison: X vs Y comparisons, alternatives, reviews
   - landing: Product/service pages, commercial intent
   - resource: Tools, templates, calculators, downloadables
   - tool: Interactive tools, calculators

5. **Suggest Publishing Order (recommended_publish_order)**:
   - Assign a sequence (1, 2, 3, ...) considering authority building:
     * Foundational/broad topics first (establish expertise)
     * Specific/detailed topics later (reference foundational content)
     * Quick wins can be sprinkled throughout
   - This is about optimal sequencing, not priority score

6. **Money Page Linking**:
   - Match topics to money pages based on funnel stage and intent:
     * TOFU → awareness pages, resources
     * MOFU → comparison, pricing pages
     * BOFU → demo, pricing, product pages

7. **Final-Cut Tier (llm_tier_recommendation)**:
   - primary: strongest brand-logical opportunities for this backlog
   - secondary: good but slightly weaker than primary
   - exclude: not logical enough for this brand right now

8. **Bounded Fit Adjustment (llm_fit_adjustment)**:
   - Provide a small adjustment in [-0.15, +0.18]
   - Positive for genuinely strong borderline topics
   - Negative for weak or noisy topics
   - Keep adjustments conservative and evidence-based

9. **Final-Cut Rationale**:
   - Explain final-cut decision in one concise sentence

10. **Validation Notes**: Flag any concerns about the prioritization

Be practical and actionable. Focus on business impact, not just SEO metrics."""

    @property
    def output_type(self) -> type[PrioritizationAgentOutput]:
        return PrioritizationAgentOutput

    def _build_prompt(self, input_data: PrioritizationAgentInput) -> str:
        logger.info(
            "Building prioritization prompt",
            extra={
                "topic_count": len(input_data.topics),
                "money_pages_count": len(input_data.money_pages),
                "has_brand_context": bool(input_data.brand_context),
                "primary_goal": input_data.primary_goal or "not set",
            },
        )
        topics_text = []
        for i, topic in enumerate(input_data.topics):
            name = topic.get("name", f"Topic {i}")
            topic_id = topic.get("topic_id", "")
            primary_kw = topic.get("primary_keyword", "")
            intent = topic.get("dominant_intent", "unknown")
            funnel = topic.get("funnel_stage", "unknown")
            volume = topic.get("total_volume", 0)
            difficulty = topic.get("avg_difficulty", 0)
            keyword_count = topic.get("keyword_count", 0)
            priority_score = topic.get("priority_score", 0)
            factors = topic.get("scoring_factors", topic.get("priority_factors", {}))
            factors_text = self._format_factors_for_prompt(factors)

            topics_text.append(
                f"Topic {i}: {name}\n"
                f"  Topic ID: {topic_id}\n"
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

        if input_data.compact_mode:
            return f"""Prioritize and assign roles to these topics:

{context_text}

Topics:
{chr(10).join(topics_text)}

Keep responses concise. For EACH topic, provide:
1. llm_business_alignment + rationale
2. llm_authority_value + rationale
3. llm_tier_recommendation (primary/secondary/exclude)
4. llm_fit_adjustment (-0.15..0.18)
5. llm_final_cut_rationale
6. expected_role
7. recommended_url_type
8. recommended_publish_order
9. target_money_pages
10. validation_notes

Also provide overall_strategy_notes for the backlog."""

        return f"""Prioritize and assign roles to these topics:

{context_text}

Topics (sorted by calculated priority score):
{chr(10).join(topics_text)}

For EACH topic, provide:
1. llm_business_alignment (0.0-1.0) + rationale
2. llm_authority_value (0.0-1.0) + rationale
3. llm_tier_recommendation (primary / secondary / exclude)
4. llm_fit_adjustment (-0.15 to 0.18)
5. llm_final_cut_rationale (one sentence)
6. Expected role (quick_win / authority_builder / revenue_driver)
7. Recommended URL type (blog / comparison / landing / resource / tool)
8. Recommended publish order (1, 2, 3, ... for authority building sequence)
9. Target money pages to link to
10. Validation notes if the prioritization seems off

Also provide overall_strategy_notes for the content backlog."""

    def _format_factors_for_prompt(self, factors: object) -> str:
        """Render factor dict safely for prompt display."""
        if not isinstance(factors, dict) or not factors:
            return "N/A"

        parts: list[str] = []
        for key, value in factors.items():
            parts.append(f"{key}: {self._format_factor_value(value)}")
        return ", ".join(parts)

    def _format_factor_value(self, value: object) -> str:
        """Format mixed value types without raising."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return f"{float(value):.2f}"
        if isinstance(value, str):
            return value
        if value is None:
            return "null"
        return str(value)
