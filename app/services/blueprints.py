"""Page type blueprint registry for structured SEO content generation.

Defines 10 page type blueprints with required sections, sub-structure specs,
quality rules, conversion elements, and common mistakes. Each blueprint is a
frozen dataclass that guides brief generation, writer templates, and QA.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SectionContract:
    """Defines what a section must accomplish and contain."""

    name: str
    purpose: str
    required_fields: tuple[str, ...] = ()
    flexible_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageTypeBlueprint:
    """Complete blueprint for a page type."""

    key: str
    label: str
    description: str
    default_content_role: str  # "pillar" / "supporting" / "high_intent"
    sections: tuple[SectionContract, ...] = ()
    quality_rules: tuple[str, ...] = ()
    conversion_elements: tuple[str, ...] = ()
    common_mistakes: tuple[str, ...] = ()
    schema_type: str = "Article"
    word_count_range: tuple[int, int] = (1500, 3000)
    default_pillar_slug: str = "learn"
    qa_module: str = "A"


# ---------------------------------------------------------------------------
# Pillar configuration (replaces the old blog/tools/guides set)
# ---------------------------------------------------------------------------

ALLOWED_PILLAR_CONFIG: dict[str, tuple[str, str]] = {
    "learn": (
        "Learn",
        "Educational and awareness content: definitions, statistics, industry insights, and foundational knowledge.",
    ),
    "compare": (
        "Compare",
        "Decision-support content: product comparisons, best-of lists, alternatives, and evaluation frameworks.",
    ),
    "guides": (
        "Guides",
        "Implementation and tactical content: how-to guides, use cases, checklists, and playbooks.",
    ),
    "resources": (
        "Resources",
        "Utility and asset content: templates, tools, calculators, and downloadable resources.",
    ),
}

# ---------------------------------------------------------------------------
# 10 Page Type Blueprints
# ---------------------------------------------------------------------------

_BEST_X_FOR_Y = PageTypeBlueprint(
    key="best-x-for-y",
    label="Best X for Y",
    description=(
        "A commercial investigation page for people comparing options before "
        "choosing a product, tool, service, or solution. Create when the keyword "
        "has buyer intent and users are clearly trying to shortlist vendors."
    ),
    default_content_role="high_intent",
    sections=(
        SectionContract(
            "Introduction",
            "Define who the page is for, explain how the list was evaluated, state selection criteria up front.",
            required_fields=("who_page_is_for", "selection_criteria"),
            flexible_fields=("market_context",),
        ),
        SectionContract(
            "Top Picks Summary",
            "Quick shortlist near the top with best-for segments.",
            required_fields=("best_overall", "best_for_beginners", "best_for_budget"),
            flexible_fields=("best_for_enterprise", "best_for_automation"),
        ),
        SectionContract(
            "Comparison Table",
            "Scannable feature comparison to help users compare shortlist options quickly.",
            required_fields=("price_range", "key_features", "ideal_user", "free_trial"),
            flexible_fields=("integrations", "support_level", "onboarding_difficulty"),
        ),
        SectionContract(
            "Detailed Reviews",
            "Structured, comparable review per option.",
            required_fields=(
                "overview", "best_fit", "key_features", "pricing_snapshot",
                "strengths", "limitations", "standout_differentiator", "verdict",
            ),
            flexible_fields=("use_cases", "integrations"),
        ),
        SectionContract(
            "Buyer Guide",
            "Help users evaluate based on their situation.",
            required_fields=("budget_considerations", "team_size", "technical_complexity"),
            flexible_fields=("scalability", "support_requirements", "implementation_time"),
        ),
        SectionContract(
            "FAQ",
            "High-intent follow-up questions.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Final Recommendation",
            "Scenario-based recommendations instead of naming only one winner.",
            required_fields=("best_fit_scenarios",),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Every reviewed option must include pros, cons, and pricing.",
        "Segment recommendations by user type, not just rank.",
        "Include a comparison table in the top 30% of the article.",
        "Explain tradeoffs, not just features.",
        "Avoid vague praise — be specific about what each option does well.",
        "Include decision support, not just summaries.",
    ),
    conversion_elements=("Product CTA", "Lead magnet", "Comparison-driven internal links"),
    common_mistakes=(
        "Listing tools without evaluation criteria.",
        "Making every option sound equally good.",
        "No segmentation by user type.",
        "Overly generic pros/cons.",
        "No conclusion or recommendation framework.",
    ),
    schema_type="ItemList",
    word_count_range=(2000, 4000),
    default_pillar_slug="compare",
    qa_module="B",
)

_COMPARISON = PageTypeBlueprint(
    key="comparison",
    label="Comparison (X vs Y)",
    description=(
        "A decision-stage page comparing two alternatives directly. Create when "
        "people search exact brand-vs-brand queries or the two options solve the "
        "same problem in different ways."
    ),
    default_content_role="high_intent",
    sections=(
        SectionContract(
            "Introduction",
            "Identify who should read this and explain both tools serve similar needs but differ in important ways.",
            required_fields=("target_reader", "shared_purpose"),
            flexible_fields=("market_context",),
        ),
        SectionContract(
            "At a Glance Verdict",
            "Quick verdict split by user type.",
            required_fields=("best_for_beginners", "best_for_advanced", "best_for_budget"),
            flexible_fields=("best_for_customization", "best_for_speed"),
        ),
        SectionContract(
            "Feature-by-Feature Comparison",
            "Compare the same dimensions in the same order.",
            required_fields=(
                "core_functionality", "ease_of_use", "pricing_model",
                "integrations", "support", "scalability",
            ),
            flexible_fields=("customization", "setup_time", "reporting"),
        ),
        SectionContract(
            "Use Case Recommendations",
            "Explain which option is better for specific user segments.",
            required_fields=("solo_users", "smbs", "enterprise_teams"),
            flexible_fields=("agencies", "technical_vs_nontechnical"),
        ),
        SectionContract(
            "Migration or Switching Considerations",
            "Lock-in risk, data portability, learning curve, implementation burden.",
            required_fields=("lock_in_risk", "data_portability"),
            flexible_fields=("learning_curve", "implementation_burden"),
        ),
        SectionContract(
            "Bottom Line",
            "Scenario-based recommendations: choose X if… / choose Y if…",
            required_fields=("choose_x_if", "choose_y_if"),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Stay balanced — compare the same dimensions in the same order.",
        "Be explicit about tradeoffs.",
        "End with 'choose X if… / choose Y if…' scenario-based recommendations.",
        "Compare by audience fit, not just features.",
    ),
    conversion_elements=("Product CTA", "Internal link to alternatives page", "Demo/trial CTA"),
    common_mistakes=(
        "Biased writing with no evidence.",
        "Comparing different categories unfairly.",
        "No audience segmentation.",
        "No practical recommendation.",
    ),
    schema_type="Article",
    word_count_range=(2000, 4000),
    default_pillar_slug="compare",
    qa_module="B",
)

_ALTERNATIVES = PageTypeBlueprint(
    key="alternatives",
    label="Alternatives",
    description=(
        "A page for users dissatisfied with a known market leader or looking for "
        "replacements. Create when a major brand dominates and users look for "
        "replacements due to price, complexity, or missing features."
    ),
    default_content_role="high_intent",
    sections=(
        SectionContract(
            "Introduction",
            "Build around dissatisfaction triggers and why users seek alternatives.",
            required_fields=("dissatisfaction_triggers", "why_users_switch"),
            flexible_fields=("market_context",),
        ),
        SectionContract(
            "Best Alternatives",
            "Curated list of alternatives compared against the original.",
            required_fields=(
                "who_its_for", "how_it_differs", "main_strengths",
                "tradeoffs", "pricing_position",
            ),
            flexible_fields=("migration_difficulty",),
        ),
        SectionContract(
            "Alternative Categories",
            "Segment alternatives by user need.",
            required_fields=("cheapest", "easiest", "feature_rich"),
            flexible_fields=("enterprise", "open_source", "simplest"),
        ),
        SectionContract(
            "Feature Comparison Table",
            "Quick-scan table comparing all alternatives to the original.",
            required_fields=("price_range", "key_differentiator", "ideal_user"),
            flexible_fields=("integrations", "support"),
        ),
        SectionContract(
            "Switching Guide",
            "What to check before migrating.",
            required_fields=("data_export_import", "team_retraining"),
            flexible_fields=("implementation_risks",),
        ),
        SectionContract(
            "Final Recommendation",
            "Map alternatives to user goals.",
            required_fields=("recommendation_matrix",),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Frame alternatives around reasons for switching, not in isolation.",
        "Compare each option against the original product.",
        "Avoid producing a random list with no logic.",
        "Differentiate clearly from 'best X' pages.",
    ),
    conversion_elements=("Product CTA", "Migration guide CTA", "Demo/trial links"),
    common_mistakes=(
        "Not mentioning why someone wants an alternative.",
        "No migration or switching guidance.",
        "Generic 'best alternatives' wording.",
        "Weak differentiation from best-X pages.",
    ),
    schema_type="ItemList",
    word_count_range=(2000, 3500),
    default_pillar_slug="compare",
    qa_module="B",
)

_USE_CASE = PageTypeBlueprint(
    key="use-case",
    label="Use Case",
    description=(
        "A page tailored to a specific situation, job-to-be-done, or workflow. "
        "Create when a niche audience has unique needs and broad category pages "
        "are too generic."
    ),
    default_content_role="supporting",
    sections=(
        SectionContract(
            "Introduction",
            "Describe the context: who the user is, what outcome they need, why generic solutions fail.",
            required_fields=("target_user", "desired_outcome"),
            flexible_fields=("why_generic_fails",),
        ),
        SectionContract(
            "Pain Points",
            "List the specific problems this audience faces.",
            required_fields=(),
            flexible_fields=("audience_specifics", "failed_alternatives"),
        ),
        SectionContract(
            "Required Capabilities",
            "Explain what matters most for this use case.",
            required_fields=(),
            flexible_fields=("automation", "compliance", "collaboration", "integrations"),
        ),
        SectionContract(
            "Recommended Approach",
            "Solution, workflow, or strategy for this use case.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Feature-to-Problem Mapping",
            "Tie each key feature or service element to a real pain point.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Examples and Scenarios",
            "Show how the solution works in practice.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Decision Framework",
            "Explain what this audience should prioritize when evaluating options.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "CTA",
            "Conversion path tailored to the use case.",
            required_fields=(),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Use the language of the audience.",
        "Focus on practical requirements, not generic features.",
        "Map needs to outcomes clearly.",
        "Avoid copying generic category-page language.",
    ),
    conversion_elements=("Tailored CTA", "Internal links to comparison pages", "Demo CTA"),
    common_mistakes=(
        "Swapping the audience label but leaving the page generic.",
        "No niche-specific pain points.",
        "No tailored examples.",
        "Weak differentiation from industry pages.",
    ),
    schema_type="Article",
    word_count_range=(1500, 3000),
    default_pillar_slug="guides",
    qa_module="A",
)

_INDUSTRY = PageTypeBlueprint(
    key="industry",
    label="Industry",
    description=(
        "A verticalized page for one industry, usually with service intent. "
        "Create when your service is useful across multiple verticals and each "
        "industry has distinct trust, regulatory, or customer-acquisition needs."
    ),
    default_content_role="supporting",
    sections=(
        SectionContract(
            "Introduction",
            "Describe the market context and typical goals for this industry.",
            required_fields=("industry_context", "typical_goals"),
            flexible_fields=(),
        ),
        SectionContract(
            "Industry Pain Points",
            "Specific problems this vertical faces.",
            required_fields=(),
            flexible_fields=("local_competition", "reputation", "compliance", "lead_quality"),
        ),
        SectionContract(
            "Common Mistakes",
            "Show expertise by highlighting what most players get wrong.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Solution Framework",
            "Explain your approach specifically for that vertical.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Proof Elements",
            "Case studies, testimonials, outcome examples, common KPIs.",
            required_fields=(),
            flexible_fields=("case_studies", "testimonials", "kpis"),
        ),
        SectionContract(
            "FAQ",
            "FAQs specific to the vertical — captures long-tail intent and builds trust.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "CTA",
            "Use language that fits the industry buyer.",
            required_fields=(),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Reflect industry nuance — not just generic advice with an industry label.",
        "Reference the right KPIs and workflows for the vertical.",
        "Include vertical-specific objections and questions.",
    ),
    conversion_elements=("Industry-specific CTA", "Case study links", "Consultation CTA"),
    common_mistakes=(
        "Writing a generic service page and replacing only the industry name.",
        "Using irrelevant examples.",
        "No proof or trust signals.",
    ),
    schema_type="Article",
    word_count_range=(1500, 3000),
    default_pillar_slug="learn",
    qa_module="A",
)

_TEMPLATE = PageTypeBlueprint(
    key="template",
    label="Template",
    description=(
        "A practical page that offers a reusable asset users can adopt immediately. "
        "Create when the query implies users want a ready-made format and the "
        "template can drive lead capture or product adoption."
    ),
    default_content_role="high_intent",
    sections=(
        SectionContract(
            "Overview",
            "What the template is, who it is for, and what it helps accomplish.",
            required_fields=("what_it_is", "who_its_for", "what_it_accomplishes"),
            flexible_fields=(),
        ),
        SectionContract(
            "When to Use",
            "Explain the situations where this template is useful.",
            required_fields=("use_situations",),
            flexible_fields=(),
        ),
        SectionContract(
            "What's Included",
            "Break down the sections or fields inside the template.",
            required_fields=("sections_breakdown",),
            flexible_fields=(),
        ),
        SectionContract(
            "Preview",
            "Show enough of the template that the page is valuable without download.",
            required_fields=("template_preview",),
            flexible_fields=(),
        ),
        SectionContract(
            "Instructions",
            "Explain how to use, customize, and maintain the template.",
            required_fields=("usage_steps",),
            flexible_fields=("customization_tips", "maintenance"),
        ),
        SectionContract(
            "Best Practices",
            "Show users how to get better results from the template.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Examples",
            "Filled-out sample versions to help users understand usage.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Download CTA",
            "Conversion element: downloadable version, gated asset, trial CTA.",
            required_fields=("conversion_path",),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Generate genuinely usable structure, not a placeholder.",
        "Explain the purpose of each field in the template.",
        "Provide both the template and usage guidance.",
        "Avoid thin pages that just say 'download here'.",
    ),
    conversion_elements=("Downloadable template", "Gated asset", "Trial CTA", "Related tool links"),
    common_mistakes=(
        "No actual template shown on page.",
        "No context for how to use it.",
        "Template too generic to be helpful.",
        "Weak alignment between template and CTA.",
    ),
    schema_type="HowTo",
    word_count_range=(1000, 2500),
    default_pillar_slug="resources",
    qa_module="C",
)

_STATISTICS = PageTypeBlueprint(
    key="statistics",
    label="Statistics",
    description=(
        "A curated page of data points, trends, and benchmark numbers around a "
        "topic. Create when the topic is widely cited and journalists, bloggers, "
        "or sales teams look for data."
    ),
    default_content_role="supporting",
    sections=(
        SectionContract(
            "Introduction",
            "Explain what the statistics cover and who they're useful for.",
            required_fields=("coverage_scope", "target_audience"),
            flexible_fields=(),
        ),
        SectionContract(
            "Key Findings",
            "Place the most quotable numbers near the top.",
            required_fields=("top_stats",),
            flexible_fields=(),
        ),
        SectionContract(
            "Statistics by Theme",
            "Organize by themes: market size, adoption, performance, ROI, demographics, etc.",
            required_fields=("themed_stat_groups",),
            flexible_fields=("market_size", "adoption", "performance", "roi", "demographics"),
        ),
        SectionContract(
            "Trends",
            "Future direction and emerging patterns.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Commentary",
            "Briefly explain what the numbers mean and why they matter.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Sources",
            "Each stat must be attributable. Source transparency matters.",
            required_fields=("source_attributions",),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Structure stats by theme, not as a random list.",
        "Avoid unsupported claims — every stat needs a source.",
        "Keep commentary concise and useful.",
        "Distinguish between old benchmark data and recent data.",
        "Include freshness cues: when updated, which stats are current.",
        "Make stats citable — easy for others to quote or reference.",
    ),
    conversion_elements=("Authority backlinks", "Data download", "Related report CTA"),
    common_mistakes=(
        "No sources cited.",
        "Outdated stats presented as current.",
        "No interpretation or commentary.",
        "Dumping numbers with no structure.",
    ),
    schema_type="Article",
    word_count_range=(1500, 3000),
    default_pillar_slug="learn",
    qa_module="A",
)

_GLOSSARY = PageTypeBlueprint(
    key="glossary",
    label="Glossary / Definition",
    description=(
        "A page that defines a concept clearly and links to deeper supporting "
        "content. Create when the term has search demand and the topic supports "
        "a larger content cluster."
    ),
    default_content_role="supporting",
    sections=(
        SectionContract(
            "Definition",
            "One or two sentences written simply, near the top.",
            required_fields=("clear_definition",),
            flexible_fields=(),
        ),
        SectionContract(
            "Expanded Explanation",
            "Add context, how it works, and why it matters.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Examples",
            "Concrete examples that make the page more useful.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Common Misconceptions",
            "Clarify what the term does NOT mean.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Related Concepts",
            "Link to adjacent terms and deeper content.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Practical Implications",
            "Explain what someone should do with this knowledge.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "FAQ",
            "Support long-tail searches and related questions.",
            required_fields=(),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Define the concept simply — avoid circular definitions.",
        "Include examples and next steps.",
        "Connect the page into a broader topic cluster via internal links.",
        "Include business relevance or practical takeaway.",
    ),
    conversion_elements=("Internal cluster links", "Related guide CTA", "Glossary index link"),
    common_mistakes=(
        "Writing definitions that are too abstract.",
        "Thin content with no examples.",
        "No internal links to deeper pages.",
        "No business relevance or practical takeaway.",
    ),
    schema_type="DefinedTerm",
    word_count_range=(500, 1500),
    default_pillar_slug="learn",
    qa_module="A",
)

_TOOL = PageTypeBlueprint(
    key="tool",
    label="Tool",
    description=(
        "A page built around an interactive utility, generator, checker, "
        "calculator, or workflow helper. Create when users want to do something "
        "immediately and the utility connects to your product or service."
    ),
    default_content_role="high_intent",
    sections=(
        SectionContract(
            "Promise",
            "State exactly what the tool does. Very clear, very specific.",
            required_fields=("tool_promise",),
            flexible_fields=(),
        ),
        SectionContract(
            "Tool Interface",
            "The tool should be accessible immediately, not buried.",
            required_fields=("tool_placement",),
            flexible_fields=(),
        ),
        SectionContract(
            "Instructions",
            "Explain how to use it in a few steps.",
            required_fields=("usage_steps",),
            flexible_fields=(),
        ),
        SectionContract(
            "Output Guidance",
            "Tell the user how to interpret results.",
            required_fields=("result_interpretation",),
            flexible_fields=(),
        ),
        SectionContract(
            "Examples",
            "Show sample inputs and outputs.",
            required_fields=("sample_input_output",),
            flexible_fields=(),
        ),
        SectionContract(
            "Limitations",
            "Be transparent about what the tool cannot do.",
            required_fields=("known_limitations",),
            flexible_fields=(),
        ),
        SectionContract(
            "Supporting Content",
            "Educational copy below the tool: why this matters, best practices, common mistakes.",
            required_fields=(),
            flexible_fields=("why_it_matters", "best_practices", "common_mistakes_content"),
        ),
        SectionContract(
            "Upgrade Path",
            "Connect the free tool to paid product, audit, service, or signup flow.",
            required_fields=("upgrade_cta",),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Write short, action-oriented copy.",
        "Support the utility with explanation, not just filler text.",
        "Generate output help text so users understand results.",
        "Design the page so value is immediate — tool first, content second.",
    ),
    conversion_elements=("Product trial CTA", "Upgrade path", "Related tool links"),
    common_mistakes=(
        "Lots of SEO text but weak utility.",
        "Hidden tool UI buried under content.",
        "No explanation of results.",
        "No conversion bridge after use.",
    ),
    schema_type="SoftwareApplication",
    word_count_range=(800, 2000),
    default_pillar_slug="resources",
    qa_module="C",
)

_CHECKLIST = PageTypeBlueprint(
    key="checklist",
    label="Checklist",
    description=(
        "A procedural page that helps users complete a task step by step. "
        "Create when the topic is process-driven and users want clear execution "
        "steps for a recurring or validatable task."
    ),
    default_content_role="high_intent",
    sections=(
        SectionContract(
            "Introduction",
            "Explain what the checklist helps accomplish and who it is for.",
            required_fields=("goal", "target_user"),
            flexible_fields=(),
        ),
        SectionContract(
            "Stage-Based Checklist",
            "Group tasks by phase (before, during, after, ongoing).",
            required_fields=(
                "action", "why_it_matters", "how_to_do_it", "done_criteria",
            ),
            flexible_fields=(),
        ),
        SectionContract(
            "Priority Indicators",
            "Label items by priority: critical, recommended, optional, beginner, advanced.",
            required_fields=("priority_labels",),
            flexible_fields=(),
        ),
        SectionContract(
            "Common Failure Points",
            "Warn users where things usually go wrong.",
            required_fields=(),
            flexible_fields=(),
        ),
        SectionContract(
            "Downloadable Version",
            "Template or printable version improves usability and conversions.",
            required_fields=("download_format",),
            flexible_fields=(),
        ),
        SectionContract(
            "Next Steps",
            "Audit, consultation, product, or related guide CTA.",
            required_fields=("next_step_cta",),
            flexible_fields=(),
        ),
    ),
    quality_rules=(
        "Turn vague advice into executable steps.",
        "Define completion criteria for each checklist item.",
        "Organize by sequence or priority, not randomly.",
        "Avoid producing a superficial list of tips.",
        "Each item must include: action, why it matters, how to do it, and what 'done' looks like.",
    ),
    conversion_elements=("Downloadable checklist", "Audit CTA", "Consultation CTA", "Related guide links"),
    common_mistakes=(
        "Checklist items with no explanation.",
        "No prioritization.",
        "No grouping by stage.",
        "No completion criteria.",
    ),
    schema_type="HowTo",
    word_count_range=(1500, 3000),
    default_pillar_slug="guides",
    qa_module="A",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BLUEPRINT_REGISTRY: dict[str, PageTypeBlueprint] = {
    bp.key: bp
    for bp in (
        _BEST_X_FOR_Y,
        _COMPARISON,
        _ALTERNATIVES,
        _USE_CASE,
        _INDUSTRY,
        _TEMPLATE,
        _STATISTICS,
        _GLOSSARY,
        _TOOL,
        _CHECKLIST,
    )
}

# ---------------------------------------------------------------------------
# Deterministic fallback mapping (safety net when LLM selector fails)
# ---------------------------------------------------------------------------

_DISCOVERY_TO_BLUEPRINT: dict[str, str] = {
    "comparison": "comparison",
    "alternatives": "alternatives",
    "list": "best-x-for-y",
    "tool": "tool",
    "calculator": "tool",
    "glossary": "glossary",
    "template": "template",
    "guide": "use-case",
    "how-to": "checklist",
    "landing": "use-case",
    "blog": "use-case",
    "opinion": "use-case",
    "thought-leadership": "use-case",
    "editorial": "use-case",
}

_KEYWORD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbest\s+.+\s+for\b", re.IGNORECASE), "best-x-for-y"),
    (re.compile(r"\btop\s+\d+\b", re.IGNORECASE), "best-x-for-y"),
    (re.compile(r"\bvs\b|\bversus\b", re.IGNORECASE), "comparison"),
    (re.compile(r"\balternatives?\b", re.IGNORECASE), "alternatives"),
    (re.compile(r"\btemplate\b|\bexample\b|\bsample\b", re.IGNORECASE), "template"),
    (re.compile(r"\bstatistics?\b|\bstats\b|\bbenchmarks?\b", re.IGNORECASE), "statistics"),
    (re.compile(r"\bchecklist\b", re.IGNORECASE), "checklist"),
    (re.compile(r"\bwhat\s+is\b|\bdefinition\b|\bmeaning\b|\bglossary\b", re.IGNORECASE), "glossary"),
)


def map_discovery_to_blueprint(
    discovery_page_type: str,
    keyword_text: str,
    search_intent: str,
) -> str:
    """Deterministic fallback: map discovery page_type + keyword to a blueprint key.

    Used as a safety net when the BlueprintSelectorAgent fails.
    """
    # Pattern-match keyword first
    for pattern, blueprint_key in _KEYWORD_PATTERNS:
        if pattern.search(keyword_text):
            return blueprint_key

    # Fall through to page-type mapping
    normalized = (discovery_page_type or "").strip().lower()
    return _DISCOVERY_TO_BLUEPRINT.get(normalized, "use-case")


def get_blueprint_summary_for_selector() -> list[dict[str, str]]:
    """Return compact blueprint summaries for the selector agent prompt."""
    return [
        {
            "key": bp.key,
            "label": bp.label,
            "description": bp.description,
            "default_content_role": bp.default_content_role,
            "default_pillar_slug": bp.default_pillar_slug,
        }
        for bp in BLUEPRINT_REGISTRY.values()
    ]


def serialize_blueprint_sections(blueprint: PageTypeBlueprint) -> list[dict]:
    """Serialize blueprint sections for inclusion in LLM prompts."""
    sections = []
    for section in blueprint.sections:
        entry: dict = {
            "name": section.name,
            "purpose": section.purpose,
        }
        if section.required_fields:
            entry["required_fields"] = list(section.required_fields)
        if section.flexible_fields:
            entry["flexible_fields"] = list(section.flexible_fields)
        sections.append(entry)
    return sections
