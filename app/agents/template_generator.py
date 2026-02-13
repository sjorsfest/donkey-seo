"""Template generator agent for Step 13: Writer Instructions + QA Gates."""

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ========== Project Style Guide Generation ==========

class QAChecklistItem(BaseModel):
    """A single QA checklist item."""

    item: str = Field(description="The check to perform")
    required: bool = Field(description="Is this required to pass?")
    threshold: str = Field(default="", description="Threshold if applicable (e.g., 'min 3 internal links')")


class VoiceToneConstraints(BaseModel):
    """Voice and tone constraints for a project."""

    do_list: list[str] = Field(
        default_factory=list,
        description="Things the content SHOULD do",
    )
    dont_list: list[str] = Field(
        default_factory=list,
        description="Things the content should NOT do",
    )
    good_examples: list[str] = Field(
        default_factory=list,
        description="Examples of good tone/voice",
    )
    bad_examples: list[str] = Field(
        default_factory=list,
        description="Examples of bad tone/voice to avoid",
    )


class ProjectStyleGuideResult(BaseModel):
    """Complete project-level style guide."""

    voice_tone_constraints: VoiceToneConstraints
    forbidden_claims: list[str] = Field(
        default_factory=list,
        description="Claims that should never be made",
    )
    compliance_notes: list[str] = Field(
        default_factory=list,
        description="Industry-specific compliance requirements",
    )
    formatting_requirements: dict = Field(
        default_factory=dict,
        description="Formatting standards (headings, lists, etc.)",
    )
    base_qa_checklist: list[QAChecklistItem] = Field(
        default_factory=list,
        description="Base QA items for all content",
    )
    common_failure_modes: list[str] = Field(
        default_factory=list,
        description="Common mistakes to watch for",
    )


class StyleGuideGeneratorInput(BaseModel):
    """Input for project style guide generation."""

    company_name: str = Field(description="Company name")
    tagline: str = Field(default="", description="Company tagline")
    products_services: list[str] = Field(
        default_factory=list,
        description="Main products/services",
    )
    tone_attributes: list[str] = Field(
        default_factory=list,
        description="Desired tone attributes",
    )
    target_audience: list[str] = Field(
        default_factory=list,
        description="Target audience roles/demographics",
    )
    allowed_claims: list[str] = Field(
        default_factory=list,
        description="Claims that are allowed",
    )
    restricted_claims: list[str] = Field(
        default_factory=list,
        description="Claims that are restricted",
    )
    compliance_flags: list[str] = Field(
        default_factory=list,
        description="Compliance requirements (HIPAA, GDPR, etc.)",
    )


class StyleGuideGeneratorOutput(BaseModel):
    """Output from style guide generation."""

    style_guide: ProjectStyleGuideResult


class StyleGuideGeneratorAgent(BaseAgent[StyleGuideGeneratorInput, StyleGuideGeneratorOutput]):
    """Agent for generating project-level style guides.

    Used once per project to create:
    - Voice/tone constraints
    - Forbidden claims
    - Compliance notes
    - Base QA checklist
    """

    model_tier = "reasoning"  # Complex generation
    temperature = 0.3

    @property
    def system_prompt(self) -> str:
        return """You are creating a comprehensive style guide for a content project.

Given a brand profile, generate:

## 1. Voice & Tone Constraints
- DO list: What the content should do (be helpful, use examples, etc.)
- DON'T list: What to avoid (jargon, superlatives, etc.)
- Good examples: Sample sentences showing good tone
- Bad examples: Sample sentences showing what to avoid

## 2. Forbidden Claims
- Claims that should NEVER be made
- Unsubstantiated superlatives to avoid
- Competitor mentions to avoid (if applicable)

## 3. Compliance Notes
- Industry-specific requirements
- Legal disclaimers needed
- Regulatory considerations

## 4. Formatting Requirements
- Heading styles and conventions
- List formatting preferences
- Image/media guidelines

## 5. Base QA Checklist
Items every piece of content must pass:
- Required: Must pass to publish
- Recommended: Should pass but can be waived

## 6. Common Failure Modes
- Frequent mistakes writers make
- Easy-to-miss requirements

Be specific and actionable. This guide will be used by writers and AI tools."""

    @property
    def output_type(self) -> type[StyleGuideGeneratorOutput]:
        return StyleGuideGeneratorOutput

    def _build_prompt(self, input_data: StyleGuideGeneratorInput) -> str:
        logger.info(
            "Building style guide prompt",
            extra={
                "company": input_data.company_name,
                "tone_attributes_count": len(input_data.tone_attributes),
                "compliance_flags": input_data.compliance_flags,
            },
        )
        products = ", ".join(input_data.products_services) if input_data.products_services else "Not specified"
        tone = ", ".join(input_data.tone_attributes) if input_data.tone_attributes else "Professional"
        audience = ", ".join(input_data.target_audience) if input_data.target_audience else "General audience"
        compliance = ", ".join(input_data.compliance_flags) if input_data.compliance_flags else "Standard (no special requirements)"
        allowed = ", ".join(input_data.allowed_claims) if input_data.allowed_claims else "Not specified"
        restricted = ", ".join(input_data.restricted_claims) if input_data.restricted_claims else "Not specified"

        return f"""Create a comprehensive style guide for this brand:

## Brand Information
- **Company**: {input_data.company_name}
- **Tagline**: {input_data.tagline or "Not provided"}
- **Products/Services**: {products}

## Desired Voice & Tone
{tone}

## Target Audience
{audience}

## Compliance Requirements
{compliance}

## Allowed Claims
{allowed}

## Restricted Claims
{restricted}

---

Generate a complete style guide with:
1. Voice/tone DO and DON'T lists with examples
2. Forbidden claims (things to never say)
3. Compliance notes for writers
4. Formatting requirements
5. Base QA checklist (required and recommended items)
6. Common failure modes to watch for"""


# ========== Brief Delta Generation ==========

class BriefDeltaResult(BaseModel):
    """Delta from style guide for a specific brief."""

    page_type_rules: dict = Field(
        default_factory=dict,
        description="Rules specific to this page type",
    )
    must_include_sections: list[str] = Field(
        default_factory=list,
        description="Required sections for this content",
    )
    h1_h2_usage: dict = Field(
        default_factory=dict,
        description="Heading structure requirements",
    )
    schema_type: str = Field(
        default="Article",
        description="Schema.org type to use",
    )
    additional_qa_items: list[QAChecklistItem] = Field(
        default_factory=list,
        description="Extra QA items for this brief",
    )


class BriefDeltaGeneratorInput(BaseModel):
    """Input for brief delta generation."""

    brief_summary: dict = Field(description="Summary of the content brief")
    page_type: str = Field(description="Page type (guide, comparison, list, etc.)")
    search_intent: str = Field(description="Search intent")
    funnel_stage: str = Field(description="TOFU, MOFU, or BOFU")


class BriefDeltaGeneratorOutput(BaseModel):
    """Output from brief delta generation."""

    delta: BriefDeltaResult


class BriefDeltaGeneratorAgent(BaseAgent[BriefDeltaGeneratorInput, BriefDeltaGeneratorOutput]):
    """Agent for generating per-brief deltas.

    Generates ONLY the page-type specific additions that differ
    from the base ProjectStyleGuide.
    """

    model_tier = "fast"  # Simple templating
    temperature = 0.3

    @property
    def system_prompt(self) -> str:
        return """You are generating page-type specific instructions for a content brief.

Given a brief's details, provide ONLY the delta (additions) from the base style guide:

## Page Type Rules
Rules specific to this content type:
- Comparison pages: Must have comparison table
- How-to guides: Must have numbered steps
- Listicles: Must have consistent item format
- Tools: Must have usage instructions

## Required Sections
Sections that MUST be present based on page type:
- Guide: Introduction, Steps/Sections, Conclusion, FAQ
- Comparison: Intro, Comparison Table, Detailed Analysis, Verdict
- List: Intro, Items with descriptions, How to choose, FAQ

## Schema Type
Appropriate schema.org type:
- HowTo: Step-by-step guides
- FAQ: Content with Q&A
- Article: General content
- Product: Product pages
- ItemList: Listicles

## Additional QA Items
Extra checks specific to this content type.

Be concise - only include what's DIFFERENT from a standard article."""

    @property
    def output_type(self) -> type[BriefDeltaGeneratorOutput]:
        return BriefDeltaGeneratorOutput

    def _build_prompt(self, input_data: BriefDeltaGeneratorInput) -> str:
        logger.info(
            "Building brief delta prompt",
            extra={
                "page_type": input_data.page_type,
                "search_intent": input_data.search_intent,
                "funnel_stage": input_data.funnel_stage,
            },
        )
        brief = input_data.brief_summary

        return f"""Generate page-type specific instructions for this brief:

## Brief Details
- **Primary Keyword**: {brief.get('primary_keyword', 'Unknown')}
- **Page Type**: {input_data.page_type}
- **Search Intent**: {input_data.search_intent}
- **Funnel Stage**: {input_data.funnel_stage}
- **Topic Name**: {brief.get('topic_name', 'Unknown')}

---

Provide ONLY the delta (additions to base style guide):
1. Page-type specific rules
2. Required sections for this content type
3. Heading structure requirements
4. Appropriate schema type
5. Additional QA items specific to this page type"""
