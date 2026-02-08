"""Brief generator agent for Step 12: Content Brief Generation."""

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent


class OutlineSection(BaseModel):
    """A section in the content outline."""

    heading: str = Field(description="Section heading text")
    level: int = Field(ge=1, le=4, description="Heading level (1=H1, 2=H2, etc.)")
    purpose: str = Field(description="Why this section is needed")
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points to cover in this section",
    )
    supporting_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to naturally include in this section",
    )


class ContentBriefResult(BaseModel):
    """Complete content brief for a topic."""

    working_titles: list[str] = Field(
        min_length=3,
        max_length=5,
        description="3-5 compelling, keyword-rich title options",
    )
    target_audience: str = Field(description="Who is reading this content")
    reader_job_to_be_done: str = Field(
        description="What job the reader is trying to accomplish"
    )
    outline: list[OutlineSection] = Field(description="Detailed H2/H3 structure")
    examples_required: list[str] = Field(
        default_factory=list,
        description="Specific examples/case studies to include",
    )
    faq_questions: list[str] = Field(
        default_factory=list,
        description="FAQ questions to answer (from PAA or inferred)",
    )
    meta_title_template: str = Field(description="Template for meta title")
    meta_description_template: str = Field(description="Template for meta description")
    target_word_count_min: int = Field(ge=300, description="Minimum word count")
    target_word_count_max: int = Field(le=10000, description="Maximum word count")
    must_include_sections: list[str] = Field(
        default_factory=list,
        description="Required sections that must be present",
    )
    recommended_schema_type: str = Field(
        default="Article",
        description="Schema.org type (Article, HowTo, FAQ, etc.)",
    )


class BriefGeneratorInput(BaseModel):
    """Input for brief generator agent."""

    topic_name: str = Field(description="Name of the topic cluster")
    primary_keyword: str = Field(description="Primary target keyword")
    supporting_keywords: list[str] = Field(
        default_factory=list,
        description="Secondary keywords in the cluster",
    )
    search_intent: str = Field(description="Dominant search intent")
    page_type: str = Field(description="Recommended page type")
    funnel_stage: str = Field(description="TOFU, MOFU, or BOFU")
    brand_context: str = Field(default="", description="Brand and product context")
    competitors_content_types: list[str] = Field(
        default_factory=list,
        description="Content types ranking for this keyword",
    )
    serp_features: list[str] = Field(
        default_factory=list,
        description="SERP features present (PAA, featured snippet, etc.)",
    )
    money_pages: list[str] = Field(
        default_factory=list,
        description="Money page URLs to link to",
    )


class BriefGeneratorOutput(BaseModel):
    """Output from brief generator agent."""

    brief: ContentBriefResult


class BriefGeneratorAgent(BaseAgent[BriefGeneratorInput, BriefGeneratorOutput]):
    """Agent for generating comprehensive content briefs.

    Used in Step 12 to create writer-ready briefs that include:
    - Title options
    - Detailed outline with key points
    - Supporting keyword mapping
    - Examples and FAQ sections
    - Meta guidelines
    - Word count targets
    """

    model = "openai:gpt-4-turbo"
    temperature = 0.5

    @property
    def system_prompt(self) -> str:
        return """You are a content strategist creating SEO-optimized content briefs.

Given a topic cluster and brand context, generate a comprehensive brief that a writer can execute without additional research.

## Brief Requirements

### 1. Title Options (3-5)
- Include primary keyword naturally
- Compelling and click-worthy
- Match the search intent
- Examples: "How to X in 2024", "X vs Y: Complete Comparison", "10 Best X for Y"

### 2. Outline
Create a detailed H2/H3 structure with:
- Clear purpose for each section
- Key points to cover
- Supporting keywords to include naturally
- Logical flow that matches reader expectations

### 3. Content Specifications
- Target audience description
- Reader's job to be done (what they want to accomplish)
- Specific examples/case studies needed
- FAQ questions (from People Also Ask or inferred)

### 4. Word Count Guidelines
Base on intent and competition:
- Informational guides: 1500-3000 words
- Comparisons: 2000-4000 words
- How-to tutorials: 1000-2500 words
- Listicles: 1500-2500 words
- Glossary/definitions: 500-1000 words

### 5. Schema Type
Recommend appropriate schema:
- HowTo: Step-by-step guides
- FAQ: Content with Q&A sections
- Article: General informational content
- Product: Product-focused landing pages
- ItemList: Listicle content

Be specific and actionable. The writer should know exactly what to create."""

    @property
    def output_type(self) -> type[BriefGeneratorOutput]:
        return BriefGeneratorOutput

    def _build_prompt(self, input_data: BriefGeneratorInput) -> str:
        # Format supporting keywords
        supporting_kws = "\n".join(f"  - {kw}" for kw in input_data.supporting_keywords[:15])

        # Format SERP features
        serp_features = ", ".join(input_data.serp_features) if input_data.serp_features else "None identified"

        # Format competitor content types
        comp_types = ", ".join(input_data.competitors_content_types) if input_data.competitors_content_types else "Not analyzed"

        # Format money pages
        money_pages = "\n".join(f"  - {mp}" for mp in input_data.money_pages[:5]) if input_data.money_pages else "  None specified"

        return f"""Generate a comprehensive content brief for this topic:

## Topic Details
- **Topic Name**: {input_data.topic_name}
- **Primary Keyword**: {input_data.primary_keyword}
- **Search Intent**: {input_data.search_intent}
- **Page Type**: {input_data.page_type}
- **Funnel Stage**: {input_data.funnel_stage}

## Supporting Keywords
{supporting_kws}

## SERP Analysis
- **SERP Features**: {serp_features}
- **Competitor Content Types**: {comp_types}

## Brand Context
{input_data.brand_context if input_data.brand_context else "No brand context provided"}

## Money Pages to Link To
{money_pages}

---

Create a detailed brief with:
1. 3-5 compelling title options
2. Detailed outline (H2s and H3s with key points)
3. Target audience and reader intent
4. Required examples and FAQs
5. Meta title and description templates
6. Appropriate word count range
7. Recommended schema type"""
