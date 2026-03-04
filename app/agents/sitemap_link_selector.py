"""Agent for selecting the most relevant sitemap URLs for interlinking."""

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SitemapLinkCandidate(BaseModel):
    """Single sitemap URL candidate for agent evaluation."""

    url: str = Field(description="Candidate sitemap URL")
    topic: str = Field(description="Human-readable topic extracted from URL slug")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Heuristic relevance score")
    keyword_overlap: float = Field(ge=0.0, le=1.0, description="Keyword overlap with source brief")
    anchor_text: str = Field(default="", description="Suggested anchor text")


class SitemapLinkSelectorInput(BaseModel):
    """Input payload for sitemap URL selector agent."""

    source_primary_keyword: str
    source_working_title: str = Field(default="")
    source_topic_name: str = Field(default="")
    source_intent: str | None = Field(default=None)
    source_funnel_stage: str | None = Field(default=None)
    source_target_audience: str = Field(default="")
    source_reader_job: str = Field(default="")
    supporting_keywords: list[str] = Field(default_factory=list)
    max_links: int = Field(ge=1, le=10, default=5)
    candidates: list[SitemapLinkCandidate] = Field(default_factory=list)


class SitemapLinkDecision(BaseModel):
    """A selected sitemap URL with rationale."""

    url: str = Field(description="Selected URL from candidate set")
    rationale: str = Field(description="Short reason why this URL is a strong link target")


class SitemapLinkSelectorOutput(BaseModel):
    """Output payload for sitemap URL selection."""

    selected_urls: list[SitemapLinkDecision] = Field(default_factory=list)
    notes: str = Field(default="")


class SitemapLinkSelectorAgent(BaseAgent[SitemapLinkSelectorInput, SitemapLinkSelectorOutput]):
    """Agent that chooses the best sitemap URLs to interlink from a source brief."""

    model_tier = "standard"
    temperature = 0.2

    @property
    def system_prompt(self) -> str:
        return """You are an internal-link strategist for SEO articles.

Task:
- Select the sitemap URLs that are most contextually useful to link from the source article.

Rules:
1. Choose only from provided candidate URLs.
2. Prioritize topical relevance and reader usefulness over raw score.
3. Avoid near-duplicate targets that cover almost the same subtopic.
4. Prefer links that naturally deepen understanding and support likely reader next steps.
5. You may return fewer than max_links if only a smaller subset is high quality.
6. Do not invent URLs. Keep rationales concise.

Return selected URLs in ranked order (best first)."""

    @property
    def output_type(self) -> type[SitemapLinkSelectorOutput]:
        return SitemapLinkSelectorOutput

    def _build_prompt(self, input_data: SitemapLinkSelectorInput) -> str:
        logger.info(
            "Building sitemap link selector prompt",
            extra={
                "candidate_count": len(input_data.candidates),
                "max_links": input_data.max_links,
                "source_keyword": input_data.source_primary_keyword,
            },
        )

        candidate_lines = []
        for idx, candidate in enumerate(input_data.candidates):
            candidate_lines.append(
                f"{idx + 1}. url={candidate.url}"
                f" | topic={candidate.topic}"
                f" | relevance={candidate.relevance_score:.3f}"
                f" | keyword_overlap={candidate.keyword_overlap:.3f}"
                f" | anchor_text={candidate.anchor_text}"
            )

        supporting_keywords_text = ", ".join(input_data.supporting_keywords[:20])
        if not supporting_keywords_text:
            supporting_keywords_text = "none"

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "- none"

        return f"""Select the best sitemap URLs for internal linking.

Source article context:
- primary_keyword: {input_data.source_primary_keyword}
- working_title: {input_data.source_working_title or "none"}
- topic_name: {input_data.source_topic_name or "none"}
- intent: {input_data.source_intent or "unknown"}
- funnel_stage: {input_data.source_funnel_stage or "unknown"}
- target_audience: {input_data.source_target_audience or "unknown"}
- reader_job_to_be_done: {input_data.source_reader_job or "unknown"}
- supporting_keywords: {supporting_keywords_text}

Maximum links to select: {input_data.max_links}

Candidates:
{candidates_text}

Return:
- selected_urls: ranked list of chosen URLs (best first), each with short rationale
- notes: one brief note on selection strategy"""
