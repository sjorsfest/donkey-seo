"""Summarizer agent for discovery iteration learnings."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LearningDraft(BaseModel):
    """Deterministic learning draft to be rewritten."""

    learning_key: str
    polarity: str
    title: str
    detail: str
    recommendation: str | None = None
    confidence: float | None = None
    current_metric: float | None = None
    delta_metric: float | None = None


class LearningRewrite(BaseModel):
    """Compact rewrite payload keyed by learning_key."""

    learning_key: str
    title: str
    detail: str
    recommendation: str | None = None
    confidence: float | None = None


class DiscoveryLearningSummarizerInput(BaseModel):
    """Input for the learning summarizer."""

    project_context: str = ""
    drafts: list[LearningDraft] = Field(default_factory=list)


class DiscoveryLearningSummarizerOutput(BaseModel):
    """Output containing rewritten learning text."""

    rewrites: list[LearningRewrite] = Field(default_factory=list)


class DiscoveryLearningSummarizerAgent(
    BaseAgent[DiscoveryLearningSummarizerInput, DiscoveryLearningSummarizerOutput]
):
    """Refine deterministic learning drafts into compact, readable guidance."""

    model_tier = "standard"
    temperature = 0.2

    @property
    def system_prompt(self) -> str:
        return """You rewrite SEO discovery learnings for execution clarity.

Rules:
1. Keep each rewrite concise, concrete, and easy to read by a human operator.
2. Preserve original meaning, polarity, and all numeric evidence.
3. Do not invent new evidence, metrics, causes, or recommendations.
4. Use plain language, not field labels, shorthand, or code-like tokens.
5. Expand machine-style tokens into readable phrases when present.
6. Keep titles action-oriented and specific.
7. Recommendations should be practical next-iteration guidance.
8. Return one rewrite per input learning_key."""

    @property
    def output_type(self) -> type[DiscoveryLearningSummarizerOutput]:
        return DiscoveryLearningSummarizerOutput

    def _build_prompt(self, input_data: DiscoveryLearningSummarizerInput) -> str:
        logger.info(
            "Building discovery learning summarizer prompt",
            extra={"draft_count": len(input_data.drafts)},
        )
        draft_lines: list[str] = []
        for draft in input_data.drafts:
            draft_lines.append(
                "\n".join([
                    f"learning_key: {draft.learning_key}",
                    f"polarity: {draft.polarity}",
                    f"title: {draft.title}",
                    f"detail: {draft.detail}",
                    f"recommendation: {draft.recommendation or ''}",
                    f"confidence: {draft.confidence}",
                    f"current_metric: {draft.current_metric}",
                    f"delta_metric: {draft.delta_metric}",
                ])
            )

        project_context = input_data.project_context or "No additional context."
        return (
            "Rewrite these deterministic discovery learnings for clarity while preserving evidence.\n\n"
            f"Project context:\n{project_context}\n\n"
            f"Drafts:\n{chr(10).join(draft_lines)}"
        )
