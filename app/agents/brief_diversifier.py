"""LLM selector for diversified brief topic batches in Step 12."""

import logging

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class BriefDiversifierTopicDecision(BaseModel):
    """Selection decision for one candidate topic."""

    topic_id: str = Field(description="Candidate topic ID")
    decision: str = Field(description="include or exclude")
    rationale: str = Field(description="Short reason for this decision")


class BriefDiversifierInput(BaseModel):
    """Input payload for brief diversification."""

    candidates: list[dict] = Field(default_factory=list)
    existing_topics: list[dict] = Field(default_factory=list)
    target_count: int = Field(ge=1, le=100, default=10)
    compact_mode: bool = Field(default=False)


class BriefDiversifierOutput(BaseModel):
    """Output from brief diversification selector."""

    decisions: list[BriefDiversifierTopicDecision] = Field(default_factory=list)
    overall_notes: str = Field(default="")


class BriefDiversifierAgent(BaseAgent[BriefDiversifierInput, BriefDiversifierOutput]):
    """Agent that selects a diverse, non-redundant subset of brief topics."""

    model_tier = "standard"
    temperature = 0.2

    @property
    def system_prompt(self) -> str:
        return """You are selecting a diversified content batch from candidate SEO topics.

Goal: include only topics that are meaningfully distinct and useful for the brand.

Rules:
1. Prefer strong variety across entities, intent/page type, and problem angle.
2. Avoid near-duplicate variants of the same article:
   - "servicenow pricing" vs "servicenow pricing model" usually one should be excluded.
   - exact comparison pair duplicates must not both be included.
3. Sibling comparisons are allowed when they compare different opponents:
   - "Zendesk vs Intercom" and "Zendesk vs Tidio" can both be included.
4. Use existing_topics history (with created_at) to avoid repeating recently covered topics.
5. Do not invent IDs. Only use candidate topic_id values.
6. Output fewer than target_count if many candidates are redundant.
7. Be conservative and quality-first.

Return decision for each candidate: include/exclude with concise rationale."""

    @property
    def output_type(self) -> type[BriefDiversifierOutput]:
        return BriefDiversifierOutput

    def _build_prompt(self, input_data: BriefDiversifierInput) -> str:
        logger.info(
            "Building brief diversification prompt",
            extra={
                "candidate_count": len(input_data.candidates),
                "existing_count": len(input_data.existing_topics),
                "target_count": input_data.target_count,
                "compact_mode": input_data.compact_mode,
            },
        )

        candidate_lines: list[str] = []
        for idx, candidate in enumerate(input_data.candidates):
            candidate_lines.append(
                f"{idx + 1}. topic_id={candidate.get('topic_id', '')}"
                f" | name={candidate.get('name', '')}"
                f" | primary_keyword={candidate.get('primary_keyword', '')}"
                f" | intent={candidate.get('intent', '')}"
                f" | page_type={candidate.get('page_type', '')}"
                f" | fit_tier={candidate.get('fit_tier', '')}"
                f" | priority_rank={candidate.get('priority_rank', '')}"
            )

        existing_cap = 30 if input_data.compact_mode else 60
        existing_lines: list[str] = []
        for row in input_data.existing_topics[:existing_cap]:
            existing_lines.append(
                f"- topic={row.get('topic_name', '')}"
                f" | primary_keyword={row.get('primary_keyword', '')}"
                f" | intent={row.get('search_intent', '')}"
                f" | page_type={row.get('page_type', '')}"
                f" | created_at={row.get('created_at', '')}"
            )

        existing_text = "\n".join(existing_lines) if existing_lines else "- none"
        candidates_text = "\n".join(candidate_lines) if candidate_lines else "- none"

        return f"""Select a diverse content batch.

Target include count: up to {input_data.target_count}

Candidate topics:
{candidates_text}

Existing covered topics (historical memory):
{existing_text}

For each candidate topic_id, return:
- decision: include or exclude
- rationale: concise reason tied to diversity/non-redundancy/brand usefulness

Quality over quantity: if candidates are redundant, include fewer topics."""
