"""Pipeline schemas."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ScopeMode = Literal["strict", "balanced_adjacent", "broad_education"]
BrandedKeywordMode = Literal["comparisons_only", "exclude_all", "allow_all"]
FitThresholdProfile = Literal["aggressive", "moderate", "lenient"]
MarketModeOverride = Literal[
    "auto",
    "established_category",
    "fragmented_workflow",
    "mixed",
]
PipelineMode = Literal["setup", "discovery", "content"]


class PipelineRunStrategy(BaseModel):
    """Run-scoped strategy overrides for product/ICP fit."""

    conversion_intents: list[str] = Field(default_factory=list)
    scope_mode: ScopeMode = "balanced_adjacent"
    branded_keyword_mode: BrandedKeywordMode = "comparisons_only"
    fit_threshold_profile: FitThresholdProfile = "aggressive"
    include_topics: list[str] = Field(default_factory=list)
    exclude_topics: list[str] = Field(default_factory=list)
    icp_roles: list[str] = Field(default_factory=list)
    icp_industries: list[str] = Field(default_factory=list)
    icp_pains: list[str] = Field(default_factory=list)
    min_eligible_target: int | None = Field(default=None, ge=1, le=100)
    market_mode_override: MarketModeOverride = "auto"


class DiscoveryLoopConfig(BaseModel):
    """Configuration for discovery module."""

    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of adaptive discovery iterations before pausing.",
    )
    min_eligible_topics: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Minimum accepted topics required to complete discovery.",
    )
    require_serp_gate: bool = Field(
        default=True,
        description="Whether topic acceptance must pass SERP-based gate checks.",
    )
    max_keyword_difficulty: float = Field(
        default=65.0,
        ge=0.0,
        le=100.0,
        description="Maximum allowed keyword difficulty for accepted discovery topics.",
    )
    min_domain_diversity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum unique-domain ratio in top SERP results for acceptance.",
    )
    require_intent_match: bool = Field(
        default=True,
        description="Reject topics with SERP intent mismatch on primary keyword.",
    )
    max_serp_servedness: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "In workflow markets, reject a topic when SERP servedness is above this level "
            "and competitor density is also high."
        ),
    )
    max_serp_competitor_density: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Maximum allowed vendor/competitor density for workflow-cluster acceptance.",
    )
    min_serp_intent_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cluster-level SERP intent confidence when intent matching is required."
        ),
    )
    auto_dispatch_content_tasks: bool = Field(
        default=True,
        description="Dispatch content tasks when topics are accepted during discovery.",
    )
    auto_resume_on_exhaustion: bool = Field(
        default=False,
        description=(
            "When max_iterations is reached without enough accepted topics, pause and "
            "auto-resume discovery after a cooldown."
        ),
    )
    exhaustion_cooldown_minutes: int = Field(
        default=60,
        ge=1,
        le=10080,
        description=(
            "Cooldown delay before auto-resuming discovery after max_iterations exhaustion."
        ),
    )


class ContentPipelineConfig(BaseModel):
    """Configuration for content module."""

    max_briefs: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of content briefs to generate in Step 12.",
    )
    posts_per_week: int = Field(
        default=1,
        ge=1,
        le=7,
        description="Publishing cadence for Step 12 calendar assignment.",
    )
    preferred_weekdays: list[int] = Field(
        default_factory=list,
        description="Preferred publish weekdays where Monday=0 ... Sunday=6.",
    )
    min_lead_days: int = Field(
        default=7,
        ge=1,
        le=60,
        description="Minimum days from today before the first proposed publish date.",
    )
    publication_start_date: date | None = Field(
        default=None,
        description="Optional earliest publication date override (YYYY-MM-DD).",
    )
    use_llm_timing_hints: bool = Field(
        default=True,
        description="Whether Step 12 may use LLM-suggested timing within schedule bounds.",
    )
    llm_timing_flex_days: int = Field(
        default=14,
        ge=0,
        le=90,
        description="Maximum day distance for snapping to an LLM timing hint.",
    )
    include_zero_data_topics: bool = Field(
        default=True,
        description=(
            "Reserve part of Step 12 capacity for high-fit topics whose primary keyword has "
            "missing demand metrics."
        ),
    )
    zero_data_topic_share: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description=(
            "Max share of Step 12 brief slots reserved for high-fit zero-data topics "
            "(0.0-0.5)."
        ),
    )
    zero_data_fit_score_min: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum fit_score required for zero-data topic inclusion.",
    )

    @field_validator("preferred_weekdays")
    @classmethod
    def validate_preferred_weekdays(cls, value: list[int]) -> list[int]:
        """Ensure weekday indexes are valid (0=Mon ... 6=Sun)."""
        invalid = [day for day in value if day < 0 or day > 6]
        if invalid:
            raise ValueError("preferred_weekdays must contain only values 0..6")
        return value


class PipelineStartRequest(BaseModel):
    """Schema for starting a pipeline run."""

    mode: PipelineMode = Field(
        default="discovery",
        description=(
            "Pipeline orchestration mode: setup, discovery, or content."
        ),
    )
    start_step: int | None = None
    end_step: int | None = None
    skip_steps: list[int] | None = None
    strategy: PipelineRunStrategy | None = None
    discovery: DiscoveryLoopConfig | None = Field(
        default=None,
        description="Optional discovery controls (used when mode=discovery).",
    )
    content: ContentPipelineConfig | None = Field(
        default=None,
        description="Optional content controls (used when mode=content).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mode": "discovery",
                    "start_step": 1,
                    "end_step": 7,
                    "strategy": {
                        "fit_threshold_profile": "aggressive",
                    },
                },
                {
                    "mode": "setup",
                    "start_step": 1,
                    "end_step": 5,
                },
                {
                    "mode": "discovery",
                    "strategy": {
                        "scope_mode": "strict",
                        "fit_threshold_profile": "aggressive",
                        "market_mode_override": "auto",
                        "include_topics": ["customer support automation"],
                        "exclude_topics": ["medical advice"],
                    },
                    "discovery": {
                        "max_iterations": 3,
                        "min_eligible_topics": 8,
                        "require_serp_gate": True,
                        "max_keyword_difficulty": 65.0,
                        "min_domain_diversity": 0.5,
                        "require_intent_match": True,
                        "max_serp_servedness": 0.75,
                        "max_serp_competitor_density": 0.70,
                        "min_serp_intent_confidence": 0.35,
                        "auto_dispatch_content_tasks": True,
                        "auto_resume_on_exhaustion": True,
                        "exhaustion_cooldown_minutes": 60,
                    },
                    "content": {
                        "max_briefs": 20,
                        "posts_per_week": 3,
                        "preferred_weekdays": [0, 2, 4],
                        "min_lead_days": 7,
                    },
                },
                {
                    "mode": "content",
                    "content": {
                        "max_briefs": 15,
                        "posts_per_week": 2,
                        "preferred_weekdays": [1, 3],
                        "min_lead_days": 5,
                    },
                },
            ]
        }
    }


class StepExecutionResponse(BaseModel):
    """Schema for step execution response."""

    id: str
    step_number: int
    step_name: str
    status: str
    progress_percent: float
    progress_message: str | None
    items_processed: int
    items_total: int | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None

    model_config = {"from_attributes": True}


class PipelineRunResponse(BaseModel):
    """Schema for pipeline run response."""

    id: str
    project_id: str
    pipeline_module: str
    parent_run_id: str | None
    source_topic_id: str | None
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    start_step: int | None
    end_step: int | None
    skip_steps: list[int] | None
    step_executions: list[StepExecutionResponse]
    created_at: datetime

    model_config = {"from_attributes": True}


class PipelineProgressResponse(BaseModel):
    """Schema for real-time pipeline progress."""

    run_id: str
    status: str
    current_step: int | None
    current_step_name: str | None
    overall_progress: float
    steps: list[StepExecutionResponse]


class DiscoveryTopicSnapshotResponse(BaseModel):
    """Snapshot response for discovery-loop topic decisions."""

    id: str
    project_id: str
    pipeline_run_id: str
    iteration_index: int
    source_topic_id: str | None
    topic_name: str
    fit_tier: str | None
    fit_score: float | None
    keyword_difficulty: float | None
    domain_diversity: float | None
    validated_intent: str | None
    validated_page_type: str | None
    top_domains: list[str] | None
    decision: str
    rejection_reasons: list[str] | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
