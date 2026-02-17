"""Reusable OpenAPI docs content for pipeline behavior."""

from __future__ import annotations

import json
from typing import Any

OPENAPI_PIPELINE_GUIDE_MARKDOWN = """
## Pipeline Architecture Guide

This API supports a two-stage SEO workflow:

1. Discovery stage (keyword/topic opportunity finding)
2. Content stage (briefs, writer instructions, and articles)

### Execution Modes

- `full`: runs a configured step range in a single run.
- `discovery_loop`: runs adaptive discovery iterations to produce enough accepted topics.
- `content_production`: generates content artifacts from eligible/selected topics.

### Current Content Granularity

The current model is:

- `1 topic -> 1 content brief -> 1 article`

Topic splitting should happen upstream in clustering/prioritization.
Content generation does not auto-create multiple articles for a single topic.

### Discovery Loop Behavior

- Iterates over Step `2 -> 8`.
- If project setup is incomplete, executes Step `0 -> 1` once before discovery iterations.
- Candidate topics are those with Step 7 fit tier `primary` or `secondary`.
- Topic acceptance uses a SERP gate on the topic primary keyword:
  - `difficulty <= max_keyword_difficulty`
  - `domain_diversity >= min_domain_diversity`
  - no `intent_mismatch` when `require_intent_match=true`
- Stores historical per-iteration decisions in discovery snapshots.
- Can auto-start content production with accepted topic IDs.
"""

_PIPELINE_GUIDE_JSON_OBJECT: dict[str, Any] = {
    "pipeline_model": {
        "content_granularity": "1_topic_to_1_brief_to_1_article",
        "topic_splitting_stage": "clustering_and_prioritization",
    },
    "modes": {
        "full": {
            "description": "Runs requested step range in one pipeline run.",
        },
        "discovery_loop": {
            "step_range_per_iteration": [2, 8],
            "bootstrap_steps_if_needed": [0, 1],
            "completion_rule": "accepted_topics >= target",
            "snapshot_endpoint": "/api/v1/pipeline/{project_id}/runs/{run_id}/discovery-snapshots",
            "auto_handoff": "optional auto-start of content_production",
        },
        "content_production": {
            "step_range": [12, 14],
            "inputs": [
                "selected_topic_ids",
                "max_briefs",
                "posts_per_week",
                "preferred_weekdays",
                "min_lead_days",
            ],
        },
    },
    "discovery_gate": {
        "fit_tiers": ["primary", "secondary"],
        "require_serp_gate": True,
        "criteria": {
            "max_keyword_difficulty": "<= threshold",
            "min_domain_diversity": ">= threshold",
            "intent_mismatch": "reject when require_intent_match=true",
        },
    },
}

OPENAPI_PIPELINE_GUIDE_JSON = json.dumps(
    _PIPELINE_GUIDE_JSON_OBJECT,
    indent=2,
    sort_keys=True,
)

PIPELINE_TAG_DESCRIPTION = """
Endpoints for orchestrating and observing pipeline runs.

- `full`: direct range execution.
- `discovery_loop`: adaptive topic discovery loop
  (Step 2-8 per iteration; Step 0-1 bootstrap when required).
- `content_production`: content-only pipeline (Step 12-14).

Use discovery snapshots to inspect accepted/rejected topic decisions per
iteration and power frontend dashboard logic.
"""

PIPELINE_START_EXAMPLES: dict[str, dict[str, Any]] = {
    "full_default": {
        "summary": "Full mode (default range-style start)",
        "description": "Runs the default full pipeline range from step 0 to step 14.",
        "value": {
            "mode": "full",
            "start_step": 0,
            "end_step": 14,
            "strategy": {
                "fit_threshold_profile": "aggressive",
            },
        },
    },
    "discovery_loop": {
        "summary": "Discovery loop with auto handoff",
        "description": (
            "Runs adaptive discovery until accepted topics reach target, "
            "then auto-starts content production."
        ),
        "value": {
            "mode": "discovery_loop",
            "strategy": {
                "scope_mode": "strict",
                "fit_threshold_profile": "aggressive",
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
                "auto_start_content": True,
            },
            "content": {
                "max_briefs": 20,
                "posts_per_week": 3,
                "preferred_weekdays": [0, 2, 4],
                "min_lead_days": 7,
            },
        },
    },
    "content_production": {
        "summary": "Content production only",
        "description": "Runs Step 12-14 using current eligible/selected topics.",
        "value": {
            "mode": "content_production",
            "content": {
                "max_briefs": 15,
                "posts_per_week": 2,
                "preferred_weekdays": [1, 3],
                "min_lead_days": 5,
            },
        },
    },
}

PIPELINE_START_OPENAPI_EXTRA = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": PIPELINE_START_EXAMPLES,
            }
        }
    }
}
