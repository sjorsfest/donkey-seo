"""Reusable OpenAPI docs content for pipeline behavior."""

from __future__ import annotations

import json
from typing import Any

OPENAPI_PIPELINE_GUIDE_MARKDOWN = """
## Pipeline Architecture Guide

This API supports two independent pipeline modules:

1. `discovery` (local steps 1-8, setup bootstrap step 0 when needed)
2. `content` (local steps 1-3)

### Execution Modes

- `discovery`: adaptive topic discovery with per-topic acceptance decisions.
- `content`: content production for selected/accepted topics.

### Dispatch Model

When discovery accepts a topic, content work is dispatched immediately as a task.
Dispatch dedupe is tracked in the database per discovery run (`parent_run_id + source_topic_id`).
"""

_PIPELINE_GUIDE_JSON_OBJECT: dict[str, Any] = {
    "modules": {
        "discovery": {
            "local_steps": [1, 8],
            "bootstrap_step": 0,
            "dispatch_behavior": "accepted topics enqueue content runs immediately",
            "snapshot_endpoint": "/api/v1/pipeline/{project_id}/runs/{run_id}/discovery-snapshots",
        },
        "content": {
            "local_steps": [1, 3],
            "granularity": "1_topic_to_1_brief_to_1_article",
        },
    },
    "dedupe": {
        "scope": "per discovery run",
        "key": ["parent_run_id", "source_topic_id", "pipeline_module=content"],
    },
}

OPENAPI_PIPELINE_GUIDE_JSON = json.dumps(
    _PIPELINE_GUIDE_JSON_OBJECT,
    indent=2,
    sort_keys=True,
)

PIPELINE_TAG_DESCRIPTION = """
Endpoints for orchestrating and observing discovery/content module runs.

- `discovery`: adaptive topic discovery loop.
- `content`: content-only module run.
"""

PIPELINE_START_EXAMPLES: dict[str, dict[str, Any]] = {
    "discovery": {
        "summary": "Discovery module",
        "description": "Runs adaptive discovery and dispatches content tasks when topics are accepted.",
        "value": {
            "mode": "discovery",
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
            },
            "content": {
                "max_briefs": 20,
                "posts_per_week": 3,
                "preferred_weekdays": [0, 2, 4],
                "min_lead_days": 7,
            },
        },
    },
    "content": {
        "summary": "Content module",
        "description": "Runs content steps 1-3 with content controls.",
        "value": {
            "mode": "content",
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
