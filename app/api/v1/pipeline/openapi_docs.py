"""Reusable OpenAPI docs content for pipeline behavior."""

from __future__ import annotations

import json
from typing import Any

OPENAPI_PIPELINE_GUIDE_MARKDOWN = """
## Pipeline Architecture Guide

This API supports three independent pipeline modules:

1. `setup` (local steps 0-1)
2. `discovery` (local steps 2-8)
3. `content` (local steps 1-3)

### Execution Modes

- `setup`: project/bootstrap + brand profile preparation before the discovery loop.
- `discovery`: adaptive topic discovery loop with per-topic acceptance decisions.
- `content`: content production for selected/accepted topics.

### Dispatch Model

When discovery accepts a topic, content work is dispatched immediately as a task.
Dispatch dedupe is tracked in the database per discovery run (`parent_run_id + source_topic_id`).
"""

_PIPELINE_GUIDE_JSON_OBJECT: dict[str, Any] = {
    "modules": {
        "setup": {
            "local_steps": [0, 1],
            "purpose": "bootstrap project + brand profile",
        },
        "discovery": {
            "local_steps": [2, 8],
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
Endpoints for orchestrating and observing setup/discovery/content module runs.

- `setup`: setup bootstrap and brand profile validation.
- `discovery`: adaptive topic discovery loop (steps 2-8).
- `content`: content-only module run.
"""

PIPELINE_START_EXAMPLES: dict[str, dict[str, Any]] = {
    "setup": {
        "summary": "Setup module",
        "description": "Runs setup steps 0-1 before discovery loop execution.",
        "value": {
            "mode": "setup",
            "start_step": 0,
            "end_step": 1,
        },
    },
    "discovery": {
        "summary": "Discovery module",
        "description": "Runs adaptive discovery loop (steps 2-8) and dispatches content tasks when topics are accepted.",
        "value": {
            "mode": "discovery",
            "start_step": 2,
            "end_step": 8,
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
