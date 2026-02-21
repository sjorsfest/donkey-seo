"""Discovery pipeline module definitions."""

from __future__ import annotations

from app.services.steps.discovery.step_02_seeds import SeedsInput, Step02SeedsService
from app.services.steps.discovery.step_03_expansion import ExpansionInput, Step03ExpansionService
from app.services.steps.discovery.step_04_metrics import MetricsInput, Step04MetricsService
from app.services.steps.discovery.step_05_intent import IntentInput, Step05IntentService
from app.services.steps.discovery.step_06_clustering import ClusteringInput, Step06ClusteringService
from app.services.steps.discovery.step_07_prioritization import (
    PrioritizationInput,
    Step07PrioritizationService,
)
from app.services.steps.discovery.step_08_serp import (
    SerpValidationInput,
    Step08SerpValidationService,
)

DISCOVERY_LOCAL_STEP_NAMES: dict[int, str] = {
    1: "seed_topics",
    2: "keyword_expansion",
    3: "keyword_metrics",
    4: "intent_labeling",
    5: "clustering",
    6: "prioritization",
    7: "serp_validation",
}

DISCOVERY_LOCAL_STEP_DEPENDENCIES: dict[int, list[int]] = {
    1: [],
    2: [1],
    3: [2],
    4: [3],
    5: [4],
    6: [5],
    7: [6],
}

DISCOVERY_LOCAL_TO_SERVICE = {
    1: Step02SeedsService,
    2: Step03ExpansionService,
    3: Step04MetricsService,
    4: Step05IntentService,
    5: Step06ClusteringService,
    6: Step07PrioritizationService,
    7: Step08SerpValidationService,
}

DISCOVERY_LOCAL_TO_INPUT = {
    1: SeedsInput,
    2: ExpansionInput,
    3: MetricsInput,
    4: IntentInput,
    5: ClusteringInput,
    6: PrioritizationInput,
    7: SerpValidationInput,
}

DISCOVERY_DEFAULT_START_STEP = 1
DISCOVERY_DEFAULT_END_STEP = 7
DISCOVERY_ITERATION_STEPS = (1, 2, 3, 4, 5, 6, 7)
