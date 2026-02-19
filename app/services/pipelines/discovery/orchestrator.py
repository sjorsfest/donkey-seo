"""Discovery pipeline module definitions."""

from __future__ import annotations

from app.services.steps.discovery.step_00_setup import SetupInput, Step00SetupService
from app.services.steps.discovery.step_01_brand import BrandInput, Step01BrandService
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
    0: "setup",
    1: "brand_profile",
    2: "seed_topics",
    3: "keyword_expansion",
    4: "keyword_metrics",
    5: "intent_labeling",
    6: "clustering",
    7: "prioritization",
    8: "serp_validation",
}

DISCOVERY_LOCAL_STEP_DEPENDENCIES: dict[int, list[int]] = {
    0: [],
    1: [0],
    2: [1],
    3: [2],
    4: [3],
    5: [4],
    6: [5],
    7: [6],
    8: [7],
}

DISCOVERY_LOCAL_TO_SERVICE = {
    0: Step00SetupService,
    1: Step01BrandService,
    2: Step02SeedsService,
    3: Step03ExpansionService,
    4: Step04MetricsService,
    5: Step05IntentService,
    6: Step06ClusteringService,
    7: Step07PrioritizationService,
    8: Step08SerpValidationService,
}

DISCOVERY_LOCAL_TO_INPUT = {
    0: SetupInput,
    1: BrandInput,
    2: SeedsInput,
    3: ExpansionInput,
    4: MetricsInput,
    5: IntentInput,
    6: ClusteringInput,
    7: PrioritizationInput,
    8: SerpValidationInput,
}

DISCOVERY_DEFAULT_START_STEP = 1
DISCOVERY_DEFAULT_END_STEP = 8
DISCOVERY_ITERATION_STEPS = (2, 3, 4, 5, 6, 7, 8)
