"""Stable capability keys for discovery learning memory."""

from __future__ import annotations

from typing import Iterable

CAPABILITY_SEED_GENERATION = "seed_generation"
CAPABILITY_KEYWORD_EXPANSION = "keyword_expansion"
CAPABILITY_INTENT_CLASSIFICATION = "intent_classification"
CAPABILITY_CLUSTERING = "clustering"
CAPABILITY_PRIORITIZATION = "prioritization"
CAPABILITY_SERP_VALIDATION = "serp_validation"

DISCOVERY_CAPABILITIES: frozenset[str] = frozenset({
    CAPABILITY_SEED_GENERATION,
    CAPABILITY_KEYWORD_EXPANSION,
    CAPABILITY_INTENT_CLASSIFICATION,
    CAPABILITY_CLUSTERING,
    CAPABILITY_PRIORITIZATION,
    CAPABILITY_SERP_VALIDATION,
})

# Legacy step-number mapping used for dual-read compatibility.
LEGACY_STEP_TO_CAPABILITY: dict[int, str] = {
    2: CAPABILITY_SEED_GENERATION,
    3: CAPABILITY_KEYWORD_EXPANSION,
    4: CAPABILITY_KEYWORD_EXPANSION,
    5: CAPABILITY_INTENT_CLASSIFICATION,
    6: CAPABILITY_CLUSTERING,
    7: CAPABILITY_PRIORITIZATION,
    8: CAPABILITY_SERP_VALIDATION,
}

DEFAULT_AGENT_BY_CAPABILITY: dict[str, str] = {
    CAPABILITY_SEED_GENERATION: "TopicGeneratorAgent",
    CAPABILITY_INTENT_CLASSIFICATION: "IntentClassifierAgent",
    CAPABILITY_CLUSTERING: "ClusterAgent",
    CAPABILITY_PRIORITIZATION: "PrioritizationAgent",
}


def normalize_capability_key(value: object) -> str | None:
    """Return normalized capability key if recognized."""
    normalized = str(value or "").strip().lower()
    if normalized in DISCOVERY_CAPABILITIES:
        return normalized
    return None


def capabilities_from_legacy_steps(steps: Iterable[object] | None) -> set[str]:
    """Map legacy step-number applicability to stable capability tags."""
    resolved: set[str] = set()
    if steps is None:
        return resolved

    for value in steps:
        try:
            step_num = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        capability = LEGACY_STEP_TO_CAPABILITY.get(step_num)
        if capability:
            resolved.add(capability)
    return resolved
