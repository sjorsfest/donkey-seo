"""Run-level strategy defaults and merge helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.models.brand import BrandProfile

ScopeMode = Literal["strict", "balanced_adjacent", "broad_education"]
BrandedKeywordMode = Literal["comparisons_only", "exclude_all", "allow_all"]
FitThresholdProfile = Literal["aggressive", "moderate", "lenient"]
MarketModeOverride = Literal[
    "auto",
    "established_category",
    "fragmented_workflow",
    "mixed",
]

THRESHOLD_PROFILES: dict[FitThresholdProfile, dict[str, float | int]] = {
    "aggressive": {"base": 0.70, "relaxed": 0.62, "target": 6},
    "moderate": {"base": 0.60, "relaxed": 0.54, "target": 8},
    "lenient": {"base": 0.50, "relaxed": 0.45, "target": 10},
}


def _normalize_goal_text(value: str) -> str:
    """Normalize free-form goal text for matching."""
    normalized = value.strip().lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", normalized)


def _default_conversion_intents_for_goal(primary_goal: str | None) -> list[str]:
    """Map high-level goal presets to richer conversion intent hints."""
    if not primary_goal:
        return ["lead_generation", "commercial", "transactional"]

    normalized = _normalize_goal_text(primary_goal)

    if re.search(r"(traffic|awareness|discover|visibility|authority|education)", normalized):
        return ["traffic_growth", "awareness", "education", "commercial"]
    if re.search(r"(lead|pipeline|demand gen|demand generation)", normalized):
        return [
            "lead_generation",
            "commercial",
            "transactional",
            "demo",
            "trial",
            "contact",
        ]
    if re.search(r"(revenue|sales|money|conversion|purchase|pricing)", normalized):
        return [
            "revenue_content",
            "transactional",
            "commercial",
            "pricing",
            "purchase",
        ]

    return [primary_goal.strip()]


@dataclass(slots=True)
class GoalIntentProfile:
    """Intent preference profile derived from run conversion intents."""

    profile_name: str
    core_intents: set[str]
    adjacent_intents: set[str]


def build_goal_intent_profile(conversion_intents: list[str]) -> GoalIntentProfile:
    """Translate conversion-intent hints into target SERP intent preferences."""
    text = " ".join(_normalize_goal_text(item) for item in conversion_intents if item)

    has_traffic = bool(re.search(r"(traffic|awareness|authority|education|discover)", text))
    has_lead = bool(re.search(r"(lead|pipeline|demo|trial|signup|contact|book)", text))
    has_revenue = bool(re.search(r"(revenue|sales|purchase|pricing|transaction)", text))

    core: set[str] = set()
    adjacent: set[str] = set()

    if has_traffic:
        core.update({"informational", "commercial"})
        adjacent.update({"transactional"})
    if has_lead:
        core.update({"commercial", "transactional"})
        adjacent.update({"informational"})
    if has_revenue:
        core.update({"transactional", "commercial"})
        adjacent.update({"informational"})

    if not core:
        core.update({"informational", "commercial", "transactional"})
        adjacent.update({"navigational"})
        return GoalIntentProfile(
            profile_name="balanced",
            core_intents=core,
            adjacent_intents=adjacent - core,
        )

    if has_revenue and not has_traffic and not has_lead:
        profile_name = "revenue_content"
    elif has_lead and not has_traffic and not has_revenue:
        profile_name = "lead_gen"
    elif has_traffic and not has_lead and not has_revenue:
        profile_name = "traffic_growth"
    else:
        profile_name = "mixed_goals"

    return GoalIntentProfile(
        profile_name=profile_name,
        core_intents=core,
        adjacent_intents=adjacent - core,
    )


def classify_intent_alignment(
    observed_intent: str | None,
    profile: GoalIntentProfile,
) -> Literal["core", "adjacent", "off_goal", "unknown"]:
    """Classify how well observed intent aligns with goal intent profile."""
    if not observed_intent:
        return "unknown"
    normalized = observed_intent.strip().lower()
    if normalized in profile.core_intents:
        return "core"
    if normalized in profile.adjacent_intents:
        return "adjacent"
    return "off_goal"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


@dataclass(slots=True)
class RunStrategy:
    """Resolved strategy used by ranking and filtering steps."""

    conversion_intents: list[str] = field(default_factory=lambda: ["lead_generation"])
    scope_mode: ScopeMode = "balanced_adjacent"
    branded_keyword_mode: BrandedKeywordMode = "comparisons_only"
    fit_threshold_profile: FitThresholdProfile = "aggressive"
    include_topics: list[str] = field(default_factory=list)
    exclude_topics: list[str] = field(default_factory=list)
    icp_roles: list[str] = field(default_factory=list)
    icp_industries: list[str] = field(default_factory=list)
    icp_pains: list[str] = field(default_factory=list)
    min_eligible_target: int | None = None
    market_mode_override: MarketModeOverride = "auto"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RunStrategy":
        if not payload:
            return cls()

        fit_profile = payload.get("fit_threshold_profile", "aggressive")
        if fit_profile not in THRESHOLD_PROFILES:
            fit_profile = "aggressive"
        market_mode_override = payload.get("market_mode_override", "auto")
        if market_mode_override not in {
            "auto",
            "established_category",
            "fragmented_workflow",
            "mixed",
        }:
            market_mode_override = "auto"

        return cls(
            conversion_intents=list(payload.get("conversion_intents") or []),
            scope_mode=payload.get("scope_mode", "balanced_adjacent"),
            branded_keyword_mode=payload.get("branded_keyword_mode", "comparisons_only"),
            fit_threshold_profile=fit_profile,
            include_topics=list(payload.get("include_topics") or []),
            exclude_topics=list(payload.get("exclude_topics") or []),
            icp_roles=list(payload.get("icp_roles") or []),
            icp_industries=list(payload.get("icp_industries") or []),
            icp_pains=list(payload.get("icp_pains") or []),
            min_eligible_target=payload.get("min_eligible_target"),
            market_mode_override=market_mode_override,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversion_intents": self.conversion_intents,
            "scope_mode": self.scope_mode,
            "branded_keyword_mode": self.branded_keyword_mode,
            "fit_threshold_profile": self.fit_threshold_profile,
            "include_topics": self.include_topics,
            "exclude_topics": self.exclude_topics,
            "icp_roles": self.icp_roles,
            "icp_industries": self.icp_industries,
            "icp_pains": self.icp_pains,
            "min_eligible_target": self.min_eligible_target,
            "market_mode_override": self.market_mode_override,
        }

    def base_threshold(self) -> float:
        return float(THRESHOLD_PROFILES[self.fit_threshold_profile]["base"])

    def relaxed_threshold(self) -> float:
        return float(THRESHOLD_PROFILES[self.fit_threshold_profile]["relaxed"])

    def eligible_target(self) -> int:
        if self.min_eligible_target is not None:
            return max(1, self.min_eligible_target)
        return int(THRESHOLD_PROFILES[self.fit_threshold_profile]["target"])


def resolve_run_strategy(
    strategy_payload: dict[str, Any] | None,
    brand: BrandProfile | None,
    primary_goal: str | None,
) -> RunStrategy:
    """Merge run overrides with brand defaults and system defaults."""
    strategy = RunStrategy.from_dict(strategy_payload)

    brand_include = list(brand.in_scope_topics or []) if brand else []
    brand_exclude = list(brand.out_of_scope_topics or []) if brand else []
    brand_roles = list(brand.target_roles or []) if brand else []
    brand_industries = list(brand.target_industries or []) if brand else []
    brand_pains = list(brand.primary_pains or []) if brand else []

    has_explicit_conversion_intents = bool(
        strategy_payload
        and isinstance(strategy_payload.get("conversion_intents"), list)
        and strategy_payload.get("conversion_intents")
    )
    if primary_goal and not has_explicit_conversion_intents:
        strategy.conversion_intents = _default_conversion_intents_for_goal(primary_goal)
    if not strategy.conversion_intents:
        strategy.conversion_intents = _default_conversion_intents_for_goal("lead_generation")

    strategy.include_topics = _dedupe(brand_include + strategy.include_topics)
    strategy.exclude_topics = _dedupe(brand_exclude + strategy.exclude_topics)
    strategy.icp_roles = _dedupe(strategy.icp_roles or brand_roles)
    strategy.icp_industries = _dedupe(strategy.icp_industries or brand_industries)
    strategy.icp_pains = _dedupe(strategy.icp_pains or brand_pains)
    strategy.conversion_intents = _dedupe(strategy.conversion_intents)

    return strategy
