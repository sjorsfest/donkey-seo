"""Run-level strategy defaults and merge helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.models.brand import BrandProfile

ScopeMode = Literal["strict", "balanced_adjacent", "broad_education"]
BrandedKeywordMode = Literal["comparisons_only", "exclude_all", "allow_all"]
FitThresholdProfile = Literal["aggressive", "moderate", "lenient"]

THRESHOLD_PROFILES: dict[FitThresholdProfile, dict[str, float | int]] = {
    "aggressive": {"base": 0.70, "relaxed": 0.62, "target": 6},
    "moderate": {"base": 0.60, "relaxed": 0.54, "target": 8},
    "lenient": {"base": 0.50, "relaxed": 0.45, "target": 10},
}


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RunStrategy":
        if not payload:
            return cls()

        fit_profile = payload.get("fit_threshold_profile", "aggressive")
        if fit_profile not in THRESHOLD_PROFILES:
            fit_profile = "aggressive"

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

    if primary_goal and not strategy.conversion_intents:
        strategy.conversion_intents = [primary_goal]
    if not strategy.conversion_intents:
        strategy.conversion_intents = ["lead_generation"]

    strategy.include_topics = _dedupe(brand_include + strategy.include_topics)
    strategy.exclude_topics = _dedupe(brand_exclude + strategy.exclude_topics)
    strategy.icp_roles = _dedupe(strategy.icp_roles or brand_roles)
    strategy.icp_industries = _dedupe(strategy.icp_industries or brand_industries)
    strategy.icp_pains = _dedupe(strategy.icp_pains or brand_pains)
    strategy.conversion_intents = _dedupe(strategy.conversion_intents)

    return strategy
