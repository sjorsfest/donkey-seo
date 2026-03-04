"""Run-level strategy defaults and merge helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeVar

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
IntentMixMode = Literal["adaptive_auto"]
FunnelMixMode = Literal["derived_soft_adjust"]
IntentKey = Literal["informational", "commercial", "transactional"]
FunnelKey = Literal["tofu", "mofu", "bofu"]

TRACKED_INTENTS: tuple[IntentKey, ...] = ("informational", "commercial", "transactional")
TRACKED_FUNNELS: tuple[FunnelKey, ...] = ("tofu", "mofu", "bofu")
INTENT_TO_FUNNEL: dict[IntentKey, FunnelKey] = {
    "informational": "tofu",
    "commercial": "mofu",
    "transactional": "bofu",
}
DEFAULT_INTENT_MIX: dict[IntentKey, float] = {
    "informational": 0.40,
    "commercial": 0.35,
    "transactional": 0.25,
}
DEFAULT_FUNNEL_MIX: dict[FunnelKey, float] = {
    "tofu": 0.40,
    "mofu": 0.35,
    "bofu": 0.25,
}
DEFAULT_CONVERSION_INTENTS: list[str] = []

THRESHOLD_PROFILES: dict[FitThresholdProfile, dict[str, float | int]] = {
    "aggressive": {"base": 0.70, "relaxed": 0.62, "target": 6},
    "moderate": {"base": 0.60, "relaxed": 0.54, "target": 8},
    "lenient": {"base": 0.50, "relaxed": 0.45, "target": 10},
}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _normalize_share_map(
    values: Mapping[str, Any] | None,
    defaults: Mapping[KeyT, float],
) -> dict[KeyT, float]:
    normalized: dict[KeyT, float] = {key: float(defaults[key]) for key in defaults}
    if values:
        for key in defaults:
            raw = values.get(key)
            if raw is None:
                continue
            try:
                normalized[key] = max(0.0, float(raw))
            except (TypeError, ValueError):
                continue

    total = sum(normalized.values())
    if total <= 0:
        return {key: float(defaults[key]) for key in defaults}

    return {key: value / total for key, value in normalized.items()}


def derive_funnel_mix_from_intent(intent_mix: dict[IntentKey, float]) -> dict[FunnelKey, float]:
    """Map intent mix to default funnel mix."""
    return {
        "tofu": float(intent_mix.get("informational", 0.0)),
        "mofu": float(intent_mix.get("commercial", 0.0)),
        "bofu": float(intent_mix.get("transactional", 0.0)),
    }


def build_adaptive_target_mix(
    *,
    base_mix: Mapping[KeyT, float],
    observed_mix: Mapping[str, float],
    influence: float,
    floor: float = 0.08,
) -> dict[KeyT, float]:
    """Softly adapt target shares based on observed distribution gaps."""
    clamped_influence = max(0.0, min(1.0, influence))
    adjusted: dict[KeyT, float] = {}
    for key, base_share in base_mix.items():
        observed_share = max(0.0, float(observed_mix.get(key, 0.0)))
        gap = base_share - observed_share
        adjusted[key] = max(floor, base_share + (gap * (0.6 * clamped_influence)))

    total = sum(adjusted.values())
    if total <= 0:
        return {key: (1.0 / max(len(base_mix), 1)) for key in base_mix}
    return {key: (value / total) for key, value in adjusted.items()}


def normalize_intent_label(value: str | None) -> IntentKey | None:
    if not value:
        return None
    normalized = str(value).strip().lower()
    if normalized in TRACKED_INTENTS:
        return normalized  # type: ignore[return-value]
    return None


def normalize_funnel_label(value: str | None) -> FunnelKey | None:
    if not value:
        return None
    normalized = str(value).strip().lower()
    if normalized in TRACKED_FUNNELS:
        return normalized  # type: ignore[return-value]
    if normalized == "awareness":
        return "tofu"
    if normalized in {"consideration", "evaluation"}:
        return "mofu"
    if normalized in {"decision", "purchase", "conversion"}:
        return "bofu"
    return None


def funnel_from_intent(intent: str | None) -> FunnelKey | None:
    key = normalize_intent_label(intent)
    if not key:
        return None
    return INTENT_TO_FUNNEL[key]


@dataclass(slots=True)
class IntentMixConfig:
    """Soft balancing settings for informational/commercial/transactional split."""

    mode: IntentMixMode = "adaptive_auto"
    informational: float = DEFAULT_INTENT_MIX["informational"]
    commercial: float = DEFAULT_INTENT_MIX["commercial"]
    transactional: float = DEFAULT_INTENT_MIX["transactional"]
    influence: float = 0.35

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "IntentMixConfig":
        payload = payload or {}
        mode = payload.get("mode", "adaptive_auto")
        if mode != "adaptive_auto":
            mode = "adaptive_auto"
        shares = _normalize_share_map(payload, DEFAULT_INTENT_MIX)
        influence_raw = payload.get("influence", 0.35)
        try:
            influence = float(influence_raw)
        except (TypeError, ValueError):
            influence = 0.35
        influence = max(0.0, min(1.0, influence))
        return cls(
            mode=mode,
            informational=shares["informational"],
            commercial=shares["commercial"],
            transactional=shares["transactional"],
            influence=influence,
        )

    def to_shares(self) -> dict[IntentKey, float]:
        return {
            "informational": float(self.informational),
            "commercial": float(self.commercial),
            "transactional": float(self.transactional),
        }

    def to_dict(self) -> dict[str, Any]:
        shares = self.to_shares()
        return {
            "mode": self.mode,
            **shares,
            "influence": self.influence,
        }


@dataclass(slots=True)
class FunnelMixConfig:
    """Soft balancing settings for TOFU/MOFU/BOFU split."""

    mode: FunnelMixMode = "derived_soft_adjust"
    tofu: float = DEFAULT_FUNNEL_MIX["tofu"]
    mofu: float = DEFAULT_FUNNEL_MIX["mofu"]
    bofu: float = DEFAULT_FUNNEL_MIX["bofu"]
    influence: float = 0.30

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        *,
        intent_mix: IntentMixConfig | None = None,
    ) -> "FunnelMixConfig":
        payload = payload or {}
        mode = payload.get("mode", "derived_soft_adjust")
        if mode != "derived_soft_adjust":
            mode = "derived_soft_adjust"

        inferred_base = DEFAULT_FUNNEL_MIX
        if intent_mix is not None:
            inferred_base = derive_funnel_mix_from_intent(intent_mix.to_shares())

        shares = _normalize_share_map(payload, inferred_base)
        influence_raw = payload.get("influence", 0.30)
        try:
            influence = float(influence_raw)
        except (TypeError, ValueError):
            influence = 0.30
        influence = max(0.0, min(1.0, influence))
        return cls(
            mode=mode,
            tofu=shares["tofu"],
            mofu=shares["mofu"],
            bofu=shares["bofu"],
            influence=influence,
        )

    def to_shares(self) -> dict[FunnelKey, float]:
        return {
            "tofu": float(self.tofu),
            "mofu": float(self.mofu),
            "bofu": float(self.bofu),
        }

    def to_dict(self) -> dict[str, Any]:
        shares = self.to_shares()
        return {
            "mode": self.mode,
            **shares,
            "influence": self.influence,
        }


@dataclass(slots=True)
class RunStrategy:
    """Resolved strategy used by ranking and filtering steps."""

    conversion_intents: list[str] = field(default_factory=lambda: list(DEFAULT_CONVERSION_INTENTS))
    intent_mix: IntentMixConfig = field(default_factory=IntentMixConfig)
    funnel_mix: FunnelMixConfig = field(default_factory=FunnelMixConfig)
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

        intent_mix = IntentMixConfig.from_dict(payload.get("intent_mix"))
        funnel_mix = FunnelMixConfig.from_dict(
            payload.get("funnel_mix"),
            intent_mix=intent_mix,
        )

        return cls(
            conversion_intents=list(payload.get("conversion_intents") or []),
            intent_mix=intent_mix,
            funnel_mix=funnel_mix,
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
            "intent_mix": self.intent_mix.to_dict(),
            "funnel_mix": self.funnel_mix.to_dict(),
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
) -> RunStrategy:
    """Merge run overrides with brand defaults and system defaults."""
    strategy = RunStrategy.from_dict(strategy_payload)

    brand_include = list(brand.in_scope_topics or []) if brand else []
    brand_exclude = list(brand.out_of_scope_topics or []) if brand else []
    brand_roles = list(brand.target_roles or []) if brand else []
    brand_industries = list(brand.target_industries or []) if brand else []
    brand_pains = list(brand.primary_pains or []) if brand else []

    strategy.include_topics = _dedupe(brand_include + strategy.include_topics)
    strategy.exclude_topics = _dedupe(brand_exclude + strategy.exclude_topics)
    strategy.icp_roles = _dedupe(strategy.icp_roles or brand_roles)
    strategy.icp_industries = _dedupe(strategy.icp_industries or brand_industries)
    strategy.icp_pains = _dedupe(strategy.icp_pains or brand_pains)
    strategy.conversion_intents = _dedupe(strategy.conversion_intents)

    return strategy
KeyT = TypeVar("KeyT", bound=str)
