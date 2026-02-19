"""Market diagnosis and workflow-signal utilities for discovery steps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal

MarketMode = Literal["established_category", "fragmented_workflow", "mixed"]
MarketModeOverride = Literal[
    "auto",
    "established_category",
    "fragmented_workflow",
    "mixed",
]

WORKFLOW_VERBS = {
    "connect",
    "integrate",
    "integration",
    "sync",
    "send",
    "forward",
    "route",
    "webhook",
    "automate",
    "notify",
    "import",
    "export",
}
INTEGRATION_TERMS = {"integrate", "integration", "webhook", "api", "connector", "sync"}
COMPARISON_TERMS = {
    "alternative",
    "alternatives",
    "replace",
    "replacement",
    "vs",
    "versus",
    "instead",
    "instead of",
}
MODIFIER_TERMS = {"for", "with", "without", "no", "best"}
UGC_DOC_DOMAINS = {
    "reddit.com",
    "quora.com",
    "stackoverflow.com",
    "stackexchange.com",
    "github.com",
    "gitlab.com",
    "medium.com",
    "dev.to",
}
CORE_WORKFLOW_NOUNS = {
    "webhook",
    "integration",
    "api",
    "automation",
    "workflow",
    "bot",
    "notification",
}


@dataclass(slots=True)
class MarketDiagnosisResult:
    """Normalized market diagnosis payload."""

    mode: MarketMode
    signals: dict[str, Any]
    reasons: list[str]
    source: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "signals": self.signals,
            "reasons": self.reasons,
            "source": self.source,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
        }


def normalize_market_mode_override(value: str | None) -> MarketMode | None:
    """Return explicit mode override when provided, otherwise None for auto."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "auto" or not normalized:
        return None
    if normalized in {"established_category", "fragmented_workflow", "mixed"}:
        return normalized  # type: ignore[return-value]
    return None


def collect_known_entities(
    *,
    brand: Any | None = None,
    seed_terms: list[str] | None = None,
) -> set[str]:
    """Collect likely tool/entity phrases used by workflow signal extraction."""
    entities: set[str] = set()

    def _add_term(term: str | None) -> None:
        if not term:
            return
        cleaned = re.sub(r"\s+", " ", term.strip().lower())
        if len(cleaned) < 2:
            return
        # Keep compact phrases only; long natural language snippets become noisy.
        if len(cleaned.split()) > 4:
            return
        entities.add(cleaned)

    if brand is not None:
        _add_term(getattr(brand, "company_name", None))
        for product in getattr(brand, "products_services", []) or []:
            _add_term(str(product.get("name") or ""))
            _add_term(str(product.get("category") or ""))
        for competitor in getattr(brand, "competitor_positioning", []) or []:
            _add_term(str(competitor.get("name") or competitor.get("brand") or ""))

    for term in seed_terms or []:
        _add_term(term)

    return entities


def extract_keyword_discovery_signals(
    keyword: str,
    *,
    known_entities: set[str] | None = None,
) -> dict[str, Any]:
    """Extract workflow/comparison signals used by downstream discovery steps."""
    keyword_clean = re.sub(r"\s+", " ", keyword.strip())
    keyword_lower = keyword_clean.lower()
    words = [w for w in re.split(r"[^a-z0-9]+", keyword_lower) if w]
    padded = f" {keyword_lower} "

    matched_entities: list[str] = []
    for entity in sorted(known_entities or set(), key=len, reverse=True):
        if not isinstance(entity, str):
            continue
        if entity and f" {entity} " in padded:
            matched_entities.append(entity)

    has_action_verb = any(re.search(rf"\b{re.escape(verb)}\b", keyword_lower) for verb in WORKFLOW_VERBS)
    integration_hits = sorted(
        term for term in INTEGRATION_TERMS if re.search(rf"\b{re.escape(term)}\b", keyword_lower)
    )
    has_integration_term = bool(integration_hits)

    comparison_hits = sorted(
        term for term in COMPARISON_TERMS if re.search(rf"\b{re.escape(term)}\b", keyword_lower)
    )
    is_comparison = bool(comparison_hits)
    has_pair_pattern = bool(
        re.search(r"\b[a-z0-9][a-z0-9.+_-]*\s+(to|vs|versus)\s+[a-z0-9][a-z0-9.+_-]*\b", keyword_lower)
    )
    has_two_entities = len(matched_entities) >= 2 or has_pair_pattern

    modifier_terms = sorted(
        term for term in MODIFIER_TERMS if re.search(rf"\b{re.escape(term)}\b", keyword_lower)
    )
    workflow_verb = next(
        (verb for verb in WORKFLOW_VERBS if re.search(rf"\b{re.escape(verb)}\b", keyword_lower)),
        None,
    )
    core_noun_phrase = next(
        (noun for noun in CORE_WORKFLOW_NOUNS if re.search(rf"\b{re.escape(noun)}\b", keyword_lower)),
        None,
    )

    comparison_target = matched_entities[0] if (is_comparison and matched_entities) else None
    if comparison_target is None and is_comparison:
        # Fall back to token after "vs"/"versus".
        vs_match = re.search(r"\b(?:vs|versus)\s+([a-z0-9.+_-]+)\b", keyword_lower)
        if vs_match:
            comparison_target = vs_match.group(1)

    return {
        "has_action_verb": has_action_verb,
        "has_integration_term": has_integration_term,
        "has_two_entities": has_two_entities,
        "is_comparison": is_comparison,
        "word_count": len(words),
        "modifier_terms": modifier_terms,
        "matched_entities": sorted(set(matched_entities)),
        "workflow_verb": workflow_verb,
        "integration_terms": integration_hits,
        "comparison_target": comparison_target,
        "core_noun_phrase": core_noun_phrase,
        "comparison_terms": comparison_hits,
    }


def diagnose_market_mode(
    *,
    source: str,
    override: str | None = None,
    seed_terms: list[str] | None = None,
    keyword_rows: list[dict[str, Any]] | None = None,
    serp_rows: list[dict[str, Any]] | None = None,
) -> MarketDiagnosisResult:
    """Diagnose market mode from early discovery evidence."""
    explicit_override = normalize_market_mode_override(override)
    if explicit_override is not None:
        return MarketDiagnosisResult(
            mode=explicit_override,
            signals={"override": explicit_override},
            reasons=[f"market_mode_override={explicit_override}"],
            source=source,
            confidence=1.0,
        )

    rows = keyword_rows or []
    seeds = seed_terms or []

    workflow_seed_ratio = _ratio(
        [
            _is_workflowish_text(seed)
            for seed in seeds
        ]
    )
    workflow_keyword_ratio = _ratio(
        [
            _row_flag(row, "has_action_verb") or _row_flag(row, "has_integration_term")
            for row in rows
        ]
    )
    two_entity_ratio = _ratio([_row_flag(row, "has_two_entities") for row in rows])
    comparison_ratio = _ratio([_row_flag(row, "is_comparison") for row in rows])

    volumes = [_extract_volume(row) for row in rows if _extract_volume(row) is not None]
    median_volume = float(median(volumes)) if volumes else None
    low_volume_ratio = _ratio(
        [
            (volume is not None and volume <= 20)
            for volume in (_extract_volume(row) for row in rows)
        ]
    )
    strong_head_term_ratio = _ratio(
        [
            (volume is not None and volume >= 200)
            for volume in (_extract_volume(row) for row in rows)
        ]
    )

    serp_vendor_density = _avg(
        [_extract_vendor_density(serp) for serp in (serp_rows or [])]
    )
    serp_ugc_docs_share = _avg(
        [_extract_ugc_docs_share(serp) for serp in (serp_rows or [])]
    )

    fragmented_checks = {
        "workflow_term_ratio": (
            max(workflow_seed_ratio, workflow_keyword_ratio) >= 0.30
        ),
        "two_entity_ratio": two_entity_ratio >= 0.25,
        "long_tail_volume": median_volume is not None and median_volume <= 10.0,
        "serp_unstructured": (
            serp_ugc_docs_share is not None
            and serp_vendor_density is not None
            and serp_ugc_docs_share >= 0.45
            and serp_vendor_density <= 0.45
        ),
        "weak_head_terms": strong_head_term_ratio <= 0.10 and low_volume_ratio >= 0.60,
    }

    established_checks = {
        "strong_head_terms": strong_head_term_ratio >= 0.25,
        "vendor_serp_dominance": serp_vendor_density is not None and serp_vendor_density >= 0.65,
        "volume_not_fragmented": median_volume is not None and median_volume >= 80.0,
    }

    fragmented_count = sum(1 for ok in fragmented_checks.values() if ok)
    established_count = sum(1 for ok in established_checks.values() if ok)

    if fragmented_count >= 2:
        mode: MarketMode = "fragmented_workflow"
        confidence = min(0.95, 0.55 + (0.08 * fragmented_count))
        reasons = [name for name, ok in fragmented_checks.items() if ok]
    elif established_checks["strong_head_terms"] and (
        established_checks["vendor_serp_dominance"] or established_checks["volume_not_fragmented"]
    ):
        mode = "established_category"
        confidence = min(0.95, 0.55 + (0.08 * established_count))
        reasons = [name for name, ok in established_checks.items() if ok]
    else:
        mode = "mixed"
        confidence = 0.5
        reasons = ["insufficient_consensus_for_strong_mode"]

    signals = {
        "workflow_seed_ratio": round(workflow_seed_ratio, 4),
        "workflow_keyword_ratio": round(workflow_keyword_ratio, 4),
        "two_entity_ratio": round(two_entity_ratio, 4),
        "comparison_ratio": round(comparison_ratio, 4),
        "median_volume": median_volume,
        "low_volume_ratio": round(low_volume_ratio, 4),
        "strong_head_term_ratio": round(strong_head_term_ratio, 4),
        "serp_vendor_density": serp_vendor_density,
        "serp_ugc_docs_share": serp_ugc_docs_share,
        "fragmented_checks": fragmented_checks,
        "established_checks": established_checks,
        "keyword_count": len(rows),
        "seed_count": len(seeds),
    }

    return MarketDiagnosisResult(
        mode=mode,
        signals=signals,
        reasons=reasons,
        source=source,
        confidence=confidence,
    )


def _is_workflowish_text(value: str) -> bool:
    signals = extract_keyword_discovery_signals(value)
    return bool(signals["has_action_verb"] or signals["has_integration_term"] or signals["has_two_entities"])


def _row_flag(row: dict[str, Any], flag: str) -> bool:
    signals = row.get("discovery_signals")
    if isinstance(signals, dict) and flag in signals:
        return bool(signals.get(flag))
    return bool(row.get(flag))


def _extract_volume(row: dict[str, Any]) -> int | None:
    volume = row.get("search_volume")
    if volume is None:
        volume = row.get("volume")
    if volume is None:
        return None
    try:
        return int(volume)
    except (TypeError, ValueError):
        return None


def _extract_vendor_density(serp: dict[str, Any]) -> float | None:
    results = serp.get("organic_results")
    if not isinstance(results, list) or not results:
        return None
    vendor = 0
    total = 0
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        total += 1
        url = str(item.get("url") or "").lower()
        domain = str(item.get("domain") or "").lower()
        title = str(item.get("title") or "").lower()
        if _is_vendorish_result(url=url, domain=domain, title=title):
            vendor += 1
    if total == 0:
        return None
    return round(vendor / total, 4)


def _extract_ugc_docs_share(serp: dict[str, Any]) -> float | None:
    results = serp.get("organic_results")
    if not isinstance(results, list) or not results:
        return None
    ugc_docs = 0
    total = 0
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        total += 1
        url = str(item.get("url") or "").lower()
        domain = str(item.get("domain") or "").lower()
        if _is_ugc_or_docs(url=url, domain=domain):
            ugc_docs += 1
    if total == 0:
        return None
    return round(ugc_docs / total, 4)


def _is_vendorish_result(*, url: str, domain: str, title: str) -> bool:
    if _is_ugc_or_docs(url=url, domain=domain):
        return False
    vendor_hints = (
        "/pricing",
        "/product",
        "/features",
        "/platform",
        "/software",
        "/solutions",
        "/integrations",
        "free trial",
        "book demo",
    )
    return any(hint in url or hint in title for hint in vendor_hints)


def _is_ugc_or_docs(*, url: str, domain: str) -> bool:
    if any(hint in domain for hint in UGC_DOC_DOMAINS):
        return True
    if "docs." in domain or "/docs" in url:
        return True
    if "forum" in domain or "/forum" in url:
        return True
    return False


def _ratio(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _avg(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 4)
