"""Scoring and selection logic for model selector."""

from __future__ import annotations

from typing import Any

from app.services.model_selector.model_aliases import resolve_arena_model_to_candidate
from app.services.model_selector.types import AgentSelection, ModelCandidate


def map_arena_scores_to_candidates(
    raw_arena_scores: dict[str, float],
    candidate_models: list[str],
) -> tuple[dict[str, float], int]:
    """Map Arena display names to candidate model IDs and normalize scores."""
    mapped_raw: dict[str, float] = {}
    unmatched = 0

    for arena_name, raw_score in raw_arena_scores.items():
        candidate_model = resolve_arena_model_to_candidate(arena_name, candidate_models)
        if candidate_model is None:
            unmatched += 1
            continue
        existing = mapped_raw.get(candidate_model)
        if existing is None or raw_score > existing:
            mapped_raw[candidate_model] = raw_score

    if not mapped_raw:
        return {}, unmatched

    values = list(mapped_raw.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return ({model: 1.0 for model in mapped_raw}, unmatched)

    normalized: dict[str, float] = {}
    for model, raw_score in mapped_raw.items():
        normalized[model] = (raw_score - min_score) / (max_score - min_score)

    return normalized, unmatched


def select_best_model(
    *,
    agent_class: str,
    candidates: list[ModelCandidate],
    raw_arena_scores: dict[str, float],
    openrouter_weight: float,
    arena_weight: float,
    max_price: float,
    fallback_model: str,
    openrouter_meta: dict[str, Any],
    arena_meta: dict[str, Any],
) -> AgentSelection:
    """Select the best candidate using weighted ranking and policy rules."""
    filtered = [candidate for candidate in candidates if candidate.price_per_million <= max_price]
    if not filtered:
        return AgentSelection(
            agent_class=agent_class,
            model=fallback_model,
            max_price=max_price,
            score_breakdown={
                "openrouter_rank_score": 0.0,
                "arena_score": 0.0,
                "final_score": 0.0,
            },
            source_metadata={
                "fallback_reason": "no_candidates_under_price_cap",
                "openrouter": openrouter_meta,
                "arena": arena_meta,
            },
            fallback_used=True,
        )

    weight_total = openrouter_weight + arena_weight
    if weight_total <= 0:
        openrouter_weight = 1.0
        arena_weight = 0.0
    else:
        openrouter_weight /= weight_total
        arena_weight /= weight_total

    candidate_ids = [candidate.model for candidate in filtered]
    arena_scores, arena_unmatched = map_arena_scores_to_candidates(raw_arena_scores, candidate_ids)

    for candidate in filtered:
        arena_score = arena_scores.get(candidate.model, 0.0)
        final_score = (openrouter_weight * candidate.openrouter_rank_score) + (
            arena_weight * arena_score
        )

        candidate.arena_score = arena_score
        candidate.final_score = final_score

    filtered.sort(
        key=lambda candidate: (
            -candidate.final_score,
            candidate.price_per_million,
            candidate.openrouter_rank,
            -(candidate.openrouter_popularity or 0.0),
        )
    )

    best = filtered[0]
    return AgentSelection(
        agent_class=agent_class,
        model=best.model,
        max_price=max_price,
        score_breakdown={
            "openrouter_rank_score": round(best.openrouter_rank_score, 6),
            "arena_score": round(best.arena_score, 6),
            "final_score": round(best.final_score, 6),
        },
        source_metadata={
            "selected_candidate": {
                "model": best.model,
                "raw_model_id": best.raw_model_id,
                "price_per_million": best.price_per_million,
                "openrouter_rank": best.openrouter_rank,
            },
            "arena_matches": len(arena_scores),
            "arena_unmatched": arena_unmatched,
            "openrouter": openrouter_meta,
            "arena": arena_meta,
        },
        fallback_used=False,
    )
