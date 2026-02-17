"""Model-name normalization and alias matching helpers."""

from __future__ import annotations

import re

_ALIAS_MAP: dict[str, str] = {
    "claude sonnet 4.5": "anthropic claude sonnet 4 5",
    "claude 3.7 sonnet": "anthropic claude 3 7 sonnet",
    "gemini 2.5 pro": "google gemini 2 5 pro",
    "gemini 2.5 flash": "google gemini 2 5 flash",
    "gpt 4o": "openai gpt 4o",
    "gpt 4.1": "openai gpt 4 1",
    "deepseek r1": "deepseek deepseek r1",
    "llama 3.3 70b": "meta llama 3 3 70b",
}


def normalize_model_name(name: str) -> str:
    """Normalize model/provider naming for robust matching across sources."""
    normalized = name.strip().lower()
    normalized = normalized.replace("openrouter:", "")
    normalized = normalized.replace(":free", " free")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def resolve_alias(name: str) -> str:
    """Resolve known aliases to canonical normalized names."""
    normalized = normalize_model_name(name)
    return _ALIAS_MAP.get(normalized, normalized)


def build_candidate_lookup(models: list[str]) -> dict[str, str]:
    """Build lookup from normalized keys to canonical model IDs."""
    lookup: dict[str, str] = {}

    for model in models:
        key = resolve_alias(model)
        lookup.setdefault(key, model)

        normalized = normalize_model_name(model)
        lookup.setdefault(normalized, model)

        # Also index by the model-id suffix (drops provider prefixes when possible).
        if "/" in model:
            suffix = model.split("/", 1)[1]
            lookup.setdefault(resolve_alias(suffix), model)

    return lookup


def resolve_arena_model_to_candidate(
    arena_model_name: str,
    candidate_models: list[str],
) -> str | None:
    """Map Arena model display names to OpenRouter candidate model IDs."""
    lookup = build_candidate_lookup(candidate_models)

    normalized = resolve_alias(arena_model_name)
    if normalized in lookup:
        return lookup[normalized]

    # Last attempt: partial token-overlap match.
    arena_tokens = set(normalized.split())
    best_match: str | None = None
    best_score = 0.0

    for key, model in lookup.items():
        key_tokens = set(key.split())
        if not key_tokens:
            continue
        overlap = len(arena_tokens & key_tokens) / len(arena_tokens | key_tokens)
        if overlap > best_score:
            best_score = overlap
            best_match = model

    if best_score >= 0.7:
        return best_match
    return None
