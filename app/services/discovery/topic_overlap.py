"""Shared overlap and comparison-entity utilities for topic diversification."""

from __future__ import annotations

import re
from collections import Counter

COMPARISON_MODIFIER_STOPWORDS = {
    "vs",
    "versus",
    "alternative",
    "alternatives",
    "comparison",
    "compare",
    "pricing",
    "review",
    "best",
    "top",
}

CONNECTOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "of",
    "or",
    "the",
    "to",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+._-]*")
PAIR_PATTERN = re.compile(
    r"\b([a-z0-9][a-z0-9+._-]{1,40})\s+(?:vs|versus)\s+([a-z0-9][a-z0-9+._-]{1,40})\b"
)
TRAILING_ALTERNATIVES_PATTERN = re.compile(
    r"\b([a-z0-9][a-z0-9+._-]{1,40})\s+(?:alternative|alternatives)\b"
)
LEADING_ALTERNATIVES_PATTERN = re.compile(
    r"\b(?:alternative|alternatives)\s+(?:to\s+)?([a-z0-9][a-z0-9+._-]{1,40})\b"
)


def normalize_text_tokens(text: str) -> set[str]:
    """Normalize free text into set tokens for overlap calculations."""
    tokens = set()
    for raw in TOKEN_PATTERN.findall((text or "").lower()):
        token = _normalize_entity_token(raw)
        if token is None:
            continue
        if token in COMPARISON_MODIFIER_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def extract_comparison_entities(text: str) -> list[str]:
    """Extract likely comparison entities from topic/keyword text."""
    lowered = (text or "").lower()
    entities: list[str] = []

    for match in PAIR_PATTERN.finditer(lowered):
        _append_entity(entities, match.group(1))
        _append_entity(entities, match.group(2))

    for match in TRAILING_ALTERNATIVES_PATTERN.finditer(lowered):
        _append_entity(entities, match.group(1))

    for match in LEADING_ALTERNATIVES_PATTERN.finditer(lowered):
        _append_entity(entities, match.group(1))

    return entities


def build_comparison_key(topic_name: str, primary_keyword: str) -> str | None:
    """Build canonical comparison key for pair/anchor-style topics."""
    entities = extract_comparison_entities(f"{topic_name} {primary_keyword}")
    if len(entities) >= 2:
        left, right = sorted(entities[:2])
        return f"pair:{left}|{right}"
    if len(entities) == 1:
        return f"anchor:{entities[0]}"
    return None


def build_family_key(topic_name: str, primary_keyword: str, keyword_texts: list[str]) -> str:
    """Build coarse family key used for diversity caps."""
    comparison_key = build_comparison_key(topic_name, primary_keyword)
    if comparison_key:
        if comparison_key.startswith("pair:"):
            entities = extract_comparison_entities(f"{topic_name} {primary_keyword}")
            anchor = entities[0] if entities else comparison_key.split(":", 1)[1].split("|", 1)[0]
            return f"family:cmp:{anchor}"
        anchor = comparison_key.split(":", 1)[1]
        return f"family:cmp:{anchor}"

    counts: Counter[str] = Counter()
    for text in [topic_name, primary_keyword, *keyword_texts[:30]]:
        for token in normalize_text_tokens(text):
            counts[token] += 1

    if not counts:
        return "family:uncategorized"

    top_terms = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    return f"family:{'|'.join(term for term, _ in top_terms)}"


def jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity for two token sets."""
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def compute_topic_overlap(
    *,
    keyword_tokens_a: set[str],
    keyword_tokens_b: set[str],
    text_tokens_a: set[str],
    text_tokens_b: set[str],
    serp_domains_a: set[str] | None = None,
    serp_domains_b: set[str] | None = None,
    intent_a: str | None = None,
    intent_b: str | None = None,
    page_type_a: str | None = None,
    page_type_b: str | None = None,
) -> float:
    """Compute weighted overlap score between two topic signatures."""
    keyword_score = jaccard(keyword_tokens_a, keyword_tokens_b)
    text_score = jaccard(text_tokens_a, text_tokens_b)
    serp_score = jaccard(serp_domains_a or set(), serp_domains_b or set())

    intent_match = bool(intent_a and intent_b and intent_a == intent_b)
    page_type_match = bool(page_type_a and page_type_b and page_type_a == page_type_b)
    if intent_match and page_type_match:
        intent_page_bonus = 1.0
    elif intent_match or page_type_match:
        intent_page_bonus = 0.5
    else:
        intent_page_bonus = 0.0

    score = (
        (0.45 * keyword_score)
        + (0.30 * text_score)
        + (0.20 * serp_score)
        + (0.05 * intent_page_bonus)
    )
    return max(0.0, min(1.0, score))


def is_exact_pair_duplicate(key_a: str | None, key_b: str | None) -> bool:
    """Return True when keys represent the exact same comparison pair."""
    pair_a = _pair_entities(key_a)
    pair_b = _pair_entities(key_b)
    if pair_a is None or pair_b is None:
        return False
    return pair_a == pair_b


def is_sibling_pair(key_a: str | None, key_b: str | None) -> bool:
    """Return True when both keys are pair comparisons sharing one entity."""
    pair_a = _pair_entities(key_a)
    pair_b = _pair_entities(key_b)
    if pair_a is None or pair_b is None:
        return False
    shared = set(pair_a) & set(pair_b)
    return len(shared) == 1 and pair_a != pair_b


def _normalize_entity_token(token: str) -> str | None:
    cleaned = token.strip().lower().strip("._-")
    if len(cleaned) < 2:
        return None
    if cleaned in COMPARISON_MODIFIER_STOPWORDS:
        return None
    if cleaned in CONNECTOR_STOPWORDS:
        return None
    return cleaned


def _append_entity(entities: list[str], raw: str) -> None:
    token = _normalize_entity_token(raw)
    if token is None:
        return
    if token not in entities:
        entities.append(token)


def _pair_entities(key: str | None) -> tuple[str, str] | None:
    if not key or not key.startswith("pair:"):
        return None
    body = key.split(":", 1)[1]
    parts = [part.strip() for part in body.split("|") if part.strip()]
    if len(parts) != 2:
        return None
    ordered = sorted(parts)
    return ordered[0], ordered[1]
