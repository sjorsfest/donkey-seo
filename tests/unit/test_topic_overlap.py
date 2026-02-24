"""Unit tests for shared topic overlap utilities."""

from app.services.discovery.topic_overlap import (
    build_comparison_key,
    compute_topic_overlap,
    extract_comparison_entities,
    is_exact_pair_duplicate,
    is_sibling_pair,
    normalize_text_tokens,
)


def test_normalize_text_tokens_drops_comparison_modifiers() -> None:
    tokens = normalize_text_tokens("Zendesk vs Intercom alternatives pricing review")
    assert "zendesk" in tokens
    assert "intercom" in tokens
    assert "vs" not in tokens
    assert "alternatives" not in tokens
    assert "pricing" not in tokens
    assert "review" not in tokens


def test_build_comparison_key_normalizes_pair_order() -> None:
    key_a = build_comparison_key("Intercom vs Zendesk", "")
    key_b = build_comparison_key("Zendesk vs Intercom", "")

    assert key_a == "pair:intercom|zendesk"
    assert key_b == "pair:intercom|zendesk"
    assert is_exact_pair_duplicate(key_a, key_b)


def test_sibling_pair_detection_for_shared_anchor() -> None:
    left = "pair:intercom|zendesk"
    right = "pair:tidio|zendesk"

    assert is_sibling_pair(left, right)
    assert not is_exact_pair_duplicate(left, right)


def test_extract_comparison_entities_from_alternatives() -> None:
    entities = extract_comparison_entities("alternatives to zendesk")
    assert entities == ["zendesk"]


def test_compute_topic_overlap_uses_weighted_components() -> None:
    overlap = compute_topic_overlap(
        keyword_tokens_a={"zendesk", "intercom"},
        keyword_tokens_b={"zendesk", "tidio"},
        text_tokens_a={"helpdesk", "comparison"},
        text_tokens_b={"helpdesk", "guide"},
        serp_domains_a={"g2.com", "capterra.com"},
        serp_domains_b={"g2.com"},
        intent_a="commercial",
        intent_b="commercial",
        page_type_a="comparison",
        page_type_b="comparison",
    )

    assert overlap > 0.39
    assert overlap < 0.8
