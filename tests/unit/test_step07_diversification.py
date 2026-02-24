"""Unit tests for Step 7 diversification and overlap suppression."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.steps.discovery.step_07_prioritization import Step07PrioritizationService


def _topic(
    topic_id: str,
    name: str,
    *,
    intent: str = "commercial",
    page_type: str = "comparison",
    score: float = 80.0,
    fit_tier: str = "primary",
    keyword: str | None = None,
) -> dict:
    topic = SimpleNamespace(
        id=topic_id,
        name=name,
        description="",
        cluster_notes="",
        dominant_intent=intent,
        dominant_page_type=page_type,
        primary_keyword_id=f"kw-{topic_id}",
        serp_evidence_keyword_id=None,
    )
    kw = SimpleNamespace(
        id=f"kw-{topic_id}",
        keyword=keyword or name,
        serp_top_results=[],
    )
    return {
        "topic_id": topic_id,
        "topic": topic,
        "keywords": [kw],
        "priority_score": score,
        "deterministic_priority_score": score,
        "fit_assessment": {
            "fit_tier": fit_tier,
            "fit_threshold_used": 0.5,
            "hard_exclusion_reason": None,
            "final_cut_reason_code": "llm_primary",
        },
    }


def test_diversification_suppresses_exact_pair_duplicates() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    first = _topic("t1", "Zendesk vs Intercom", score=92.0)
    second = _topic("t2", "Intercom vs Zendesk", score=88.0)
    scored_topics = [first, second]

    summary = service._apply_diversification(  # type: ignore[attr-defined]
        scored_topics=scored_topics,
        primary_threshold=0.6,
        secondary_threshold=0.5,
    )

    assert summary["exact_pair_duplicates_removed"] == 1
    assert second["fit_assessment"]["fit_tier"] == "excluded"
    assert second["fit_assessment"]["final_cut_reason_code"] == "exact_pair_duplicate"
    assert second["diversification"]["action"] == "exclude"


def test_diversification_keeps_sibling_pairs() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    first = _topic("t1", "Zendesk vs Intercom", score=91.0)
    second = _topic("t2", "Zendesk vs Tidio", score=87.0)
    scored_topics = [first, second]

    summary = service._apply_diversification(  # type: ignore[attr-defined]
        scored_topics=scored_topics,
        primary_threshold=0.6,
        secondary_threshold=0.5,
    )

    assert summary["sibling_pairs_kept"] >= 1
    assert first["fit_assessment"]["fit_tier"] == "primary"
    assert second["fit_assessment"]["fit_tier"] == "primary"
    assert second["diversification"]["relationship_type"] == "sibling_pair"


def test_diversification_applies_hard_and_soft_overlap_thresholds(monkeypatch) -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    base = _topic("t1", "Helpdesk software guide", intent="informational", page_type="guide", score=95.0, keyword="base")
    hard = _topic("t2", "Help desk software", intent="informational", page_type="guide", score=90.0, keyword="hard")
    soft = _topic("t3", "Customer support workflow", intent="informational", page_type="guide", score=85.0, keyword="soft")
    scored_topics = [base, hard, soft]

    def _fake_overlap(**kwargs) -> float:  # type: ignore[no-untyped-def]
        tokens = kwargs["keyword_tokens_a"] | kwargs["keyword_tokens_b"]
        if "hard" in tokens:
            return 0.90
        if "soft" in tokens:
            return 0.75
        return 0.10

    monkeypatch.setattr(
        "app.services.steps.discovery.step_07_prioritization.compute_topic_overlap",
        _fake_overlap,
    )

    summary = service._apply_diversification(  # type: ignore[attr-defined]
        scored_topics=scored_topics,
        primary_threshold=0.6,
        secondary_threshold=0.5,
    )

    assert summary["near_duplicate_excluded"] == 1
    assert summary["overlap_demoted"] == 1
    assert hard["fit_assessment"]["fit_tier"] == "excluded"
    assert hard["fit_assessment"]["final_cut_reason_code"] == "near_duplicate_hard"
    assert soft["fit_assessment"]["fit_tier"] == "secondary"
    assert soft["fit_assessment"]["final_cut_reason_code"] == "overlap_demoted"


def test_diversification_primary_cap_demotes_overflow() -> None:
    service = Step07PrioritizationService.__new__(Step07PrioritizationService)
    first = _topic("t1", "Helpdesk pricing software", intent="commercial", page_type="landing", score=90.0, keyword="alpha")
    second = _topic("t2", "Support platform pricing", intent="commercial", page_type="landing", score=85.0, keyword="beta")
    third = _topic("t3", "Customer service tool pricing", intent="commercial", page_type="landing", score=80.0, keyword="gamma")
    scored_topics = [first, second, third]

    summary = service._apply_diversification(  # type: ignore[attr-defined]
        scored_topics=scored_topics,
        primary_threshold=0.6,
        secondary_threshold=0.5,
    )

    assert summary["diversity_cap_demotions"] == 1
    assert first["fit_assessment"]["fit_tier"] == "primary"
    assert second["fit_assessment"]["fit_tier"] == "primary"
    assert third["fit_assessment"]["fit_tier"] == "secondary"
    assert third["fit_assessment"]["final_cut_reason_code"] == "diversity_cap_demotion"
