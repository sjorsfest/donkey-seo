"""Unit tests for discovery learning synthesis and prompt context plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.discovery_learning import (
    DiscoveryLearningService,
    LearningCandidate,
    STATUS_REGRESSED,
)
from app.services.steps.base_step import BaseStepService


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _RowsResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarRows:
        return _ScalarRows(self._rows)


class _SessionRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    async def execute(self, _query: object) -> _RowsResult:
        return _RowsResult(self._rows)


def test_compute_usable_keywords_prefers_iteration_acceptance_then_fallback() -> None:
    service = DiscoveryLearningService.__new__(DiscoveryLearningService)

    topics_by_id = {
        "topic-1": SimpleNamespace(priority_factors={"fit_tier": "primary"}, priority_rank=1),
        "topic-2": SimpleNamespace(priority_factors={"fit_tier": "secondary"}, priority_rank=2),
    }
    keywords = [
        SimpleNamespace(id="kw-1", topic_id="topic-1"),
        SimpleNamespace(id="kw-2", topic_id="topic-2"),
        SimpleNamespace(id="kw-3", topic_id=None),
    ]

    by_acceptance = service._compute_usable_keywords(  # type: ignore[attr-defined]
        keywords=keywords,
        topics_by_id=topics_by_id,
        accepted_topic_ids={"topic-1"},
    )
    assert by_acceptance["kw-1"] is True
    assert by_acceptance["kw-2"] is False
    assert by_acceptance["kw-3"] is False

    by_fallback = service._compute_usable_keywords(  # type: ignore[attr-defined]
        keywords=keywords,
        topics_by_id=topics_by_id,
        accepted_topic_ids=set(),
    )
    assert by_fallback["kw-1"] is True
    assert by_fallback["kw-2"] is True
    assert by_fallback["kw-3"] is False


def test_build_deterministic_candidates_captures_archetype_and_bottleneck() -> None:
    service = DiscoveryLearningService.__new__(DiscoveryLearningService)
    seed_topics = {"seed-1": SimpleNamespace(id="seed-1", pillar_type="core_offer")}
    topics_by_id = {
        "topic-a": SimpleNamespace(
            id="topic-a",
            priority_factors={"fit_tier": "primary"},
            priority_rank=1,
            recommended_url_type="landing",
            dominant_intent="commercial",
        ),
        "topic-b": SimpleNamespace(
            id="topic-b",
            priority_factors={"fit_tier": "secondary"},
            priority_rank=2,
            recommended_url_type="blog",
            dominant_intent="informational",
        ),
    }

    keywords = []
    for idx in range(8):
        keywords.append(
            SimpleNamespace(
                id=f"cmp-{idx}",
                topic_id="topic-a",
                seed_topic_id="seed-1",
                discovery_signals={"is_comparison": True},
            )
        )
    for idx in range(8):
        keywords.append(
            SimpleNamespace(
                id=f"gen-{idx}",
                topic_id="topic-b",
                seed_topic_id="seed-1",
                discovery_signals={},
            )
        )

    usable_map = {f"cmp-{idx}": True for idx in range(8)}
    usable_map.update({f"gen-{idx}": False for idx in range(8)})
    decisions = [
        SimpleNamespace(
            source_topic_id="topic-a",
            decision="accepted",
            rejection_reasons=[],
        ),
        SimpleNamespace(
            source_topic_id="topic-b",
            decision="rejected",
            rejection_reasons=["missing_serp_results"],
        ),
    ]

    candidates = service._build_deterministic_candidates(  # type: ignore[attr-defined]
        decisions=decisions,
        step_summaries={2: {"known_gaps_count": 4, "seeds_created": 16}},
        keywords=keywords,
        seed_topics=seed_topics,
        topics_by_id=topics_by_id,
        usable_by_keyword=usable_map,
    )
    keys = {item.learning_key for item in candidates}
    assert "overall:usable_keyword_rate" in keys
    assert "archetype:comparison:usable_rate" in keys
    assert "bottleneck:missing_serp_results" in keys


@pytest.mark.asyncio
async def test_apply_history_marks_regressed_on_negative_delta_for_positive_signal() -> None:
    previous = SimpleNamespace(
        pipeline_run_id="run-old",
        iteration_index=1,
        learning_key="overall:usable_keyword_rate",
        current_metric=0.8,
        created_at=None,
    )
    service = DiscoveryLearningService(_SessionRows([previous]))
    candidates = [
        LearningCandidate(
            learning_key="overall:usable_keyword_rate",
            source_capability="prioritization",
            source_agent=None,
            learning_type="strategy_effect",
            polarity="positive",
            title="title",
            detail="detail",
            recommendation=None,
            confidence=0.8,
            current_metric=0.6,
            applies_to_capabilities=["prioritization"],
            applies_to_agents=None,
            evidence={},
        )
    ]

    await service._apply_history(  # type: ignore[attr-defined]
        project_id="project-1",
        pipeline_run_id="run-new",
        iteration_index=2,
        candidates=candidates,
    )
    assert candidates[0].status == STATUS_REGRESSED
    assert candidates[0].delta_metric == pytest.approx(-0.2, rel=1e-4)


@pytest.mark.asyncio
async def test_rewrite_candidates_falls_back_when_summarizer_fails(monkeypatch) -> None:
    class _FailingAgent:
        async def run(self, _input: object) -> object:
            raise RuntimeError("boom")

    service = DiscoveryLearningService.__new__(DiscoveryLearningService)
    candidates = [
        LearningCandidate(
            learning_key="overall:usable_keyword_rate",
            source_capability="prioritization",
            source_agent=None,
            learning_type="strategy_effect",
            polarity="positive",
            title="Original title",
            detail="Original detail",
            recommendation="Original recommendation",
            confidence=0.5,
            current_metric=0.5,
            applies_to_capabilities=["prioritization"],
            applies_to_agents=None,
            evidence={},
        )
    ]

    monkeypatch.setattr(
        "app.services.discovery_learning.DiscoveryLearningSummarizerAgent",
        lambda: _FailingAgent(),
    )

    await service._rewrite_candidates(project_id="project-1", candidates=candidates)  # type: ignore[attr-defined]
    assert candidates[0].title == "Original title"
    assert candidates[0].detail == "Original detail"


class _DummyStep(BaseStepService[dict, dict]):
    step_number = 99
    step_name = "dummy"

    async def _execute(self, input_data: dict) -> dict:
        return input_data

    async def _validate_preconditions(self, input_data: dict) -> None:
        return None

    async def _persist_results(self, result: dict) -> None:
        return None


@pytest.mark.asyncio
async def test_build_learning_context_reads_capabilities_and_legacy_steps() -> None:
    rows = [
        SimpleNamespace(
            learning_key="alpha",
            title="Prioritization insight",
            detail="Use comparison intent more often.",
            recommendation="Increase comparison weighting.",
            status="new",
            polarity="positive",
            novelty_score=0.8,
            confidence=0.7,
            applies_to_capabilities=["prioritization"],
            applies_to_agents=["PrioritizationAgent"],
            evidence={},
        ),
        SimpleNamespace(
            learning_key="beta",
            title="Legacy compatibility insight",
            detail="Legacy step-based applicability still maps correctly.",
            recommendation=None,
            status="regressed",
            polarity="negative",
            novelty_score=0.6,
            confidence=0.8,
            applies_to_capabilities=[],
            applies_to_agents=[],
            evidence={"applies_to_steps": [7]},
        ),
    ]

    service = _DummyStep(
        session=_SessionRows(rows),  # type: ignore[arg-type]
        project_id="project-1",
        execution=SimpleNamespace(
            checkpoint_data=None,
            pipeline_run_id=None,
        ),
    )
    context = await service.build_learning_context(
        "prioritization",
        "PrioritizationAgent",
    )

    assert "Prioritization insight" in context
    assert "Legacy compatibility insight" in context


@pytest.mark.asyncio
async def test_build_learning_context_respects_char_budget() -> None:
    rows = [
        SimpleNamespace(
            learning_key="a",
            title="A",
            detail="x" * 200,
            recommendation=None,
            status="new",
            polarity="positive",
            novelty_score=1.0,
            confidence=1.0,
            applies_to_capabilities=["prioritization"],
            applies_to_agents=[],
            evidence={},
        ),
        SimpleNamespace(
            learning_key="b",
            title="B",
            detail="y" * 200,
            recommendation=None,
            status="new",
            polarity="positive",
            novelty_score=1.0,
            confidence=1.0,
            applies_to_capabilities=["prioritization"],
            applies_to_agents=[],
            evidence={},
        ),
    ]
    service = _DummyStep(
        session=_SessionRows(rows),  # type: ignore[arg-type]
        project_id="project-1",
        execution=SimpleNamespace(
            checkpoint_data=None,
            pipeline_run_id=None,
        ),
    )
    context = await service.build_learning_context(
        "prioritization",
        None,
        max_items=8,
        max_chars=260,
    )
    assert "A" in context
    assert "B" not in context
