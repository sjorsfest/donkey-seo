"""Unit tests for discovery loop strategy and filtering helpers."""

from types import SimpleNamespace

import pytest

from app.schemas.pipeline import DiscoveryLoopConfig
from app.services.pipelines.discovery.loop import DiscoveryLoopSupervisor, TopicDecision


def test_iteration_strategy_progression_and_excludes() -> None:
    service = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)

    strategy = service._build_iteration_strategy_payload(  # type: ignore[attr-defined]
        base_strategy_payload={"exclude_topics": ["adult content"]},
        iteration=3,
        dynamic_excludes=["legacy systems", "Adult Content"],
    )

    assert strategy["scope_mode"] == "broad_education"
    assert strategy["fit_threshold_profile"] == "lenient"
    assert strategy["exclude_topics"] == ["adult content", "legacy systems"]


def test_next_dynamic_excludes_adds_only_hard_or_low_icp_and_keeps_immutable() -> None:
    service = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)
    decisions = [
        TopicDecision(
            source_topic_id="t-1",
            topic_name="Keep Existing",
            fit_tier="secondary",
            fit_score=0.5,
            keyword_difficulty=20.0,
            domain_diversity=0.7,
            validated_intent="informational",
            validated_page_type="guide",
            decision="rejected",
            rejection_reasons=[],
            is_hard_excluded=True,
            is_very_low_icp=False,
        ),
        TopicDecision(
            source_topic_id="t-2",
            topic_name="Low ICP Topic",
            fit_tier="secondary",
            fit_score=0.5,
            keyword_difficulty=20.0,
            domain_diversity=0.7,
            validated_intent="informational",
            validated_page_type="guide",
            decision="rejected",
            rejection_reasons=[],
            is_hard_excluded=False,
            is_very_low_icp=True,
        ),
    ]

    excludes = service._next_dynamic_excludes(  # type: ignore[attr-defined]
        current_dynamic_excludes=["existing topic"],
        decisions=decisions,
        immutable_excludes={"keep existing"},
    )

    assert "existing topic" in excludes
    assert "Keep Existing" not in excludes
    assert "Low ICP Topic" in excludes


def test_extract_top_domains_dedupes_and_normalizes() -> None:
    service = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)

    domains = service._extract_top_domains(  # type: ignore[attr-defined]
        [
            {"domain": "Example.com"},
            {"domain": "example.com"},
            {"domain": "another.com"},
            {"domain": ""},
        ]
    )

    assert domains == ["example.com", "another.com"]


def test_collect_iteration_step_summaries_keeps_dict_only() -> None:
    service = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)

    summaries = service._collect_iteration_step_summaries(  # type: ignore[attr-defined]
        None,
        step_num=2,
        execution=SimpleNamespace(result_summary={"seeds_created": 12}),
    )
    summaries = service._collect_iteration_step_summaries(  # type: ignore[attr-defined]
        summaries,
        step_num=3,
        execution=SimpleNamespace(result_summary="ignored"),
    )

    assert summaries[2] == {"seeds_created": 12}
    assert summaries[3] == {}


def test_merge_accepted_topics_accumulates_across_iterations() -> None:
    service = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)

    iteration_1 = [
        TopicDecision(
            source_topic_id="t-1",
            topic_name="IT Service Catalog",
            fit_tier="secondary",
            fit_score=0.3,
            keyword_difficulty=10.0,
            domain_diversity=0.8,
            validated_intent="informational",
            validated_page_type="guide",
            decision="accepted",
            rejection_reasons=[],
        )
    ]
    pool = service._merge_accepted_topics(current_pool={}, decisions=iteration_1)  # type: ignore[attr-defined]
    assert len(pool) == 1

    iteration_2 = [
        TopicDecision(
            source_topic_id="t-2",
            topic_name="IT Service Management",
            fit_tier="primary",
            fit_score=0.5,
            keyword_difficulty=15.0,
            domain_diversity=0.9,
            validated_intent="commercial",
            validated_page_type="landing",
            decision="accepted",
            rejection_reasons=[],
        )
    ]
    pool = service._merge_accepted_topics(current_pool=pool, decisions=iteration_2)  # type: ignore[attr-defined]

    assert len(pool) == 2
    selected_ids = service._collect_selected_topic_ids(pool)  # type: ignore[attr-defined]
    assert selected_ids == ["t-1", "t-2"]


def test_merge_accepted_topics_updates_same_topic_to_latest_id() -> None:
    service = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)

    pool = service._merge_accepted_topics(  # type: ignore[attr-defined]
        current_pool={},
        decisions=[
            TopicDecision(
                source_topic_id="old-id",
                topic_name="Support Pricing",
                fit_tier="secondary",
                fit_score=0.4,
                keyword_difficulty=12.0,
                domain_diversity=0.7,
                validated_intent="commercial",
                validated_page_type="landing",
                decision="accepted",
                rejection_reasons=[],
            )
        ],
    )
    pool = service._merge_accepted_topics(  # type: ignore[attr-defined]
        current_pool=pool,
        decisions=[
            TopicDecision(
                source_topic_id="new-id",
                topic_name="  support pricing  ",
                fit_tier="secondary",
                fit_score=0.45,
                keyword_difficulty=11.0,
                domain_diversity=0.8,
                validated_intent="commercial",
                validated_page_type="landing",
                decision="accepted",
                rejection_reasons=[],
            )
        ],
    )

    selected_ids = service._collect_selected_topic_ids(pool)  # type: ignore[attr-defined]
    selected_names = service._collect_selected_topic_names(pool)  # type: ignore[attr-defined]

    assert selected_ids == ["new-id"]
    assert selected_names == ["Support Pricing"]


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class _RowsResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarRows:
        return _ScalarRows(self._rows)


class _SessionSequence:
    def __init__(self, topic_rows: list[object], keyword_rows: list[object]) -> None:
        self._topic_rows = topic_rows
        self._keyword_rows = keyword_rows
        self._calls = 0

    async def execute(self, _query: object) -> _RowsResult:
        self._calls += 1
        if self._calls == 1:
            return _RowsResult(self._topic_rows)
        return _RowsResult(self._keyword_rows)


@pytest.mark.asyncio
async def test_evaluate_decisions_accepts_workflow_topic_with_alternate_serp() -> None:
    topic = SimpleNamespace(
        id="topic-1",
        name="Slack to Notion Workflow",
        fit_tier="primary",
        fit_score=0.81,
        prioritization_diagnostics={"fit_reasons": []},
        serp_intent_confidence=0.8,
        serp_evidence_keyword_id="kw-alt",
        hard_exclusion_reason=None,
        primary_keyword_id="kw-primary",
        market_mode="fragmented_workflow",
        serp_servedness_score=0.25,
        serp_competitor_density=0.2,
        avg_difficulty=40.0,
    )
    primary_kw = SimpleNamespace(
        id="kw-primary",
        serp_top_results=[],
        serp_mismatch_flags=[],
        difficulty=40.0,
        validated_intent=None,
        validated_page_type=None,
    )
    alternate_kw = SimpleNamespace(
        id="kw-alt",
        serp_top_results=[{"domain": "reddit.com"}],
        serp_mismatch_flags=[],
        difficulty=40.0,
        validated_intent="informational",
        validated_page_type="guide",
    )
    supervisor = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)
    supervisor.project_id = "project-1"
    supervisor.run = SimpleNamespace(id="run-1")
    supervisor.session = _SessionSequence([topic], [primary_kw, alternate_kw])

    decisions = await supervisor._evaluate_topic_decisions(  # type: ignore[attr-defined]
        iteration_index=1,
        discovery=DiscoveryLoopConfig(
            require_serp_gate=True,
            require_intent_match=True,
            max_keyword_difficulty=65.0,
            min_domain_diversity=0.5,
            max_serp_servedness=0.75,
            max_serp_competitor_density=0.70,
            min_serp_intent_confidence=0.35,
        ),
    )

    assert len(decisions) == 1
    assert decisions[0].decision == "accepted"


@pytest.mark.asyncio
async def test_evaluate_topic_decisions_rejects_workflow_topic_when_saturated() -> None:
    topic = SimpleNamespace(
        id="topic-2",
        name="CRM Integration",
        fit_tier="primary",
        fit_score=0.77,
        prioritization_diagnostics={"fit_reasons": []},
        serp_intent_confidence=0.9,
        serp_evidence_keyword_id="kw-primary",
        hard_exclusion_reason=None,
        primary_keyword_id="kw-primary",
        market_mode="fragmented_workflow",
        serp_servedness_score=0.9,
        serp_competitor_density=0.9,
        avg_difficulty=30.0,
    )
    primary_kw = SimpleNamespace(
        id="kw-primary",
        serp_top_results=[{"domain": "vendor-a.com"}, {"domain": "vendor-b.com"}],
        serp_mismatch_flags=[],
        difficulty=30.0,
        validated_intent="commercial",
        validated_page_type="landing",
    )
    supervisor = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)
    supervisor.project_id = "project-1"
    supervisor.run = SimpleNamespace(id="run-1")
    supervisor.session = _SessionSequence([topic], [primary_kw])

    decisions = await supervisor._evaluate_topic_decisions(  # type: ignore[attr-defined]
        iteration_index=1,
        discovery=DiscoveryLoopConfig(
            require_serp_gate=True,
            require_intent_match=True,
            max_keyword_difficulty=65.0,
            min_domain_diversity=0.5,
            max_serp_servedness=0.75,
            max_serp_competitor_density=0.70,
            min_serp_intent_confidence=0.35,
        ),
    )

    assert len(decisions) == 1
    assert decisions[0].decision == "rejected"
    assert any("workflow_serp_saturated" in reason for reason in decisions[0].rejection_reasons)


@pytest.mark.asyncio
async def test_evaluate_topic_decisions_rejects_secondary_when_off_goal() -> None:
    topic = SimpleNamespace(
        id="topic-3",
        name="Helpdesk Login Portal",
        dominant_intent="navigational",
        fit_tier="secondary",
        fit_score=0.62,
        prioritization_diagnostics={"fit_reasons": []},
        serp_intent_confidence=0.0,
        serp_evidence_keyword_id="kw-primary",
        hard_exclusion_reason=None,
        primary_keyword_id="kw-primary",
        market_mode="established_category",
        avg_difficulty=18.0,
    )
    keyword = SimpleNamespace(
        id="kw-primary",
        serp_top_results=[{"domain": "vendor-a.com"}, {"domain": "vendor-b.com"}],
        serp_mismatch_flags=["intent_mismatch"],
        difficulty=18.0,
        validated_intent="navigational",
        validated_page_type="landing",
    )
    supervisor = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)
    supervisor.project_id = "project-1"
    supervisor.run = SimpleNamespace(
        id="run-1",
        steps_config={"primary_goal": "revenue_content", "strategy": {}},
    )
    supervisor.session = _SessionSequence([topic], [keyword])

    decisions = await supervisor._evaluate_topic_decisions(  # type: ignore[attr-defined]
        iteration_index=1,
        discovery=DiscoveryLoopConfig(
            require_serp_gate=True,
            require_intent_match=True,
            max_keyword_difficulty=65.0,
            min_domain_diversity=0.5,
            max_serp_servedness=0.75,
            max_serp_competitor_density=0.70,
            min_serp_intent_confidence=0.35,
        ),
    )

    assert len(decisions) == 1
    assert decisions[0].decision == "rejected"
    assert any("secondary_tier_strict_gate" in reason for reason in decisions[0].rejection_reasons)


@pytest.mark.asyncio
async def test_evaluate_topic_decisions_accepts_secondary_with_strict_thresholds() -> None:
    topic = SimpleNamespace(
        id="topic-3b",
        name="Helpdesk Pricing Alternatives",
        dominant_intent="commercial",
        fit_tier="secondary",
        fit_score=0.71,
        prioritization_diagnostics={"fit_reasons": []},
        serp_intent_confidence=1.0,
        serp_evidence_keyword_id="kw-primary",
        hard_exclusion_reason=None,
        primary_keyword_id="kw-primary",
        market_mode="established_category",
        avg_difficulty=40.0,
    )
    keyword = SimpleNamespace(
        id="kw-primary",
        serp_top_results=[{"domain": "vendor-a.com"}, {"domain": "vendor-b.com"}],
        serp_mismatch_flags=[],
        difficulty=40.0,
        validated_intent="commercial",
        validated_page_type="comparison",
    )
    supervisor = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)
    supervisor.project_id = "project-1"
    supervisor.run = SimpleNamespace(
        id="run-1",
        steps_config={"primary_goal": "revenue_content", "strategy": {}},
    )
    supervisor.session = _SessionSequence([topic], [keyword])

    decisions = await supervisor._evaluate_topic_decisions(  # type: ignore[attr-defined]
        iteration_index=1,
        discovery=DiscoveryLoopConfig(
            require_serp_gate=True,
            require_intent_match=True,
            max_keyword_difficulty=65.0,
            min_domain_diversity=0.5,
            max_serp_servedness=0.75,
            max_serp_competitor_density=0.70,
            min_serp_intent_confidence=0.35,
        ),
    )

    assert len(decisions) == 1
    assert decisions[0].decision == "accepted"


@pytest.mark.asyncio
async def test_evaluate_topic_decisions_accepts_core_goal_intent_despite_mismatch_flag() -> None:
    topic = SimpleNamespace(
        id="topic-4",
        name="Helpdesk Pricing Software",
        dominant_intent="transactional",
        fit_tier="primary",
        fit_score=0.63,
        prioritization_diagnostics={"fit_reasons": []},
        serp_intent_confidence=1.0,
        serp_evidence_keyword_id="kw-primary",
        hard_exclusion_reason=None,
        primary_keyword_id="kw-primary",
        market_mode="established_category",
        avg_difficulty=22.0,
    )
    keyword = SimpleNamespace(
        id="kw-primary",
        serp_top_results=[{"domain": "vendor-a.com"}, {"domain": "vendor-b.com"}],
        serp_mismatch_flags=["intent_mismatch"],
        difficulty=22.0,
        validated_intent="commercial",
        validated_page_type="landing",
    )
    supervisor = DiscoveryLoopSupervisor.__new__(DiscoveryLoopSupervisor)
    supervisor.project_id = "project-1"
    supervisor.run = SimpleNamespace(
        id="run-1",
        steps_config={"primary_goal": "revenue_content", "strategy": {}},
    )
    supervisor.session = _SessionSequence([topic], [keyword])

    decisions = await supervisor._evaluate_topic_decisions(  # type: ignore[attr-defined]
        iteration_index=1,
        discovery=DiscoveryLoopConfig(
            require_serp_gate=True,
            require_intent_match=True,
            max_keyword_difficulty=65.0,
            min_domain_diversity=0.5,
            max_serp_servedness=0.75,
            max_serp_competitor_density=0.70,
            min_serp_intent_confidence=0.35,
        ),
    )

    assert len(decisions) == 1
    assert decisions[0].decision == "accepted"
