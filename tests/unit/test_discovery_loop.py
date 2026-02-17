"""Unit tests for discovery loop strategy and filtering helpers."""

from app.services.discovery_loop import DiscoveryLoopSupervisor, TopicDecision


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
