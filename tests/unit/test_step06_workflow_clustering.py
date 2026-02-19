"""Unit tests for workflow-aware Step 6 clustering behavior."""

from types import SimpleNamespace

from app.services.steps.discovery.step_06_clustering import Step06ClusteringService


def test_stage1_workflow_clustering_uses_entity_pair_constraints() -> None:
    service = Step06ClusteringService.__new__(Step06ClusteringService)
    keywords = [
        SimpleNamespace(
            keyword="connect integration workflow",
            intent="informational",
            discovery_signals={
                "matched_entities": ["slack", "notion"],
                "workflow_verb": "connect",
                "comparison_target": None,
                "core_noun_phrase": "integration",
            },
        ),
        SimpleNamespace(
            keyword="connect integration workflow",
            intent="informational",
            discovery_signals={
                "matched_entities": ["slack", "jira"],
                "workflow_verb": "connect",
                "comparison_target": None,
                "core_noun_phrase": "integration",
            },
        ),
    ]

    clusters = service._stage1_coarse_cluster(  # type: ignore[attr-defined]
        keywords=keywords,
        workflowish_mode=True,
    )

    assert len(clusters) == 2


def test_select_primary_keyword_workflow_prefers_clear_intent_over_noise() -> None:
    service = Step06ClusteringService.__new__(Step06ClusteringService)
    candidates = [
        SimpleNamespace(
            keyword="slack integration",
            intent_score=0.65,
            difficulty=20.0,
            discovery_signals={
                "word_count": 2,
                "has_action_verb": False,
                "has_integration_term": True,
                "has_two_entities": False,
                "is_comparison": False,
            },
        ),
        SimpleNamespace(
            keyword="how to connect slack to notion webhook quickly",
            intent_score=0.92,
            difficulty=30.0,
            discovery_signals={
                "word_count": 8,
                "has_action_verb": True,
                "has_integration_term": True,
                "has_two_entities": True,
                "is_comparison": False,
            },
        ),
    ]

    primary = service._select_primary_keyword_workflow(  # type: ignore[attr-defined]
        keywords=candidates,
        cluster_embeddings=None,
    )

    assert primary is not None
    assert primary.keyword == "how to connect slack to notion webhook quickly"
