"""Unit tests for Step 5 secondary intent taxonomy and scoring."""

from app.services.steps.discovery.step_05_intent import Step05IntentService


def test_derive_intent_layer_workflow_integration() -> None:
    service = Step05IntentService.__new__(Step05IntentService)

    layer = service._derive_intent_layer(  # type: ignore[attr-defined]
        keyword="connect slack to notion",
        intent="informational",
        discovery_signals={
            "has_action_verb": True,
            "has_integration_term": True,
            "has_two_entities": True,
            "is_comparison": False,
            "word_count": 4,
        },
    )

    assert layer == "workflow_integration"


def test_derive_intent_layer_comparison_replacement() -> None:
    service = Step05IntentService.__new__(Step05IntentService)

    layer = service._derive_intent_layer(  # type: ignore[attr-defined]
        keyword="zendesk alternative",
        intent="commercial",
        discovery_signals={"is_comparison": True},
    )

    assert layer == "comparison_replacement"


def test_calculate_intent_score_weights_signals() -> None:
    service = Step05IntentService.__new__(Step05IntentService)

    workflow_score = service._calculate_intent_score(  # type: ignore[attr-defined]
        intent_layer="workflow_integration",
        intent_confidence=0.85,
        discovery_signals={
            "has_action_verb": True,
            "has_integration_term": True,
            "has_two_entities": True,
            "is_comparison": False,
        },
    )
    category_score = service._calculate_intent_score(  # type: ignore[attr-defined]
        intent_layer="category",
        intent_confidence=0.85,
        discovery_signals={},
    )

    assert workflow_score > category_score
    assert 0 <= workflow_score <= 1
