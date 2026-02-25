"""Unit tests for onboarding bootstrap project schemas."""

from app.schemas.project import (
    ProjectOnboardingBootstrapRequest,
)


def test_onboarding_bootstrap_request_accepts_strategy() -> None:
    request = ProjectOnboardingBootstrapRequest.model_validate(
        {
            "name": "ACME",
            "domain": "acme.com",
            "description": "Demo project",
            "primary_locale": "en-US",
            "goals": {"primary_objective": "lead_generation"},
            "strategy": {
                "scope_mode": "strict",
                "fit_threshold_profile": "aggressive",
                "include_topics": ["customer support automation"],
            },
        }
    )

    assert request.name == "ACME"
    assert request.domain == "acme.com"
    assert request.goals is not None
    assert request.goals.primary_objective == "lead_generation"
    assert request.strategy is not None
    assert request.strategy.scope_mode == "strict"


def test_onboarding_bootstrap_request_accepts_optional_authors() -> None:
    request = ProjectOnboardingBootstrapRequest.model_validate(
        {
            "name": "ACME",
            "domain": "acme.com",
            "authors": [
                {
                    "name": "Jamie Doe",
                    "bio": "SEO lead",
                    "social_urls": {"linkedin": "https://linkedin.com/in/jamie-doe"},
                    "basic_info": {"title": "Head of Content"},
                }
            ],
        }
    )

    assert request.authors is not None
    assert len(request.authors) == 1
    assert request.authors[0].name == "Jamie Doe"
