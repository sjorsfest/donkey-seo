"""Unit tests for onboarding bootstrap project schemas."""

import pytest

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
            "goals": {"secondary_goals": ["lead_generation"]},
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
    assert request.goals.secondary_goals == ["lead_generation"]
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


def test_onboarding_bootstrap_posts_per_week_defaults_to_one() -> None:
    request = ProjectOnboardingBootstrapRequest.model_validate(
        {
            "name": "ACME",
            "domain": "acme.com",
        }
    )

    assert request.posts_per_week == 1


def test_onboarding_bootstrap_posts_per_week_validates_bounds() -> None:
    with pytest.raises(ValueError):
        ProjectOnboardingBootstrapRequest.model_validate(
            {
                "name": "ACME",
                "domain": "acme.com",
                "posts_per_week": 0,
            }
        )
