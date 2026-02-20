"""Unit tests for project domain validation."""

import pytest

from app.schemas.project import ProjectCreate, ProjectOnboardingBootstrapRequest


def test_project_create_accepts_valid_domain() -> None:
    model = ProjectCreate.model_validate(
        {
            "name": "Acme",
            "domain": "example.com",
        }
    )

    assert model.domain == "example.com"


def test_project_create_normalizes_url_input_to_hostname() -> None:
    model = ProjectCreate.model_validate(
        {
            "name": "Acme",
            "domain": "https://www.Example.com/pricing?x=1",
        }
    )

    assert model.domain == "www.example.com"


def test_project_create_rejects_random_string_domain() -> None:
    with pytest.raises(ValueError):
        ProjectCreate.model_validate(
            {
                "name": "Acme",
                "domain": "thisisnotadomain",
            }
        )


def test_project_create_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError):
        ProjectCreate.model_validate(
            {
                "name": "Acme",
                "domain": "ftp://example.com",
            }
        )


def test_onboarding_bootstrap_rejects_invalid_domain() -> None:
    with pytest.raises(ValueError):
        ProjectOnboardingBootstrapRequest.model_validate(
            {
                "name": "Acme",
                "domain": "random",
            }
        )
