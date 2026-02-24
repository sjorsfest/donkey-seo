"""Unit tests for project webhook settings mapping helpers."""

from app.api.v1.projects.routes import (
    _project_settings_create_fields,
    _project_settings_patch_fields,
)
from app.schemas.project import ProjectCreate


def test_project_settings_create_fields_include_webhook_configuration() -> None:
    payload = ProjectCreate.model_validate(
        {
            "name": "ACME",
            "domain": "acme.com",
            "settings": {
                "skip_steps": [2, 3],
                "notification_webhook": "https://example.com/webhooks/publish",
                "notification_webhook_secret": "secret-123",
            },
        }
    )

    fields = _project_settings_create_fields(payload)

    assert fields.skip_steps == [2, 3]
    assert fields.notification_webhook == "https://example.com/webhooks/publish"
    assert fields.notification_webhook_secret == "secret-123"


def test_project_settings_patch_fields_detect_webhook_updates() -> None:
    patch_fields, webhook_touched = _project_settings_patch_fields(
        {
            "notification_webhook": "https://example.com/webhooks/publish",
            "notification_webhook_secret": "secret-abc",
        }
    )

    assert webhook_touched is True
    assert patch_fields["notification_webhook"] == "https://example.com/webhooks/publish"
    assert patch_fields["notification_webhook_secret"] == "secret-abc"


def test_project_settings_patch_fields_handles_missing_settings() -> None:
    patch_fields, webhook_touched = _project_settings_patch_fields(None)

    assert patch_fields == {}
    assert webhook_touched is False
