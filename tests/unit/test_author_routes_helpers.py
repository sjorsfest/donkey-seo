"""Tests for author route helper behavior."""

from __future__ import annotations

import pytest

from app.api.v1.authors.routes import _build_profile_image_upload_object_key
from app.schemas.author import AuthorProfileImageSignedUploadRequest


def test_build_profile_image_upload_object_key_uses_expected_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.v1.authors.routes.generate_cuid", lambda: "c123")

    key = _build_profile_image_upload_object_key(
        project_id="project_1",
        author_id="author_1",
        content_type="image/png",
    )

    assert key == "projects/project_1/authors/author_1/uploads/c123.png"


def test_signed_upload_request_normalizes_content_type() -> None:
    payload = AuthorProfileImageSignedUploadRequest(content_type="IMAGE/JPEG; charset=utf-8")
    assert payload.content_type == "image/jpeg"


def test_signed_upload_request_rejects_non_image_content_type() -> None:
    with pytest.raises(ValueError):
        AuthorProfileImageSignedUploadRequest(content_type="application/pdf")
