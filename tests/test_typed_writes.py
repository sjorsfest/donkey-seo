"""Tests for typed write layer."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.generated_dtos import (
    TopicPatchDTO,
    UserCreateDTO,
    UserPatchDTO,
)
from app.models.topic import Topic
from app.models.user import User
from app.persistence.typed import create, delete, patch
from app.persistence.typed.errors import InvalidPatchFieldError
from scripts.check_typed_writes import find_violations


class FakeAsyncSession:
    """Minimal async session stub for adapter tests."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.deleted: list[object] = []

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def delete(self, instance: object) -> None:
        self.deleted.append(instance)


def test_writes_create_and_patch_user() -> None:
    session = FakeAsyncSession()
    typed_session = cast(AsyncSession, session)

    user = create(
        typed_session,
        User,
        UserCreateDTO(email="user@example.com", hashed_password="hashed"),
    )

    assert isinstance(user, User)
    assert user.email == "user@example.com"
    assert session.added == [user]

    patch(
        typed_session,
        User,
        user,
        UserPatchDTO.from_partial({"full_name": "Test User"}),
    )

    assert user.full_name == "Test User"


@pytest.mark.asyncio
async def test_writes_delete_uses_adapter() -> None:
    session = FakeAsyncSession()
    typed_session = cast(AsyncSession, session)
    user = User(email="delete@example.com", hashed_password="x")

    await delete(typed_session, User, user)

    assert session.deleted == [user]


def test_patch_rejects_protected_topic_fields() -> None:
    session = FakeAsyncSession()
    typed_session = cast(AsyncSession, session)
    topic = Topic(project_id="project-1", name="Topic")

    with pytest.raises(InvalidPatchFieldError):
        patch(
            typed_session,
            Topic,
            topic,
            TopicPatchDTO.from_partial({"project_id": "new-project"}),
        )


def test_patch_dto_is_sparse() -> None:
    dto = TopicPatchDTO.from_partial({"name": "Updated", "description": None})

    assert dto.to_patch_dict() == {"name": "Updated", "description": None}


def test_guardrail_scanner_finds_direct_constructor(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    services_dir = app_dir / "services"
    services_dir.mkdir(parents=True)

    file_path = services_dir / "sample.py"
    file_path.write_text("topic = Topic(name='x', project_id='y')\n", encoding="utf-8")

    violations = find_violations([app_dir])

    assert violations
    found_path, line, _ = violations[0]
    assert found_path == file_path
    assert line == 1
