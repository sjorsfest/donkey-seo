"""Author profile model for project content attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin

try:
    from app.models.generated_dtos import AuthorCreateDTO, AuthorPatchDTO
except ImportError:  # pragma: no cover - bootstrap before DTO regeneration
    AuthorCreateDTO = Any  # type: ignore[assignment]
    AuthorPatchDTO = Any  # type: ignore[assignment]

if TYPE_CHECKING:
    from app.models.content import ContentArticle
    from app.models.project import Project


class Author(
    TypedModelMixin["AuthorCreateDTO", "AuthorPatchDTO"],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """Author metadata attached to project content."""

    __tablename__ = "authors"

    project_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    social_urls: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    basic_info: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    profile_image_source_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    profile_image_object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    profile_image_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    profile_image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    profile_image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    profile_image_byte_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    profile_image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    project: Mapped[Project] = relationship("Project", back_populates="authors")
    content_articles: Mapped[list[ContentArticle]] = relationship(
        "ContentArticle",
        back_populates="author",
    )

    def __repr__(self) -> str:
        return f"<Author project={self.project_id} name={self.name}>"
