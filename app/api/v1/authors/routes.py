"""Author profile API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.v1.authors.constants import AUTHOR_NOT_FOUND_DETAIL
from app.api.v1.dependencies import get_user_project
from app.dependencies import CurrentUser, DbSession
from app.models.author import Author
from app.models.generated_dtos import AuthorCreateDTO, AuthorPatchDTO
from app.schemas.author import AuthorCreate, AuthorListResponse, AuthorResponse, AuthorUpdate
from app.services.author_profiles import (
    build_author_profile_image_signed_url,
    sync_author_profile_image_from_source,
)

router = APIRouter()


@router.get(
    "/{project_id}",
    response_model=AuthorListResponse,
    summary="List project authors",
    description="Return all author profiles configured for a project.",
)
async def list_authors(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> AuthorListResponse:
    """List all author profiles for a project."""
    await get_user_project(project_id, current_user, session)

    result = await session.execute(
        select(Author)
        .where(Author.project_id == project_id)
        .order_by(Author.created_at.asc())
    )
    items = list(result.scalars())
    return AuthorListResponse(
        items=[_to_author_response(author) for author in items],
        total=len(items),
    )


@router.get(
    "/{project_id}/{author_id}",
    response_model=AuthorResponse,
    summary="Get project author",
    description="Return one author profile by ID within a project.",
)
async def get_author(
    project_id: str,
    author_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> AuthorResponse:
    """Get an author profile by ID."""
    await get_user_project(project_id, current_user, session)
    author = await _get_project_author_or_404(session=session, project_id=project_id, author_id=author_id)
    return _to_author_response(author)


@router.post(
    "/{project_id}",
    response_model=AuthorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create project author",
    description="Create a new author profile for byline attribution.",
)
async def create_author(
    project_id: str,
    payload: AuthorCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> AuthorResponse:
    """Create an author profile and optionally ingest profile image into R2."""
    await get_user_project(project_id, current_user, session)

    author = Author.create(
        session,
        AuthorCreateDTO(
            project_id=project_id,
            name=payload.name,
            bio=payload.bio,
            social_urls=payload.social_urls,
            basic_info=payload.basic_info,
            profile_image_source_url=payload.profile_image_source_url,
        ),
    )

    if payload.profile_image_source_url:
        try:
            await sync_author_profile_image_from_source(
                session=session,
                author=author,
                strict=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to ingest author profile image: {exc}",
            ) from exc

    await session.flush()
    await session.refresh(author)
    return _to_author_response(author)


@router.patch(
    "/{project_id}/{author_id}",
    response_model=AuthorResponse,
    summary="Update project author",
    description="Patch mutable author profile fields.",
)
async def update_author(
    project_id: str,
    author_id: str,
    payload: AuthorUpdate,
    current_user: CurrentUser,
    session: DbSession,
) -> AuthorResponse:
    """Patch an author profile."""
    await get_user_project(project_id, current_user, session)
    author = await _get_project_author_or_404(session=session, project_id=project_id, author_id=author_id)

    update_data = payload.model_dump(exclude_unset=True)
    author.patch(
        session,
        AuthorPatchDTO.from_partial(update_data),
    )

    if "profile_image_source_url" in update_data:
        try:
            await sync_author_profile_image_from_source(
                session=session,
                author=author,
                strict=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to ingest author profile image: {exc}",
            ) from exc

    await session.flush()
    await session.refresh(author)
    return _to_author_response(author)


@router.delete(
    "/{project_id}/{author_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete project author",
    description="Delete an author profile from a project.",
)
async def delete_author(
    project_id: str,
    author_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    """Delete an author profile."""
    await get_user_project(project_id, current_user, session)
    author = await _get_project_author_or_404(session=session, project_id=project_id, author_id=author_id)
    await author.delete(session)


async def _get_project_author_or_404(
    *,
    session: DbSession,
    project_id: str,
    author_id: str,
) -> Author:
    result = await session.execute(
        select(Author).where(
            Author.project_id == project_id,
            Author.id == author_id,
        )
    )
    author = result.scalar_one_or_none()
    if author is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=AUTHOR_NOT_FOUND_DETAIL,
        )
    return author


def _to_author_response(author: Author) -> AuthorResponse:
    return AuthorResponse(
        id=str(author.id),
        project_id=str(author.project_id),
        name=author.name,
        bio=author.bio,
        social_urls=author.social_urls,
        basic_info=author.basic_info,
        profile_image_source_url=author.profile_image_source_url,
        profile_image_object_key=author.profile_image_object_key,
        profile_image_mime_type=author.profile_image_mime_type,
        profile_image_width=author.profile_image_width,
        profile_image_height=author.profile_image_height,
        profile_image_byte_size=author.profile_image_byte_size,
        profile_image_sha256=author.profile_image_sha256,
        profile_image_signed_url=build_author_profile_image_signed_url(author),
        created_at=author.created_at,
        updated_at=author.updated_at,
    )
