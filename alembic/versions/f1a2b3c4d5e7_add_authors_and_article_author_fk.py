"""add authors table and article author foreign key

Revision ID: f1a2b3c4d5e7
Revises: 8e9f0a1b2c3d
Create Date: 2026-02-25 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e7"
down_revision: Union[str, None] = "8e9f0a1b2c3d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "authors",
        sa.Column("project_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("bio", sa.Text(), nullable=True),
        sa.Column("social_urls", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("basic_info", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("profile_image_source_url", sa.String(length=2000), nullable=True),
        sa.Column("profile_image_object_key", sa.String(length=1024), nullable=True),
        sa.Column("profile_image_mime_type", sa.String(length=100), nullable=True),
        sa.Column("profile_image_width", sa.Integer(), nullable=True),
        sa.Column("profile_image_height", sa.Integer(), nullable=True),
        sa.Column("profile_image_byte_size", sa.Integer(), nullable=True),
        sa.Column("profile_image_sha256", sa.String(length=64), nullable=True),
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_authors_project_id"), "authors", ["project_id"], unique=False)

    op.add_column(
        "content_articles",
        sa.Column("author_id", sa.String(length=32), nullable=True),
    )
    op.create_index(
        op.f("ix_content_articles_author_id"),
        "content_articles",
        ["author_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_content_articles_author_id_authors",
        "content_articles",
        "authors",
        ["author_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_content_articles_author_id_authors",
        "content_articles",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_content_articles_author_id"), table_name="content_articles")
    op.drop_column("content_articles", "author_id")

    op.drop_index(op.f("ix_authors_project_id"), table_name="authors")
    op.drop_table("authors")
