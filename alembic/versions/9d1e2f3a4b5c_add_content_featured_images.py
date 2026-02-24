"""add content featured images artifacts

Revision ID: 9d1e2f3a4b5c
Revises: 8c3d4e5f6a7b
Create Date: 2026-02-24 14:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "9d1e2f3a4b5c"
down_revision: Union[str, None] = "8c3d4e5f6a7b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "content_featured_images",
        sa.Column("project_id", sa.String(length=32), nullable=False),
        sa.Column("brief_id", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("title_text", sa.String(length=500), nullable=False),
        sa.Column("style_variant_id", sa.String(length=120), nullable=True),
        sa.Column("template_version", sa.String(length=50), nullable=True),
        sa.Column("template_spec", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("object_key", sa.String(length=1024), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=50), nullable=True),
        sa.Column("generation_error", sa.Text(), nullable=True),
        sa.Column("last_generated_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["brief_id"], ["content_briefs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("brief_id", name="uq_content_featured_images_brief_id"),
    )

    op.create_index(
        "ix_content_featured_images_brief_id",
        "content_featured_images",
        ["brief_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_featured_images_project_id",
        "content_featured_images",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_featured_images_project_status",
        "content_featured_images",
        ["project_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_content_featured_images_status",
        "content_featured_images",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_content_featured_images_status", table_name="content_featured_images")
    op.drop_index("ix_content_featured_images_project_status", table_name="content_featured_images")
    op.drop_index("ix_content_featured_images_project_id", table_name="content_featured_images")
    op.drop_index("ix_content_featured_images_brief_id", table_name="content_featured_images")
    op.drop_table("content_featured_images")
