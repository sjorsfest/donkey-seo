"""add content article artifacts and version history

Revision ID: 2c1d9af4b8e1
Revises: 9f8b7c6d5e4f
Create Date: 2026-02-17 13:45:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.models.base import StringUUID

# revision identifiers, used by Alembic.
revision: str = "2c1d9af4b8e1"
down_revision: Union[str, None] = "9f8b7c6d5e4f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "content_articles",
        sa.Column("project_id", StringUUID(), nullable=False),
        sa.Column("brief_id", StringUUID(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("slug", sa.String(length=500), nullable=False),
        sa.Column("primary_keyword", sa.String(length=500), nullable=False),
        sa.Column("modular_document", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rendered_html", sa.Text(), nullable=False),
        sa.Column("qa_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("generation_model", sa.String(length=255), nullable=True),
        sa.Column("generation_temperature", sa.Float(), nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", StringUUID(), nullable=False),
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
    )
    op.create_index(op.f("ix_content_articles_project_id"), "content_articles", ["project_id"], unique=False)
    op.create_index(op.f("ix_content_articles_status"), "content_articles", ["status"], unique=False)
    op.create_index(op.f("ix_content_articles_brief_id"), "content_articles", ["brief_id"], unique=True)

    op.create_table(
        "content_article_versions",
        sa.Column("article_id", StringUUID(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("slug", sa.String(length=500), nullable=False),
        sa.Column("primary_keyword", sa.String(length=500), nullable=False),
        sa.Column("modular_document", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rendered_html", sa.Text(), nullable=False),
        sa.Column("qa_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("change_reason", sa.Text(), nullable=True),
        sa.Column("generation_model", sa.String(length=255), nullable=True),
        sa.Column("generation_temperature", sa.Float(), nullable=True),
        sa.Column(
            "created_by_regeneration",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("id", StringUUID(), nullable=False),
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
        sa.ForeignKeyConstraint(["article_id"], ["content_articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "article_id",
            "version_number",
            name="uq_content_article_versions_article_version",
        ),
    )
    op.create_index(
        op.f("ix_content_article_versions_article_id"),
        "content_article_versions",
        ["article_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_content_article_versions_article_id"), table_name="content_article_versions")
    op.drop_table("content_article_versions")

    op.drop_index(op.f("ix_content_articles_brief_id"), table_name="content_articles")
    op.drop_index(op.f("ix_content_articles_status"), table_name="content_articles")
    op.drop_index(op.f("ix_content_articles_project_id"), table_name="content_articles")
    op.drop_table("content_articles")
