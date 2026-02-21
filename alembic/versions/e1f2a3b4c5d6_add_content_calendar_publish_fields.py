"""add content calendar publish fields and indexes

Revision ID: e1f2a3b4c5d6
Revises: e6f7a8b9c0d1
Create Date: 2026-02-20 18:10:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "e6f7a8b9c0d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "content_articles",
        sa.Column("publish_status", sa.String(length=30), nullable=True),
    )
    op.add_column(
        "content_articles",
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "content_articles",
        sa.Column("published_url", sa.String(length=2000), nullable=True),
    )

    op.create_index(
        "ix_content_articles_publish_status",
        "content_articles",
        ["publish_status"],
        unique=False,
    )
    op.create_index(
        "ix_content_articles_published_at",
        "content_articles",
        ["published_at"],
        unique=False,
    )
    op.create_index(
        "ix_content_briefs_project_publication_date",
        "content_briefs",
        ["project_id", "proposed_publication_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_content_briefs_project_publication_date", table_name="content_briefs")
    op.drop_index("ix_content_articles_published_at", table_name="content_articles")
    op.drop_index("ix_content_articles_publish_status", table_name="content_articles")

    op.drop_column("content_articles", "published_url")
    op.drop_column("content_articles", "published_at")
    op.drop_column("content_articles", "publish_status")
