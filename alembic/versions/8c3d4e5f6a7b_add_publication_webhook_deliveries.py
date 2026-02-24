"""add project webhook config and publication webhook deliveries

Revision ID: 8c3d4e5f6a7b
Revises: 7b4c2d1e9f0a
Create Date: 2026-02-24 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c3d4e5f6a7b"
down_revision: Union[str, None] = "7b4c2d1e9f0a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("notification_webhook", sa.String(length=2000), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("notification_webhook_secret", sa.String(length=255), nullable=True),
    )

    op.create_table(
        "publication_webhook_deliveries",
        sa.Column("project_id", sa.String(length=32), nullable=False),
        sa.Column("article_id", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            sa.String(length=30),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_http_status", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(["article_id"], ["content_articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "article_id",
            "event_type",
            name="uq_publication_webhook_deliveries_article_event",
        ),
    )

    op.create_index(
        "ix_publication_webhook_deliveries_article_id",
        "publication_webhook_deliveries",
        ["article_id"],
        unique=False,
    )
    op.create_index(
        "ix_publication_webhook_deliveries_project_id",
        "publication_webhook_deliveries",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_publication_webhook_deliveries_status",
        "publication_webhook_deliveries",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_publication_webhook_deliveries_status_next_attempt",
        "publication_webhook_deliveries",
        ["status", "next_attempt_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_publication_webhook_deliveries_status_next_attempt",
        table_name="publication_webhook_deliveries",
    )
    op.drop_index(
        "ix_publication_webhook_deliveries_status",
        table_name="publication_webhook_deliveries",
    )
    op.drop_index(
        "ix_publication_webhook_deliveries_project_id",
        table_name="publication_webhook_deliveries",
    )
    op.drop_index(
        "ix_publication_webhook_deliveries_article_id",
        table_name="publication_webhook_deliveries",
    )
    op.drop_table("publication_webhook_deliveries")

    op.drop_column("projects", "notification_webhook_secret")
    op.drop_column("projects", "notification_webhook")
