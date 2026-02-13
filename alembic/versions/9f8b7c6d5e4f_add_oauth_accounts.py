"""add oauth accounts and allow null passwords for social users

Revision ID: 9f8b7c6d5e4f
Revises: ef7ed32ca3a8
Create Date: 2026-02-13 13:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op
from app.models.base import StringUUID

# revision identifiers, used by Alembic.
revision: str = "9f8b7c6d5e4f"
down_revision: Union[str, None] = "ef7ed32ca3a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "users",
        "hashed_password",
        existing_type=sa.String(length=255),
        nullable=True,
    )

    op.create_table(
        "oauth_accounts",
        sa.Column("user_id", StringUUID(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("provider_user_id", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
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
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "provider_user_id",
            name="uq_oauth_accounts_provider_user_id",
        ),
    )
    op.create_index(
        op.f("ix_oauth_accounts_provider"),
        "oauth_accounts",
        ["provider"],
        unique=False,
    )
    op.create_index(op.f("ix_oauth_accounts_user_id"), "oauth_accounts", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_oauth_accounts_user_id"), table_name="oauth_accounts")
    op.drop_index(op.f("ix_oauth_accounts_provider"), table_name="oauth_accounts")
    op.drop_table("oauth_accounts")

    op.alter_column(
        "users",
        "hashed_password",
        existing_type=sa.String(length=255),
        nullable=False,
    )
