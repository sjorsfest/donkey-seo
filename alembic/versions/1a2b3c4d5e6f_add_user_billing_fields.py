"""add user billing fields for stripe subscription sync

Revision ID: 1a2b3c4d5e6f
Revises: e1f2a3b4c5d6
Create Date: 2026-02-23 13:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1a2b3c4d5e6f"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("stripe_customer_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("stripe_subscription_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("subscription_plan", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("subscription_interval", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column(
            "subscription_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'inactive'"),
        ),
    )
    op.add_column(
        "users",
        sa.Column("subscription_current_period_end", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("subscription_trial_ends_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("stripe_price_id", sa.String(length=255), nullable=True),
    )

    op.create_index(
        op.f("ix_users_stripe_customer_id"),
        "users",
        ["stripe_customer_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_users_stripe_subscription_id"),
        "users",
        ["stripe_subscription_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_users_subscription_status"),
        "users",
        ["subscription_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_users_subscription_status"), table_name="users")
    op.drop_index(op.f("ix_users_stripe_subscription_id"), table_name="users")
    op.drop_index(op.f("ix_users_stripe_customer_id"), table_name="users")

    op.drop_column("users", "stripe_price_id")
    op.drop_column("users", "subscription_trial_ends_at")
    op.drop_column("users", "subscription_current_period_end")
    op.drop_column("users", "subscription_status")
    op.drop_column("users", "subscription_interval")
    op.drop_column("users", "subscription_plan")
    op.drop_column("users", "stripe_subscription_id")
    op.drop_column("users", "stripe_customer_id")
