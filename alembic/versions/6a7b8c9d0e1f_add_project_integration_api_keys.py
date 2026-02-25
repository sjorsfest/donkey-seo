"""add project integration api key fields

Revision ID: 6a7b8c9d0e1f
Revises: 9d1e2f3a4b5c
Create Date: 2026-02-24 18:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6a7b8c9d0e1f"
down_revision: Union[str, None] = "9d1e2f3a4b5c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("integration_api_key_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("integration_api_key_last4", sa.String(length=4), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("integration_api_key_created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_projects_integration_api_key_hash",
        "projects",
        ["integration_api_key_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_projects_integration_api_key_hash", table_name="projects")
    op.drop_column("projects", "integration_api_key_created_at")
    op.drop_column("projects", "integration_api_key_last4")
    op.drop_column("projects", "integration_api_key_hash")
