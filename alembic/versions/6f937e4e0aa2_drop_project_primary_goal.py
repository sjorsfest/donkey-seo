"""drop project primary goal

Revision ID: 6f937e4e0aa2
Revises: e4e5d051eaeb
Create Date: 2026-03-04 11:45:23.918094

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f937e4e0aa2'
down_revision: Union[str, None] = 'e4e5d051eaeb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("posts_per_week", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("projects", "posts_per_week", server_default=None)
    op.drop_column("projects", "primary_goal")


def downgrade() -> None:
    op.drop_column("projects", "posts_per_week")
    op.add_column("projects", sa.Column("primary_goal", sa.String(length=100), nullable=True))
