"""add proposed publication date to content briefs

Revision ID: a5d4e3f2c1b0
Revises: 4a6b0c1d2e3f
Create Date: 2026-02-17 18:10:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a5d4e3f2c1b0"
down_revision: Union[str, None] = "4a6b0c1d2e3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "content_briefs",
        sa.Column("proposed_publication_date", sa.Date(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("content_briefs", "proposed_publication_date")
