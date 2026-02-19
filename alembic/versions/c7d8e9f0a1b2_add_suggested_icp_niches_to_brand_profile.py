"""add suggested icp niches to brand profiles

Revision ID: c7d8e9f0a1b2
Revises: b1e2d3f4a5b6
Create Date: 2026-02-18 10:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, None] = "b1e2d3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "brand_profiles",
        sa.Column(
            "suggested_icp_niches",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("brand_profiles", "suggested_icp_niches")
