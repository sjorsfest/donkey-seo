"""market aware discovery

Revision ID: b1e2d3f4a5b6
Revises: a5d4e3f2c1b0
Create Date: 2026-02-17 22:05:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1e2d3f4a5b6"
down_revision: Union[str, None] = "a5d4e3f2c1b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("keywords", sa.Column("adjusted_volume", sa.Integer(), nullable=True))
    op.add_column("keywords", sa.Column("intent_layer", sa.String(length=50), nullable=True))
    op.add_column("keywords", sa.Column("intent_score", sa.Float(), nullable=True))
    op.add_column(
        "keywords",
        sa.Column("discovery_signals", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.add_column("topics", sa.Column("adjusted_volume_sum", sa.Integer(), nullable=True))
    op.add_column("topics", sa.Column("market_mode", sa.String(length=50), nullable=True))
    op.add_column("topics", sa.Column("demand_fragmentation_index", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("serp_servedness_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("serp_competitor_density", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("topics", "serp_competitor_density")
    op.drop_column("topics", "serp_servedness_score")
    op.drop_column("topics", "demand_fragmentation_index")
    op.drop_column("topics", "market_mode")
    op.drop_column("topics", "adjusted_volume_sum")

    op.drop_column("keywords", "discovery_signals")
    op.drop_column("keywords", "intent_score")
    op.drop_column("keywords", "intent_layer")
    op.drop_column("keywords", "adjusted_volume")

