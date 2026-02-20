"""add brand visual asset columns

Revision ID: e6f7a8b9c0d1
Revises: f3a1b2c4d5e6
Create Date: 2026-02-19 18:06:10.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, None] = "f3a1b2c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "brand_profiles",
        sa.Column("brand_assets", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "brand_profiles",
        sa.Column("visual_style_guide", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "brand_profiles",
        sa.Column("visual_prompt_contract", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "brand_profiles",
        sa.Column("visual_extraction_confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "brand_profiles",
        sa.Column("visual_last_synced_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("brand_profiles", "visual_last_synced_at")
    op.drop_column("brand_profiles", "visual_extraction_confidence")
    op.drop_column("brand_profiles", "visual_prompt_contract")
    op.drop_column("brand_profiles", "visual_style_guide")
    op.drop_column("brand_profiles", "brand_assets")
