"""add discovery topic snapshots

Revision ID: 4a6b0c1d2e3f
Revises: 2c1d9af4b8e1
Create Date: 2026-02-17 17:10:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.models.base import StringUUID

# revision identifiers, used by Alembic.
revision: str = "4a6b0c1d2e3f"
down_revision: Union[str, None] = "2c1d9af4b8e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "discovery_topic_snapshots",
        sa.Column("project_id", StringUUID(), nullable=False),
        sa.Column("pipeline_run_id", StringUUID(), nullable=False),
        sa.Column("iteration_index", sa.Integer(), nullable=False),
        sa.Column("source_topic_id", StringUUID(), nullable=True),
        sa.Column("topic_name", sa.String(length=255), nullable=False),
        sa.Column("fit_tier", sa.String(length=20), nullable=True),
        sa.Column("fit_score", sa.Float(), nullable=True),
        sa.Column("keyword_difficulty", sa.Float(), nullable=True),
        sa.Column("domain_diversity", sa.Float(), nullable=True),
        sa.Column("validated_intent", sa.String(length=50), nullable=True),
        sa.Column("validated_page_type", sa.String(length=50), nullable=True),
        sa.Column("top_domains", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("decision", sa.String(length=20), nullable=False),
        sa.Column("rejection_reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_discovery_topic_snapshots_project_id"),
        "discovery_topic_snapshots",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_topic_snapshots_pipeline_run_id"),
        "discovery_topic_snapshots",
        ["pipeline_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_topic_snapshots_iteration_index"),
        "discovery_topic_snapshots",
        ["iteration_index"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_topic_snapshots_decision"),
        "discovery_topic_snapshots",
        ["decision"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_discovery_topic_snapshots_decision"),
        table_name="discovery_topic_snapshots",
    )
    op.drop_index(
        op.f("ix_discovery_topic_snapshots_iteration_index"),
        table_name="discovery_topic_snapshots",
    )
    op.drop_index(
        op.f("ix_discovery_topic_snapshots_pipeline_run_id"),
        table_name="discovery_topic_snapshots",
    )
    op.drop_index(
        op.f("ix_discovery_topic_snapshots_project_id"),
        table_name="discovery_topic_snapshots",
    )
    op.drop_table("discovery_topic_snapshots")
