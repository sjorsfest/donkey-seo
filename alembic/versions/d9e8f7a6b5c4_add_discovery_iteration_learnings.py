"""add discovery iteration learnings

Revision ID: d9e8f7a6b5c4
Revises: c7d8e9f0a1b2
Create Date: 2026-02-19 16:10:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.models.base import StringUUID

# revision identifiers, used by Alembic.
revision: str = "d9e8f7a6b5c4"
down_revision: Union[str, None] = "c7d8e9f0a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "discovery_iteration_learnings",
        sa.Column("project_id", StringUUID(), nullable=False),
        sa.Column("pipeline_run_id", StringUUID(), nullable=False),
        sa.Column("iteration_index", sa.Integer(), nullable=False),
        sa.Column("source_capability", sa.String(length=100), nullable=False),
        sa.Column("source_agent", sa.String(length=100), nullable=True),
        sa.Column("learning_key", sa.String(length=255), nullable=False),
        sa.Column("learning_type", sa.String(length=50), nullable=False),
        sa.Column("polarity", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("novelty_score", sa.Float(), nullable=True),
        sa.Column("baseline_metric", sa.Float(), nullable=True),
        sa.Column("current_metric", sa.Float(), nullable=True),
        sa.Column("delta_metric", sa.Float(), nullable=True),
        sa.Column("applies_to_capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("applies_to_agents", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
        op.f("ix_discovery_iteration_learnings_project_id"),
        "discovery_iteration_learnings",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_iteration_learnings_pipeline_run_id"),
        "discovery_iteration_learnings",
        ["pipeline_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_iteration_learnings_iteration_index"),
        "discovery_iteration_learnings",
        ["iteration_index"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_iteration_learnings_learning_key"),
        "discovery_iteration_learnings",
        ["learning_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_iteration_learnings_learning_type"),
        "discovery_iteration_learnings",
        ["learning_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_discovery_iteration_learnings_status"),
        "discovery_iteration_learnings",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_discovery_iteration_learnings_project_created_at",
        "discovery_iteration_learnings",
        ["project_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_discovery_iteration_learnings_project_created_at",
        table_name="discovery_iteration_learnings",
    )
    op.drop_index(
        op.f("ix_discovery_iteration_learnings_status"),
        table_name="discovery_iteration_learnings",
    )
    op.drop_index(
        op.f("ix_discovery_iteration_learnings_learning_type"),
        table_name="discovery_iteration_learnings",
    )
    op.drop_index(
        op.f("ix_discovery_iteration_learnings_learning_key"),
        table_name="discovery_iteration_learnings",
    )
    op.drop_index(
        op.f("ix_discovery_iteration_learnings_iteration_index"),
        table_name="discovery_iteration_learnings",
    )
    op.drop_index(
        op.f("ix_discovery_iteration_learnings_pipeline_run_id"),
        table_name="discovery_iteration_learnings",
    )
    op.drop_index(
        op.f("ix_discovery_iteration_learnings_project_id"),
        table_name="discovery_iteration_learnings",
    )
    op.drop_table("discovery_iteration_learnings")
