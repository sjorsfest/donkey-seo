"""split pipeline modules clean

Revision ID: f3a1b2c4d5e6
Revises: d9e8f7a6b5c4
Create Date: 2026-02-19 17:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op
from app.models.base import StringUUID

# revision identifiers, used by Alembic.
revision: str = "f3a1b2c4d5e6"
down_revision: Union[str, None] = "d9e8f7a6b5c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column(
            "pipeline_module",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'discovery'"),
        ),
    )
    op.add_column(
        "pipeline_runs",
        sa.Column(
            "parent_run_id",
            StringUUID(),
            nullable=True,
        ),
    )
    op.add_column(
        "pipeline_runs",
        sa.Column(
            "source_topic_id",
            StringUUID(),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_pipeline_runs_parent_run_id_pipeline_runs",
        "pipeline_runs",
        "pipeline_runs",
        ["parent_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_pipeline_runs_source_topic_id_topics",
        "pipeline_runs",
        "topics",
        ["source_topic_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_pipeline_runs_project_module_status",
        "pipeline_runs",
        ["project_id", "pipeline_module", "status"],
        unique=False,
    )
    op.create_index(
        "ix_pipeline_runs_parent_source_module",
        "pipeline_runs",
        ["parent_run_id", "source_topic_id", "pipeline_module"],
        unique=False,
    )
    op.create_index(
        "uq_pipeline_runs_content_dispatch_once",
        "pipeline_runs",
        ["parent_run_id", "source_topic_id", "pipeline_module"],
        unique=True,
        postgresql_where=sa.text(
            "pipeline_module = 'content' "
            "AND parent_run_id IS NOT NULL "
            "AND source_topic_id IS NOT NULL"
        ),
    )
    op.alter_column("pipeline_runs", "pipeline_module", server_default=None)


def downgrade() -> None:
    op.drop_index(
        "uq_pipeline_runs_content_dispatch_once",
        table_name="pipeline_runs",
    )
    op.drop_index(
        "ix_pipeline_runs_parent_source_module",
        table_name="pipeline_runs",
    )
    op.drop_index(
        "ix_pipeline_runs_project_module_status",
        table_name="pipeline_runs",
    )
    op.drop_constraint(
        "fk_pipeline_runs_source_topic_id_topics",
        "pipeline_runs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_pipeline_runs_parent_run_id_pipeline_runs",
        "pipeline_runs",
        type_="foreignkey",
    )
    op.drop_column("pipeline_runs", "source_topic_id")
    op.drop_column("pipeline_runs", "parent_run_id")
    op.drop_column("pipeline_runs", "pipeline_module")
