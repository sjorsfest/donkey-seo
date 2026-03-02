"""add content pillars and brief assignments

Revision ID: e2f3a4b5c6d7
Revises: d4c5b6a7e8f9
Create Date: 2026-03-02 18:15:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from app.models.base import StringUUID

# revision identifiers, used by Alembic.
revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "d4c5b6a7e8f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "content_pillars",
        sa.Column("project_id", StringUUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="auto"),
        sa.Column("locked", sa.Boolean(), nullable=False, server_default=sa.false()),
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
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "slug",
            name="uq_content_pillars_project_slug",
        ),
    )
    op.create_index(
        "ix_content_pillars_project_status",
        "content_pillars",
        ["project_id", "status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_content_pillars_project_id"),
        "content_pillars",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_content_pillars_status"),
        "content_pillars",
        ["status"],
        unique=False,
    )

    op.create_table(
        "content_brief_pillar_assignments",
        sa.Column("project_id", StringUUID(), nullable=False),
        sa.Column("brief_id", StringUUID(), nullable=False),
        sa.Column("pillar_id", StringUUID(), nullable=False),
        sa.Column("relationship_type", sa.String(length=20), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("assignment_method", sa.String(length=20), nullable=False, server_default="auto"),
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
        sa.ForeignKeyConstraint(["brief_id"], ["content_briefs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pillar_id"], ["content_pillars.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "brief_id",
            "pillar_id",
            name="uq_content_brief_pillar_assignments_brief_pillar",
        ),
    )
    op.create_index(
        "ix_content_brief_pillar_assignments_project_pillar",
        "content_brief_pillar_assignments",
        ["project_id", "pillar_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_brief_pillar_assignments_project_brief",
        "content_brief_pillar_assignments",
        ["project_id", "brief_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_brief_pillar_assignments_one_primary_per_brief",
        "content_brief_pillar_assignments",
        ["brief_id"],
        unique=True,
        postgresql_where=sa.text("relationship_type = 'primary'"),
    )
    op.create_index(
        op.f("ix_content_brief_pillar_assignments_project_id"),
        "content_brief_pillar_assignments",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_content_brief_pillar_assignments_brief_id"),
        "content_brief_pillar_assignments",
        ["brief_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_content_brief_pillar_assignments_pillar_id"),
        "content_brief_pillar_assignments",
        ["pillar_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_content_brief_pillar_assignments_pillar_id"),
        table_name="content_brief_pillar_assignments",
    )
    op.drop_index(
        op.f("ix_content_brief_pillar_assignments_brief_id"),
        table_name="content_brief_pillar_assignments",
    )
    op.drop_index(
        op.f("ix_content_brief_pillar_assignments_project_id"),
        table_name="content_brief_pillar_assignments",
    )
    op.drop_index(
        "ix_content_brief_pillar_assignments_one_primary_per_brief",
        table_name="content_brief_pillar_assignments",
    )
    op.drop_index(
        "ix_content_brief_pillar_assignments_project_brief",
        table_name="content_brief_pillar_assignments",
    )
    op.drop_index(
        "ix_content_brief_pillar_assignments_project_pillar",
        table_name="content_brief_pillar_assignments",
    )
    op.drop_table("content_brief_pillar_assignments")

    op.drop_index(op.f("ix_content_pillars_status"), table_name="content_pillars")
    op.drop_index(op.f("ix_content_pillars_project_id"), table_name="content_pillars")
    op.drop_index("ix_content_pillars_project_status", table_name="content_pillars")
    op.drop_table("content_pillars")
