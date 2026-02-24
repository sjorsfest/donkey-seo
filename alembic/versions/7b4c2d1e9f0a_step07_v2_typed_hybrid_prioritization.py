"""step7 v2 typed hybrid prioritization

Revision ID: 7b4c2d1e9f0a
Revises: 1a2b3c4d5e6f
Create Date: 2026-02-24 10:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b4c2d1e9f0a"
down_revision: Union[str, None] = "1a2b3c4d5e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("topics", sa.Column("fit_tier", sa.String(length=20), nullable=True))
    op.add_column("topics", sa.Column("fit_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("brand_fit_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("opportunity_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("dynamic_fit_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("dynamic_opportunity_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("deterministic_priority_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("final_priority_score", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("llm_rerank_delta", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("llm_fit_adjustment", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("llm_tier_recommendation", sa.String(length=20), nullable=True))
    op.add_column("topics", sa.Column("fit_threshold_primary", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("fit_threshold_secondary", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("hard_exclusion_reason", sa.String(length=80), nullable=True))
    op.add_column("topics", sa.Column("final_cut_reason_code", sa.String(length=80), nullable=True))
    op.add_column("topics", sa.Column("serp_intent_confidence", sa.Float(), nullable=True))
    op.add_column("topics", sa.Column("serp_evidence_keyword_id", sa.String(length=36), nullable=True))
    op.add_column("topics", sa.Column("serp_evidence_source", sa.String(length=20), nullable=True))
    op.add_column("topics", sa.Column("serp_evidence_keyword_count", sa.Integer(), nullable=True))
    op.add_column("topics", sa.Column("prioritization_diagnostics", postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    op.create_index(op.f("ix_topics_fit_tier"), "topics", ["fit_tier"], unique=False)
    op.create_index(
        "ix_topics_project_fit_rank",
        "topics",
        ["project_id", "fit_tier", "priority_rank"],
        unique=False,
    )

    op.drop_column("topics", "priority_factors")


def downgrade() -> None:
    op.add_column(
        "topics",
        sa.Column("priority_factors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.drop_index("ix_topics_project_fit_rank", table_name="topics")
    op.drop_index(op.f("ix_topics_fit_tier"), table_name="topics")

    op.drop_column("topics", "prioritization_diagnostics")
    op.drop_column("topics", "serp_evidence_keyword_count")
    op.drop_column("topics", "serp_evidence_source")
    op.drop_column("topics", "serp_evidence_keyword_id")
    op.drop_column("topics", "serp_intent_confidence")
    op.drop_column("topics", "final_cut_reason_code")
    op.drop_column("topics", "hard_exclusion_reason")
    op.drop_column("topics", "fit_threshold_secondary")
    op.drop_column("topics", "fit_threshold_primary")
    op.drop_column("topics", "llm_tier_recommendation")
    op.drop_column("topics", "llm_fit_adjustment")
    op.drop_column("topics", "llm_rerank_delta")
    op.drop_column("topics", "final_priority_score")
    op.drop_column("topics", "deterministic_priority_score")
    op.drop_column("topics", "dynamic_opportunity_score")
    op.drop_column("topics", "dynamic_fit_score")
    op.drop_column("topics", "opportunity_score")
    op.drop_column("topics", "brand_fit_score")
    op.drop_column("topics", "fit_score")
    op.drop_column("topics", "fit_tier")
