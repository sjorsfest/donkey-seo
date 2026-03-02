"""add content keyword traceability tables

Revision ID: d4c5b6a7e8f9
Revises: c2d3e4f5a6b7
Create Date: 2026-03-02 16:20:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from app.models.base import StringUUID

# revision identifiers, used by Alembic.
revision: str = "d4c5b6a7e8f9"
down_revision: Union[str, None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "content_brief_keywords",
        sa.Column("brief_id", StringUUID(), nullable=False),
        sa.Column("keyword_id", StringUUID(), nullable=True),
        sa.Column("keyword_text", sa.String(length=500), nullable=False),
        sa.Column("keyword_text_normalized", sa.String(length=500), nullable=False),
        sa.Column("keyword_role", sa.String(length=20), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default=sa.text("0")),
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
        sa.ForeignKeyConstraint(["keyword_id"], ["keywords.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "brief_id",
            "keyword_role",
            "keyword_text_normalized",
            name="uq_content_brief_keywords_brief_role_text",
        ),
    )
    op.create_index(
        "ix_content_brief_keywords_brief_id",
        "content_brief_keywords",
        ["brief_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_brief_keywords_keyword_id",
        "content_brief_keywords",
        ["keyword_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_brief_keywords_keyword_role",
        "content_brief_keywords",
        ["keyword_role"],
        unique=False,
    )

    op.create_table(
        "content_article_keyword_usages",
        sa.Column("article_id", StringUUID(), nullable=False),
        sa.Column("article_version_number", sa.Integer(), nullable=False),
        sa.Column("brief_id", StringUUID(), nullable=False),
        sa.Column("brief_keyword_id", StringUUID(), nullable=True),
        sa.Column("keyword_id", StringUUID(), nullable=True),
        sa.Column("keyword_text", sa.String(length=500), nullable=False),
        sa.Column("keyword_role", sa.String(length=20), nullable=False),
        sa.Column("keyword_intent", sa.String(length=50), nullable=True),
        sa.Column("search_volume", sa.Integer(), nullable=True),
        sa.Column("adjusted_volume", sa.Integer(), nullable=True),
        sa.Column("used", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "usage_density_pct",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("in_h1", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("in_first_150_words", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("in_h2_h3", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("section_hits", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "seo_incorporation_score",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
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
        sa.ForeignKeyConstraint(["article_id"], ["content_articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["brief_id"], ["content_briefs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["brief_keyword_id"], ["content_brief_keywords.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["keyword_id"], ["keywords.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "article_id",
            "article_version_number",
            "brief_keyword_id",
            name="uq_content_article_keyword_usages_article_version_brief_keyword",
        ),
    )
    op.create_index(
        "ix_content_article_keyword_usages_article_id",
        "content_article_keyword_usages",
        ["article_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_article_keyword_usages_article_version",
        "content_article_keyword_usages",
        ["article_id", "article_version_number"],
        unique=False,
    )
    op.create_index(
        "ix_content_article_keyword_usages_brief_id",
        "content_article_keyword_usages",
        ["brief_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_article_keyword_usages_used",
        "content_article_keyword_usages",
        ["used"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_content_article_keyword_usages_used",
        table_name="content_article_keyword_usages",
    )
    op.drop_index(
        "ix_content_article_keyword_usages_brief_id",
        table_name="content_article_keyword_usages",
    )
    op.drop_index(
        "ix_content_article_keyword_usages_article_version",
        table_name="content_article_keyword_usages",
    )
    op.drop_index(
        "ix_content_article_keyword_usages_article_id",
        table_name="content_article_keyword_usages",
    )
    op.drop_table("content_article_keyword_usages")

    op.drop_index("ix_content_brief_keywords_keyword_role", table_name="content_brief_keywords")
    op.drop_index("ix_content_brief_keywords_keyword_id", table_name="content_brief_keywords")
    op.drop_index("ix_content_brief_keywords_brief_id", table_name="content_brief_keywords")
    op.drop_table("content_brief_keywords")
