"""add_pipeline_run_id_to_discovery_models

Revision ID: 878310ff008b
Revises: 775c3878190d
Create Date: 2026-03-09 12:17:57.505615

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '878310ff008b'
down_revision: Union[str, None] = '775c3878190d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add pipeline_run_id to seed_topics
    op.add_column('seed_topics', sa.Column('pipeline_run_id', sa.String(), nullable=True))
    op.create_index(op.f('ix_seed_topics_pipeline_run_id'), 'seed_topics', ['pipeline_run_id'], unique=False)
    op.create_foreign_key(None, 'seed_topics', 'pipeline_runs', ['pipeline_run_id'], ['id'], ondelete='CASCADE')

    # Add pipeline_run_id to keywords
    op.add_column('keywords', sa.Column('pipeline_run_id', sa.String(), nullable=True))
    op.create_index(op.f('ix_keywords_pipeline_run_id'), 'keywords', ['pipeline_run_id'], unique=False)
    op.create_foreign_key(None, 'keywords', 'pipeline_runs', ['pipeline_run_id'], ['id'], ondelete='CASCADE')

    # Add pipeline_run_id to topics
    op.add_column('topics', sa.Column('pipeline_run_id', sa.String(), nullable=True))
    op.create_index(op.f('ix_topics_pipeline_run_id'), 'topics', ['pipeline_run_id'], unique=False)
    op.create_foreign_key(None, 'topics', 'pipeline_runs', ['pipeline_run_id'], ['id'], ondelete='CASCADE')

    # Backfill existing records with pipeline_run_id from the discovery run created before them
    connection = op.get_bind()

    # Backfill seed_topics
    connection.execute(sa.text("""
        UPDATE seed_topics
        SET pipeline_run_id = (
            SELECT id
            FROM pipeline_runs
            WHERE pipeline_runs.project_id = seed_topics.project_id
              AND pipeline_runs.pipeline_module = 'discovery'
              AND pipeline_runs.created_at <= seed_topics.created_at
            ORDER BY pipeline_runs.created_at DESC
            LIMIT 1
        )
        WHERE seed_topics.pipeline_run_id IS NULL
    """))

    # Backfill keywords
    connection.execute(sa.text("""
        UPDATE keywords
        SET pipeline_run_id = (
            SELECT id
            FROM pipeline_runs
            WHERE pipeline_runs.project_id = keywords.project_id
              AND pipeline_runs.pipeline_module = 'discovery'
              AND pipeline_runs.created_at <= keywords.created_at
            ORDER BY pipeline_runs.created_at DESC
            LIMIT 1
        )
        WHERE keywords.pipeline_run_id IS NULL
    """))

    # Backfill topics
    connection.execute(sa.text("""
        UPDATE topics
        SET pipeline_run_id = (
            SELECT id
            FROM pipeline_runs
            WHERE pipeline_runs.project_id = topics.project_id
              AND pipeline_runs.pipeline_module = 'discovery'
              AND pipeline_runs.created_at <= topics.created_at
            ORDER BY pipeline_runs.created_at DESC
            LIMIT 1
        )
        WHERE topics.pipeline_run_id IS NULL
    """))


def downgrade() -> None:
    # Remove pipeline_run_id from topics
    op.drop_constraint(None, 'topics', type_='foreignkey')
    op.drop_index(op.f('ix_topics_pipeline_run_id'), table_name='topics')
    op.drop_column('topics', 'pipeline_run_id')

    # Remove pipeline_run_id from keywords
    op.drop_constraint(None, 'keywords', type_='foreignkey')
    op.drop_index(op.f('ix_keywords_pipeline_run_id'), table_name='keywords')
    op.drop_column('keywords', 'pipeline_run_id')

    # Remove pipeline_run_id from seed_topics
    op.drop_constraint(None, 'seed_topics', type_='foreignkey')
    op.drop_index(op.f('ix_seed_topics_pipeline_run_id'), table_name='seed_topics')
    op.drop_column('seed_topics', 'pipeline_run_id')
