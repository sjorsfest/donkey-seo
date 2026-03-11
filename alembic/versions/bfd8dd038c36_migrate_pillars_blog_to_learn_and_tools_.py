"""migrate pillars blog to learn and tools to compare

Revision ID: bfd8dd038c36
Revises: 3f0dba597dfa
Create Date: 2026-03-11 14:24:30.997906

"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = 'bfd8dd038c36'
down_revision: Union[str, None] = '3f0dba597dfa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Old slug -> (new slug, new name, new description)
PILLAR_RENAMES = {
    "blog": (
        "learn",
        "Learn",
        "Educational and awareness content: definitions, statistics, "
        "industry insights, and foundational knowledge.",
    ),
    "tools": (
        "compare",
        "Compare",
        "Decision-support content: product comparisons, best-of lists, "
        "alternatives, and evaluation frameworks.",
    ),
}

# Updated description for the pillar that keeps its slug
GUIDES_DESCRIPTION = (
    "Implementation and tactical content: how-to guides, use cases, "
    "checklists, and playbooks."
)


def upgrade() -> None:
    conn = op.get_bind()

    for old_slug, (new_slug, new_name, new_description) in PILLAR_RENAMES.items():
        # Find projects that have the old pillar but NOT the new one yet.
        # For those, we can simply rename in place — assignments follow via FK.
        conn.execute(
            text("""
                UPDATE content_pillars
                SET slug = :new_slug,
                    name = :new_name,
                    description = :new_desc
                WHERE slug = :old_slug
                  AND project_id NOT IN (
                      SELECT project_id FROM content_pillars WHERE slug = :new_slug
                  )
            """),
            {
                "old_slug": old_slug,
                "new_slug": new_slug,
                "new_name": new_name,
                "new_desc": new_description,
            },
        )

        # For projects that already have BOTH old and new pillars (e.g. partial
        # run of _ensure_allowed_pillars), re-point assignments from old -> new,
        # then delete the orphaned old pillar.
        conn.execute(
            text("""
                UPDATE content_brief_pillar_assignments
                SET pillar_id = new_p.id
                FROM content_pillars old_p
                JOIN content_pillars new_p
                  ON old_p.project_id = new_p.project_id
                 AND new_p.slug = :new_slug
                WHERE old_p.slug = :old_slug
                  AND content_brief_pillar_assignments.pillar_id = old_p.id
            """),
            {"old_slug": old_slug, "new_slug": new_slug},
        )

        conn.execute(
            text("""
                DELETE FROM content_pillars
                WHERE slug = :old_slug
                  AND project_id IN (
                      SELECT project_id FROM content_pillars WHERE slug = :new_slug
                  )
            """),
            {"old_slug": old_slug, "new_slug": new_slug},
        )

    # Update the guides pillar description to match the new config
    conn.execute(
        text("""
            UPDATE content_pillars
            SET description = :desc,
                name = 'Guides'
            WHERE slug = 'guides'
        """),
        {"desc": GUIDES_DESCRIPTION},
    )


def downgrade() -> None:
    conn = op.get_bind()

    # Reverse: learn -> blog, compare -> tools
    reverse_renames = {
        "learn": ("blog", "Blog", "Broad thought leadership, trends, general education"),
        "compare": ("tools", "Tools", "Comparisons, alternatives, software/tooling, pricing and evaluation content"),
    }

    for new_slug, (old_slug, old_name, old_description) in reverse_renames.items():
        conn.execute(
            text("""
                UPDATE content_pillars
                SET slug = :old_slug,
                    name = :old_name,
                    description = :old_desc
                WHERE slug = :new_slug
                  AND project_id NOT IN (
                      SELECT project_id FROM content_pillars WHERE slug = :old_slug
                  )
            """),
            {
                "new_slug": new_slug,
                "old_slug": old_slug,
                "old_name": old_name,
                "old_desc": old_description,
            },
        )

        conn.execute(
            text("""
                UPDATE content_brief_pillar_assignments
                SET pillar_id = old_p.id
                FROM content_pillars new_p
                JOIN content_pillars old_p
                  ON new_p.project_id = old_p.project_id
                 AND old_p.slug = :old_slug
                WHERE new_p.slug = :new_slug
                  AND content_brief_pillar_assignments.pillar_id = new_p.id
            """),
            {"new_slug": new_slug, "old_slug": old_slug},
        )

        conn.execute(
            text("""
                DELETE FROM content_pillars
                WHERE slug = :new_slug
                  AND project_id IN (
                      SELECT project_id FROM content_pillars WHERE slug = :old_slug
                  )
            """),
            {"new_slug": new_slug, "old_slug": old_slug},
        )
