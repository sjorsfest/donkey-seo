"""Backfill content pillar taxonomy and brief assignments for existing projects."""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from app.core.database import get_session_context
from app.models.content import ContentBrief
from app.models.project import Project
from app.models.topic import Topic
from app.services.content_pillar_service import ContentPillarService


async def backfill_project(*, project_id: str) -> tuple[int, int]:
    """Backfill one project's taxonomy and brief assignments.

    Returns:
        Tuple of (taxonomy_topics_scanned, briefs_assigned).
    """
    async with get_session_context() as session:
        topic_result = await session.execute(
            select(Topic).where(
                Topic.project_id == project_id,
                Topic.fit_tier.in_(["primary", "secondary"]),
            )
        )
        topics = list(topic_result.scalars().all())

        service = ContentPillarService(session)
        assignments = await service.plan_assignments(project_id=project_id, topics=topics)

        brief_result = await session.execute(
            select(ContentBrief).where(ContentBrief.project_id == project_id)
        )
        briefs = list(brief_result.scalars().all())

        assigned = 0
        for brief in briefs:
            assignment = assignments.get(str(brief.topic_id))
            if assignment is None:
                continue
            await service.persist_assignments(
                project_id=project_id,
                brief_id=str(brief.id),
                assignment=assignment,
            )
            assigned += 1

        await session.commit()
        return len(topics), assigned


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", help="Backfill a single project id")
    args = parser.parse_args()

    if args.project_id:
        scanned, assigned = await backfill_project(project_id=args.project_id)
        print(
            f"project={args.project_id} topics_scanned={scanned} briefs_assigned={assigned}"
        )
        return

    async with get_session_context() as session:
        project_result = await session.execute(select(Project.id))
        project_ids = [str(project_id) for project_id in project_result.scalars().all()]

    for project_id in project_ids:
        scanned, assigned = await backfill_project(project_id=project_id)
        print(f"project={project_id} topics_scanned={scanned} briefs_assigned={assigned}")


if __name__ == "__main__":
    asyncio.run(main())
