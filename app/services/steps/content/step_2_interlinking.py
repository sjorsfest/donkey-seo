"""Step 2: Intelligent Interlinking Enrichment.

Enhances content briefs with semantic internal linking recommendations.
Supports both sitemap-based linking (to existing content) and batch cross-linking
(between articles being written).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.integrations.embeddings import EmbeddingsClient
from app.integrations.sitemap_fetcher import SitemapFetcher
from app.models.content import ContentBrief
from app.models.project import Project
from app.models.topic import Topic
from app.services.interlinking_service import InterlinkingResult, InterlinkingService
from app.services.steps.base_step import BaseStepService

logger = logging.getLogger(__name__)


@dataclass
class InterlinkingInput:
    """Input for Step 2."""

    project_id: str
    brief_ids: list[str] | None = None  # Process all briefs if None
    fetch_sitemap: bool = True
    min_relevance_score: float = 0.65
    max_links_per_brief: int = 8
    max_sitemap_links: int = 5
    max_batch_links: int = 3
    max_sitemap_urls: int = 1000


@dataclass
class InterlinkingOutput:
    """Output from Step 2."""

    briefs_enhanced: int
    sitemap_links_added: int
    batch_links_added: int
    sitemap_pages_found: int
    low_relevance_skipped: int
    result_details: dict[str, Any] = field(default_factory=dict)


class Step2InterlinkingService(BaseStepService[InterlinkingInput, InterlinkingOutput]):
    """Step 2: Intelligent Interlinking Enrichment.

    Enhances content briefs with high-quality internal link recommendations:
    - Sitemap-based links to existing published content
    - Semantic cross-links between briefs in the current batch
    - Relevance scoring based on embeddings, intent, funnel stage, and keywords
    - Contextual anchor text generation
    - Bi-directional link tracking

    Key features:
    - Uses semantic similarity (embeddings) for relevance scoring
    - Filters out cannibalization conflicts
    - Populates both internal_links_out and internal_links_in
    - Gracefully handles missing sitemap
    """

    step_number = 2
    step_name = "interlinking_enrichment"
    is_optional = True  # Gracefully skips if issues arise

    async def _validate_preconditions(self, input_data: InterlinkingInput) -> None:
        """Validate required upstream artifacts exist."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        # Check content briefs exist (Step 1 output)
        briefs_result = await self.session.execute(
            select(ContentBrief).where(
                ContentBrief.project_id == input_data.project_id,
            ).limit(1)
        )
        if not briefs_result.scalars().first():
            raise ValueError("No content briefs found. Run Step 1 (content_brief) first.")

    async def _execute(self, input_data: InterlinkingInput) -> InterlinkingOutput:
        """Execute interlinking enrichment."""
        logger.info(
            "Starting interlinking enrichment",
            extra={
                "project_id": input_data.project_id,
                "fetch_sitemap": input_data.fetch_sitemap,
            },
        )

        await self._update_progress(5, "Loading content briefs...")

        # Load briefs to process
        if input_data.brief_ids:
            briefs_result = await self.session.execute(
                select(ContentBrief).where(
                    ContentBrief.project_id == input_data.project_id,
                    ContentBrief.id.in_(input_data.brief_ids),
                )
            )
        else:
            # Process all briefs for this project
            briefs_result = await self.session.execute(
                select(ContentBrief).where(
                    ContentBrief.project_id == input_data.project_id,
                )
            )

        briefs = list(briefs_result.scalars())

        if not briefs:
            logger.warning(
                "No briefs to process",
                extra={"project_id": input_data.project_id},
            )
            return InterlinkingOutput(
                briefs_enhanced=0,
                sitemap_links_added=0,
                batch_links_added=0,
                sitemap_pages_found=0,
                low_relevance_skipped=0,
            )

        await self._update_progress(10, f"Loading topics for {len(briefs)} briefs...")

        # Load topics for briefs
        topic_ids = [brief.topic_id for brief in briefs if brief.topic_id]
        topics_result = await self.session.execute(
            select(Topic).where(Topic.id.in_(topic_ids))
        )
        topics_list = list(topics_result.scalars())
        topics_dict = {str(topic.id): topic for topic in topics_list}

        await self._update_progress(15, "Fetching sitemap...")

        # Fetch sitemap if enabled
        sitemap_pages = []
        if input_data.fetch_sitemap:
            sitemap_pages = await self._fetch_sitemap(
                project_id=input_data.project_id,
                max_urls=input_data.max_sitemap_urls,
            )

            if sitemap_pages:
                logger.info(
                    "Sitemap fetched successfully",
                    extra={"pages": len(sitemap_pages)},
                )
            else:
                logger.info("No sitemap found, continuing with batch-only linking")

        await self._update_progress(30, "Generating semantic embeddings...")

        # Initialize services
        async with EmbeddingsClient() as embeddings_client:
            interlinking_service = InterlinkingService(
                session=self.session,
                embeddings_client=embeddings_client,
                min_relevance_score=input_data.min_relevance_score,
                max_links_per_brief=input_data.max_links_per_brief,
                max_sitemap_links=input_data.max_sitemap_links,
                max_batch_links=input_data.max_batch_links,
            )

            await self._update_progress(40, "Analyzing semantic relationships...")

            # Analyze and enrich links
            result = await interlinking_service.analyze_and_enrich_links(
                briefs=briefs,
                topics=topics_dict,
                sitemap_pages=sitemap_pages or None,
            )

        await self._update_progress(90, "Saving enhanced briefs...")

        logger.info(
            "Interlinking enrichment complete",
            extra={
                "briefs_enhanced": result.briefs_enhanced,
                "sitemap_links": result.sitemap_links_added,
                "batch_links": result.batch_links_added,
                "low_relevance_skipped": result.low_relevance_skipped,
            },
        )

        return InterlinkingOutput(
            briefs_enhanced=result.briefs_enhanced,
            sitemap_links_added=result.sitemap_links_added,
            batch_links_added=result.batch_links_added,
            sitemap_pages_found=result.sitemap_pages_found,
            low_relevance_skipped=result.low_relevance_skipped,
            result_details={
                "min_relevance_score": input_data.min_relevance_score,
                "max_links_per_brief": input_data.max_links_per_brief,
                "sitemap_fetched": input_data.fetch_sitemap,
            },
        )

    async def _fetch_sitemap(
        self,
        project_id: str,
        max_urls: int = 1000,
    ) -> list:
        """Fetch sitemap for the project's domain.

        Args:
            project_id: Project ID
            max_urls: Maximum URLs to fetch

        Returns:
            List of SitemapPage objects, or empty list if not available
        """
        # Load project to get domain
        result = await self.session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            logger.warning("Project not found", extra={"project_id": project_id})
            return []

        # Get domain from project (assuming it has a domain field)
        # Adjust this based on your Project model structure
        domain = getattr(project, "domain", None) or getattr(project, "website", None)

        if not domain:
            logger.info(
                "No domain configured for project, skipping sitemap fetch",
                extra={"project_id": project_id},
            )
            return []

        try:
            async with SitemapFetcher(max_urls=max_urls) as fetcher:
                pages = await fetcher.fetch_sitemap(domain)
                return pages
        except Exception as e:
            logger.warning(
                "Failed to fetch sitemap",
                extra={
                    "project_id": project_id,
                    "domain": domain,
                    "error": str(e),
                },
            )
            return []

    async def _persist_results(self, result: InterlinkingOutput) -> None:
        """Persist enhanced briefs to database.

        Note: The InterlinkingService already modified the ContentBrief objects
        in memory. This method just commits those changes.
        """
        # Briefs were already updated in memory by InterlinkingService
        # Just commit the session to persist changes
        await self.session.commit()

        logger.info(
            "Enhanced briefs persisted",
            extra={"briefs_enhanced": result.briefs_enhanced},
        )
