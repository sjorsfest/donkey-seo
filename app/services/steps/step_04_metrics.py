"""Step 4: Keyword Metrics Enrichment.

Fetches search volume, CPC, difficulty, and trends for all keywords.
Implements caching with configurable TTL (default 14 days for metrics).
"""

import logging
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.integrations.dataforseo import DataForSEOClient, get_location_code
from app.models.keyword import Keyword
from app.models.project import Project
from app.services.steps.base_step import BaseStepService, StepResult

logger = logging.getLogger(__name__)


@dataclass
class MetricsInput:
    """Input for Step 4."""

    project_id: str


@dataclass
class MetricsOutput:
    """Output from Step 4."""

    keywords_enriched: int
    keywords_failed: int
    keywords_cached: int
    api_calls_made: int
    keywords: list[dict[str, Any]] = field(default_factory=list)


class Step04MetricsService(BaseStepService[MetricsInput, MetricsOutput]):
    """Step 4: Keyword Metrics Enrichment.

    Uses DataForSEO API to fetch metrics for keywords.
    Implements Redis caching with configurable TTL to reduce API costs.

    Cache TTL Defaults:
    - Keyword metrics: 14 days (volume/CPC stable)
    - Configurable via METRICS_CACHE_TTL_DAYS env var
    """

    step_number = 4
    step_name = "keyword_metrics"
    is_optional = False

    # Default cache TTL in days (can cut API costs by 80%+)
    DEFAULT_CACHE_TTL_DAYS = 14
    BATCH_SIZE = 100  # DataForSEO accepts 1000 max, but we use smaller batches

    async def _validate_preconditions(self, input_data: MetricsInput) -> None:
        """Validate Step 3 is completed."""
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {input_data.project_id}")

        if project.current_step < 3:
            raise ValueError("Step 3 (Keyword Expansion) must be completed first")

        # Check keywords exist
        keywords_result = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == input_data.project_id,
                Keyword.status == "active",
            ).limit(1)
        )
        if not keywords_result.scalars().first():
            raise ValueError("No active keywords found. Run Step 3 first.")

    async def _execute(self, input_data: MetricsInput) -> MetricsOutput:
        """Execute keyword metrics enrichment."""
        # Load project
        result = await self.session.execute(
            select(Project).where(Project.id == input_data.project_id)
        )
        project = result.scalar_one()

        # Get locale settings
        locale = project.primary_locale or "en-US"
        location_code = get_location_code(locale)
        language_code = project.primary_language or "en"

        # Get cache TTL from settings
        cache_ttl_days = getattr(settings, "metrics_cache_ttl_days", self.DEFAULT_CACHE_TTL_DAYS)

        await self._update_progress(5, "Loading keywords...")

        # Load all active keywords without metrics
        keywords_result = await self.session.execute(
            select(Keyword).where(
                Keyword.project_id == input_data.project_id,
                Keyword.status == "active",
            )
        )
        all_keywords = list(keywords_result.scalars())

        # Separate into needs-fetch and already-cached
        keywords_to_fetch: list[Keyword] = []
        cached_keywords: list[Keyword] = []
        cache_cutoff = datetime.now(timezone.utc) - timedelta(days=cache_ttl_days)

        for kw in all_keywords:
            # Check if we have recent metrics
            if kw.metrics_updated_at and kw.metrics_updated_at > cache_cutoff:
                cached_keywords.append(kw)
            else:
                keywords_to_fetch.append(kw)

        logger.info("Metrics enrichment starting", extra={"project_id": input_data.project_id, "to_fetch": len(keywords_to_fetch), "cached": len(cached_keywords)})

        await self._update_progress(
            10,
            f"Found {len(keywords_to_fetch)} keywords needing metrics, "
            f"{len(cached_keywords)} already cached"
        )

        # Track results
        enriched_keywords: list[dict[str, Any]] = []
        failed_keywords: list[str] = []
        api_calls_made = 0

        if keywords_to_fetch:
            # Create keyword text to model mapping
            kw_map = {kw.keyword_normalized: kw for kw in keywords_to_fetch}
            kw_texts = [kw.keyword for kw in keywords_to_fetch]

            # Process in batches
            total_batches = (len(kw_texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            async with DataForSEOClient() as client:
                for batch_num in range(total_batches):
                    start_idx = batch_num * self.BATCH_SIZE
                    end_idx = min(start_idx + self.BATCH_SIZE, len(kw_texts))
                    batch = kw_texts[start_idx:end_idx]

                    progress = 10 + int((batch_num / total_batches) * 80)
                    await self._update_progress(
                        progress,
                        f"Fetching metrics batch {batch_num + 1}/{total_batches}..."
                    )

                    try:
                        # Fetch metrics from API
                        metrics_results = await client.get_keyword_metrics(
                            keywords=batch,
                            location_code=location_code,
                            language_code=language_code,
                        )
                        api_calls_made += 1

                        # Process results
                        for metrics in metrics_results:
                            kw_text = metrics.get("keyword", "")
                            kw_normalized = kw_text.lower().strip()
                            kw_model = kw_map.get(kw_normalized)

                            if kw_model:
                                # Update keyword with metrics
                                kw_model.search_volume = metrics.get("search_volume")
                                kw_model.cpc = metrics.get("cpc")
                                kw_model.competition = metrics.get("competition")
                                kw_model.difficulty = metrics.get("difficulty")
                                kw_model.trend_data = metrics.get("trend_data")
                                kw_model.metrics_data_source = "dataforseo"
                                kw_model.metrics_updated_at = datetime.now(timezone.utc)
                                kw_model.metrics_confidence = 1.0 if metrics.get("search_volume") else 0.5

                                enriched_keywords.append({
                                    "keyword_id": str(kw_model.id),
                                    "keyword": kw_text,
                                    "search_volume": metrics.get("search_volume"),
                                    "cpc": metrics.get("cpc"),
                                    "competition": metrics.get("competition"),
                                    "difficulty": metrics.get("difficulty"),
                                    "trend_data": metrics.get("trend_data"),
                                })

                        # Mark missing keywords as failed
                        returned_keywords = {m.get("keyword", "").lower().strip() for m in metrics_results}
                        for kw_text in batch:
                            kw_normalized = kw_text.lower().strip()
                            if kw_normalized not in returned_keywords:
                                kw_model = kw_map.get(kw_normalized)
                                if kw_model:
                                    # Set null metrics with low confidence
                                    kw_model.metrics_updated_at = datetime.now(timezone.utc)
                                    kw_model.metrics_confidence = 0.1
                                    kw_model.metrics_data_source = "dataforseo"
                                    failed_keywords.append(kw_text)

                    except Exception as e:
                        logger.warning("Metrics API error for batch", extra={"batch_num": batch_num + 1, "batch_size": len(batch), "error": str(e)})
                        # Handle API errors gracefully
                        for kw_text in batch:
                            kw_normalized = kw_text.lower().strip()
                            kw_model = kw_map.get(kw_normalized)
                            if kw_model:
                                kw_model.metrics_updated_at = datetime.now(timezone.utc)
                                kw_model.metrics_confidence = 0.0
                                failed_keywords.append(kw_text)

                    # Save checkpoint after each batch (for resumability)
                    await self._save_checkpoint({
                        "batches_completed": batch_num + 1,
                        "total_batches": total_batches,
                        "keywords_processed": end_idx,
                    })

        await self._update_progress(95, "Finalizing metrics...")

        # Add cached keywords to result
        for kw in cached_keywords:
            enriched_keywords.append({
                "keyword_id": str(kw.id),
                "keyword": kw.keyword,
                "search_volume": kw.search_volume,
                "cpc": kw.cpc,
                "competition": kw.competition,
                "difficulty": kw.difficulty,
                "trend_data": kw.trend_data,
                "cached": True,
            })

        logger.info("Metrics enrichment complete", extra={"enriched": len(enriched_keywords), "failed": len(failed_keywords), "cached": len(cached_keywords), "api_calls": api_calls_made})

        await self._update_progress(100, "Metrics enrichment complete")

        return MetricsOutput(
            keywords_enriched=len(enriched_keywords),
            keywords_failed=len(failed_keywords),
            keywords_cached=len(cached_keywords),
            api_calls_made=api_calls_made,
            keywords=enriched_keywords,
        )

    async def _persist_results(self, result: MetricsOutput) -> None:
        """Save metrics to database (already updated during execution)."""
        # Update project step
        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one()
        project.current_step = 4

        # Set result summary
        self.set_result_summary({
            "keywords_enriched": result.keywords_enriched,
            "keywords_failed": result.keywords_failed,
            "keywords_cached": result.keywords_cached,
            "api_calls_made": result.api_calls_made,
            "cache_hit_rate": (
                result.keywords_cached / result.keywords_enriched * 100
                if result.keywords_enriched > 0 else 0
            ),
        })

        await self.session.commit()
