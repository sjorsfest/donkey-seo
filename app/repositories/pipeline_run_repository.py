"""Repository for PipelineRun read/write operations."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from app.core.database import get_session_context
from app.core.db_retry import run_with_transient_db_retry
from app.models.generated_dtos import PipelineRunPatchDTO
from app.models.pipeline import PipelineRun

logger = logging.getLogger(__name__)


class PipelineRunRepository:
    """Handles PipelineRun updates via short-lived sessions."""

    def __init__(self, *, project_id: str | None = None) -> None:
        self.project_id = project_id

    async def patch(
        self,
        *,
        run_id: str,
        updates: Mapping[str, Any],
        attempts: int = 3,
    ) -> None:
        """Patch a pipeline run with transient connection retries."""
        if not updates:
            return

        run_id_str = str(run_id)

        async def _patch_once() -> None:
            async with get_session_context() as session:
                stmt = select(PipelineRun).where(PipelineRun.id == run_id_str)
                if self.project_id is not None:
                    stmt = stmt.where(PipelineRun.project_id == self.project_id)
                result = await session.execute(stmt)
                run = result.scalar_one_or_none()
                if run is None:
                    if self.project_id is not None:
                        raise ValueError(
                            f"Pipeline run not found for project {self.project_id}: {run_id_str}"
                        )
                    raise ValueError(f"Pipeline run not found: {run_id_str}")
                run.patch(session, PipelineRunPatchDTO.from_partial(dict(updates)))

        await run_with_transient_db_retry(
            _patch_once,
            operation_name="pipeline_run_patch",
            attempts=attempts,
            log_context={"project_id": self.project_id, "run_id": run_id_str},
        )
