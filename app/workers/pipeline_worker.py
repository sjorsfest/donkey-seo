"""Pipeline queue worker process entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from contextlib import suppress
from datetime import datetime, timezone
from typing import Awaitable, cast

from sqlalchemy import select

from app.config import settings
from app.core.database import close_db, get_session_context
from app.core.logging import setup_logging
from app.core.redis import close_redis, get_redis_client
from app.models.pipeline import PipelineRun
from app.services.discovery_pipeline_halt import (
    reconcile_discovery_auto_halted_runs,
    write_discovery_reconciliation_metrics,
)
from app.services.pipeline_task_manager import (
    PipelineQueueFullError,
    PipelineModule,
    PipelineTaskWorker,
    get_pipeline_task_manager,
)
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

ALL_MODULES: tuple[PipelineModule, ...] = ("setup", "discovery", "content")
RECOVERABLE_PIPELINE_STATUSES: tuple[str, ...] = ("pending", "running")
RECOVERY_TASK_STAGE = "Recovered pipeline run after worker restart"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        action="append",
        choices=["all", "setup", "discovery", "content"],
        help="Pipeline module to process. Repeat to select multiple modules.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=5,
        help="Redis blocking-pop timeout (seconds).",
    )
    parser.add_argument(
        "--requeue-delay",
        type=float,
        default=1.0,
        help="Delay before requeue when module is already running.",
    )
    return parser.parse_args()


def resolve_modules(selected: list[str] | None) -> list[PipelineModule]:
    """Resolve requested modules to a deterministic ordered list."""
    if not selected or "all" in selected:
        return list(ALL_MODULES)

    modules: list[PipelineModule] = []
    for module in selected:
        if module == "all":
            continue
        if module == "setup":
            typed_module: PipelineModule = "setup"
        elif module == "discovery":
            typed_module = "discovery"
        else:
            typed_module = "content"
        if typed_module not in modules:
            modules.append(typed_module)
    return modules


async def run_workers(
    *,
    modules: list[PipelineModule],
    poll_timeout: int,
    requeue_delay: float,
) -> None:
    """Start workers and block until a shutdown signal arrives."""
    setup_logging()

    try:
        await _recover_stale_pipeline_runs_on_startup(modules=modules)
    except Exception:
        logger.exception(
            "Startup stale-run recovery sweep failed; continuing worker startup",
            extra={"modules": modules},
        )

    workers: list[PipelineTaskWorker] = []
    for module in modules:
        manager = get_pipeline_task_manager(module)
        worker = PipelineTaskWorker(
            manager=manager,
            poll_timeout_seconds=poll_timeout,
            requeue_delay_seconds=requeue_delay,
        )
        await worker.start()
        workers.append(worker)

    logger.info("Pipeline worker process started", extra={"modules": modules})

    stop_event = asyncio.Event()
    reconciliation_task: asyncio.Task[None] | None = None
    if "discovery" in modules:
        reconciliation_task = asyncio.create_task(
            _run_discovery_auto_halt_reconciliation_loop(stop_event),
            name="discovery-auto-halt-reconciliation",
        )
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        if not stop_event.is_set():
            logger.info("Shutdown signal received")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping pipeline worker process")
        if reconciliation_task is not None:
            reconciliation_task.cancel()
            await asyncio.gather(reconciliation_task, return_exceptions=True)
        for worker in workers:
            await worker.stop()
        await close_redis()
        await close_db()


async def _recover_stale_pipeline_runs_on_startup(*, modules: list[PipelineModule]) -> int:
    """Re-enqueue pending/running runs that have no queued job after worker restart."""
    active_modules: list[PipelineModule] = []
    for module in modules:
        if module not in active_modules:
            active_modules.append(module)
    if not active_modules:
        return 0

    queued_run_ids = await _load_queued_run_ids(modules=active_modules)
    managers = {module: get_pipeline_task_manager(module) for module in active_modules}

    async with get_session_context(commit_on_exit=False) as session:
        result = await session.execute(
            select(
                PipelineRun.id,
                PipelineRun.project_id,
                PipelineRun.pipeline_module,
                PipelineRun.status,
                PipelineRun.paused_at_step,
                PipelineRun.source_topic_id,
            )
            .where(
                PipelineRun.pipeline_module.in_(active_modules),
                PipelineRun.status.in_(list(RECOVERABLE_PIPELINE_STATUSES)),
            )
            .order_by(PipelineRun.updated_at.asc())
        )
    rows = list(result.all())

    if not rows:
        logger.info(
            "Startup stale-run recovery sweep found no recoverable runs",
            extra={"modules": active_modules},
        )
        return 0

    task_manager = TaskManager()
    recovered = 0
    skipped_already_queued = 0
    failed_enqueue = 0

    for run_id, project_id, pipeline_module, status, paused_at_step, source_topic_id in rows:
        module_str = str(pipeline_module)
        if module_str not in managers:
            continue
        module = cast(PipelineModule, module_str)
        run_id_str = str(run_id)
        if run_id_str in queued_run_ids[module]:
            skipped_already_queued += 1
            continue

        manager = managers[module]
        project_id_str = str(project_id)
        source_topic_id_str = str(source_topic_id) if source_topic_id is not None else None
        is_pending = str(status) == "pending"

        try:
            if is_pending:
                await manager.enqueue_start(
                    project_id=project_id_str,
                    run_id=run_id_str,
                )
            else:
                await manager.enqueue_resume(
                    project_id=project_id_str,
                    run_id=run_id_str,
                )
        except PipelineQueueFullError:
            failed_enqueue += 1
            logger.warning(
                "Startup stale-run recovery skipped enqueue because queue is full",
                extra={
                    "pipeline_module": module,
                    "project_id": project_id_str,
                    "run_id": run_id_str,
                    "status": str(status),
                },
            )
            continue
        except Exception:
            failed_enqueue += 1
            logger.exception(
                "Startup stale-run recovery enqueue failed",
                extra={
                    "pipeline_module": module,
                    "project_id": project_id_str,
                    "run_id": run_id_str,
                    "status": str(status),
                },
            )
            continue

        queued_run_ids[module].add(run_id_str)
        recovered += 1
        current_step = int(paused_at_step) if paused_at_step is not None else None
        await task_manager.set_task_state(
            task_id=run_id_str,
            status="queued",
            stage=RECOVERY_TASK_STAGE,
            project_id=project_id_str,
            pipeline_module=module,
            source_topic_id=source_topic_id_str,
            current_step=current_step,
            current_step_name=None,
            error_message=None,
        )

    logger.info(
        "Startup stale-run recovery sweep finished",
        extra={
            "modules": active_modules,
            "candidates": len(rows),
            "recovered_runs": recovered,
            "skipped_already_queued": skipped_already_queued,
            "failed_enqueues": failed_enqueue,
        },
    )
    return recovered


async def _load_queued_run_ids(*, modules: list[PipelineModule]) -> dict[PipelineModule, set[str]]:
    """Return run IDs currently present in module queues for dedupe."""
    redis = get_redis_client()
    queued_run_ids: dict[PipelineModule, set[str]] = {module: set() for module in modules}
    for module in modules:
        queue_key = f"pipeline:queue:{module}"
        try:
            payloads = await cast(Awaitable[list[str]], redis.lrange(queue_key, 0, -1))
        except Exception as exc:
            logger.warning(
                "Failed to read queue while building startup recovery dedupe set",
                extra={
                    "pipeline_module": module,
                    "error": str(exc),
                },
            )
            continue
        for payload in payloads:
            try:
                data = json.loads(str(payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            run_id = data.get("run_id")
            if isinstance(run_id, str) and run_id:
                queued_run_ids[module].add(run_id)

    return queued_run_ids


async def _run_discovery_auto_halt_reconciliation_loop(stop_event: asyncio.Event) -> None:
    """Sweep paused discovery runs and auto-resume eligible projects."""
    interval_seconds = max(60, int(settings.discovery_pipeline_halt_reconcile_interval_seconds))
    logger.info(
        "Discovery auto-halt reconciliation loop started",
        extra={"interval_seconds": interval_seconds},
    )
    try:
        while not stop_event.is_set():
            sweep_started_at = datetime.now(timezone.utc)
            try:
                resumed = await reconcile_discovery_auto_halted_runs()
                logger.info(
                    "Discovery auto-halt reconciliation sweep finished",
                    extra={"resumed_runs": resumed},
                )
                await write_discovery_reconciliation_metrics(
                    started_at=sweep_started_at,
                    finished_at=datetime.now(timezone.utc),
                    status="ok",
                    resumed_runs=resumed,
                    error_message=None,
                )
            except Exception:
                logger.exception("Discovery auto-halt reconciliation sweep failed")
                await write_discovery_reconciliation_metrics(
                    started_at=sweep_started_at,
                    finished_at=datetime.now(timezone.utc),
                    status="error",
                    resumed_runs=0,
                    error_message="reconciliation_sweep_failed",
                )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=float(interval_seconds))
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        return


def main() -> int:
    """Run the worker process."""
    args = parse_args()
    modules = resolve_modules(args.module)
    try:
        asyncio.run(
            run_workers(
                modules=modules,
                poll_timeout=args.poll_timeout,
                requeue_delay=args.requeue_delay,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
