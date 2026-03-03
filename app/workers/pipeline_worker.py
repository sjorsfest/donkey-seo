"""Pipeline queue worker process entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress
from datetime import datetime, timezone

from app.config import settings
from app.core.database import close_db
from app.core.logging import setup_logging
from app.core.redis import close_redis
from app.services.discovery_pipeline_halt import (
    reconcile_discovery_auto_halted_runs,
    write_discovery_reconciliation_metrics,
)
from app.services.pipeline_task_manager import (
    PipelineModule,
    PipelineTaskWorker,
    get_pipeline_task_manager,
)

logger = logging.getLogger(__name__)

ALL_MODULES: tuple[PipelineModule, ...] = ("setup", "discovery", "content")


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
