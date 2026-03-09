"""Discovery pipeline reconciliation worker entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress
from datetime import datetime, time, timedelta, timezone

from app.config import settings
from app.core.database import close_db
from app.core.logging import setup_logging
from app.core.redis import close_redis, get_redis_client
from app.services.discovery_pipeline_halt import (
    reconcile_discovery_auto_halted_runs,
    write_discovery_reconciliation_metrics,
)

logger = logging.getLogger(__name__)

DISCOVERY_RECONCILIATION_LAST_RUN_KEY = "pipeline:discovery:reconciliation:last_run"
DISCOVERY_RECONCILIATION_DEFAULT_HOUR_UTC = 0


def parse_args() -> argparse.Namespace:
    """Parse worker runtime arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-hour-utc",
        type=int,
        default=DISCOVERY_RECONCILIATION_DEFAULT_HOUR_UTC,
        help="Hour of day (UTC) to run daily reconciliation (0-23).",
    )
    parser.add_argument(
        "--max-projects",
        type=int,
        default=200,
        help="Maximum projects to process per reconciliation sweep.",
    )
    return parser.parse_args()


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def next_reconciliation_run_at(*, run_hour_utc: int, now: datetime | None = None) -> datetime:
    """Calculate the next scheduled reconciliation run timestamp."""
    now_utc = (now or _utc_now()).astimezone(timezone.utc)
    hour = max(0, min(23, int(run_hour_utc)))

    # Calculate today's run time
    today_run_time = datetime.combine(
        now_utc.date(),
        time(hour=hour, minute=0, second=0, tzinfo=timezone.utc),
    )

    # If we haven't reached today's run time yet, run today
    if now_utc < today_run_time:
        return today_run_time

    # Otherwise, run tomorrow
    return today_run_time + timedelta(days=1)


async def get_last_run_timestamp() -> datetime | None:
    """Retrieve the last successful reconciliation run timestamp from Redis."""
    try:
        redis = get_redis_client()
        raw = await redis.get(DISCOVERY_RECONCILIATION_LAST_RUN_KEY)
        if raw is None:
            return None
        timestamp_str = str(raw)
        return datetime.fromisoformat(timestamp_str)
    except Exception:
        logger.exception("Failed to read last reconciliation run timestamp")
        return None


async def set_last_run_timestamp(timestamp: datetime) -> None:
    """Persist the last successful reconciliation run timestamp to Redis."""
    try:
        redis = get_redis_client()
        # Store with 7-day TTL to prevent stale data accumulation
        await redis.set(
            DISCOVERY_RECONCILIATION_LAST_RUN_KEY,
            timestamp.isoformat(),
            ex=7 * 24 * 60 * 60,
        )
    except Exception:
        logger.exception("Failed to write last reconciliation run timestamp")


async def should_run_missed_reconciliation(*, run_hour_utc: int) -> bool:
    """Check if we missed a scheduled reconciliation run and should catch up."""
    last_run = await get_last_run_timestamp()
    if last_run is None:
        # Never run before, so run now
        return True

    now_utc = _utc_now()
    last_scheduled_run = next_reconciliation_run_at(run_hour_utc=run_hour_utc, now=now_utc)

    # If the last scheduled run time is after our last actual run, we missed it
    if last_scheduled_run > last_run:
        # But only if enough time has passed (e.g., we're past the scheduled time)
        if now_utc >= last_scheduled_run:
            return True

    return False


async def run_reconciliation_sweep(*, max_projects: int) -> int:
    """Execute one reconciliation sweep and return count of resumed runs."""
    sweep_started_at = _utc_now()
    try:
        resumed = await reconcile_discovery_auto_halted_runs(max_projects=max_projects)
        logger.info(
            "Discovery reconciliation sweep completed",
            extra={"resumed_runs": resumed, "max_projects": max_projects},
        )
        await write_discovery_reconciliation_metrics(
            started_at=sweep_started_at,
            finished_at=_utc_now(),
            status="ok",
            resumed_runs=resumed,
            error_message=None,
        )
        await set_last_run_timestamp(sweep_started_at)
        return resumed
    except Exception as exc:
        logger.exception("Discovery reconciliation sweep failed")
        await write_discovery_reconciliation_metrics(
            started_at=sweep_started_at,
            finished_at=_utc_now(),
            status="error",
            resumed_runs=0,
            error_message=str(exc),
        )
        raise


async def run_worker(*, run_hour_utc: int, max_projects: int) -> None:
    """Run daily discovery reconciliation worker loop until shutdown signal."""
    setup_logging()

    hour = max(0, min(23, int(run_hour_utc)))

    # Debug: Force output to confirm code is executing
    print(f"[DEBUG] Worker starting: run_hour_utc={hour}, max_projects={max_projects}")

    logger.info(
        "Discovery reconciliation worker started",
        extra={
            "run_hour_utc": hour,
            "max_projects": max_projects,
        },
    )

    print("[DEBUG] Logger.info called for worker started")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        if not stop_event.is_set():
            logger.info("Discovery reconciliation worker received shutdown signal")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    try:
        # Always run reconciliation sweep on startup (ensures fresh deployments trigger immediately)
        print("[DEBUG] About to run initial reconciliation sweep")
        logger.info("Running initial reconciliation sweep on startup")
        try:
            await run_reconciliation_sweep(max_projects=max_projects)
            print("[DEBUG] Initial reconciliation sweep completed successfully")
        except Exception:
            print("[DEBUG] Initial reconciliation sweep failed with exception")
            logger.exception("Initial reconciliation sweep failed")

        # Main scheduling loop
        while not stop_event.is_set():
            next_run_at = next_reconciliation_run_at(run_hour_utc=hour)
            delay_seconds = max(0.0, (next_run_at - _utc_now()).total_seconds())

            logger.info(
                "Next reconciliation sweep scheduled",
                extra={
                    "next_run_at": next_run_at.isoformat(),
                    "delay_seconds": int(delay_seconds),
                },
            )

            if delay_seconds > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay_seconds)
                    # If we get here, stop was requested
                    continue
                except asyncio.TimeoutError:
                    # Timeout reached, time to run
                    pass

            # Run the reconciliation sweep
            try:
                await run_reconciliation_sweep(max_projects=max_projects)
            except Exception:
                logger.exception("Reconciliation sweep failed, will retry at next scheduled time")

    finally:
        logger.info("Discovery reconciliation worker stopping")
        await close_redis()
        await close_db()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    try:
        asyncio.run(
            run_worker(
                run_hour_utc=args.run_hour_utc,
                max_projects=args.max_projects,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
