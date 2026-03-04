"""Publication webhook delivery worker entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress

from app.core.database import close_db
from app.core.logging import setup_logging
from app.services.publication_webhook import run_publication_webhook_nightly_scheduler

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse worker runtime arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum deliveries to process per nightly sweep batch.",
    )
    return parser.parse_args()


async def run_worker(*, batch_size: int) -> None:
    """Run nightly publication webhook worker loop until shutdown signal."""
    setup_logging()
    logger.info(
        "Publication webhook worker started",
        extra={"batch_size": batch_size, "run_time_utc": "00:00"},
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        if not stop_event.is_set():
            logger.info("Publication webhook worker received shutdown signal")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    try:
        await run_publication_webhook_nightly_scheduler(
            stop_event=stop_event,
            batch_size=max(1, int(batch_size)),
        )
    finally:
        logger.info("Publication webhook worker stopping")
        await close_db()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    try:
        asyncio.run(
            run_worker(
                batch_size=args.batch_size,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
