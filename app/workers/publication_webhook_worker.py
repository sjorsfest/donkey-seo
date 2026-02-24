"""Publication webhook delivery worker entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress

from app.core.database import close_db
from app.core.logging import setup_logging
from app.services.publication_webhook import process_due_publication_webhook_deliveries

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse worker runtime arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Maximum due deliveries to process in one polling iteration.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait when no deliveries are due.",
    )
    return parser.parse_args()


async def run_worker(*, batch_size: int, poll_interval: float) -> None:
    """Run publication webhook worker loop until shutdown signal."""
    setup_logging()
    logger.info(
        "Publication webhook worker started",
        extra={"batch_size": batch_size, "poll_interval": poll_interval},
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
        while not stop_event.is_set():
            processed = await process_due_publication_webhook_deliveries(
                batch_size=max(1, int(batch_size))
            )
            if processed > 0:
                continue
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=max(0.1, float(poll_interval)),
                )
            except asyncio.TimeoutError:
                continue
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
                poll_interval=args.poll_interval,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
