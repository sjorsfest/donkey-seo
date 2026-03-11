#!/usr/bin/env python3
"""Advanced webhook processor with more options."""

import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import get_session_context
from app.services.publication_webhook import (
    PublicationWebhookProcessor,
    process_due_publication_webhook_deliveries,
    process_publication_webhook_deliveries_for_date,
)


async def process_all():
    """Process all due webhooks."""
    print("🚀 Processing all due publication webhooks...")
    total = 0
    batch = 1

    while True:
        processed = await process_due_publication_webhook_deliveries(batch_size=50)
        if processed == 0:
            break
        total += processed
        print(f"✅ Batch {batch}: Processed {processed} deliveries")
        batch += 1

    print(f"\n🎉 Complete! Total: {total} deliveries")
    return total


async def process_for_today():
    """Process webhooks scheduled for today."""
    today = datetime.now().date()
    print(f"🚀 Processing webhooks for {today}...")
    total = 0
    batch = 1

    while True:
        processed = await process_publication_webhook_deliveries_for_date(
            publication_date=today, batch_size=50
        )
        if processed == 0:
            break
        total += processed
        print(f"✅ Batch {batch}: Processed {processed} deliveries")
        batch += 1

    print(f"\n🎉 Complete! Total: {total} deliveries for {today}")
    return total


async def process_single_delivery(delivery_id: str):
    """Process a specific delivery by ID."""
    print(f"🚀 Processing delivery {delivery_id}...")

    async with get_session_context() as session:
        processor = PublicationWebhookProcessor(
            session=session,
            delivery_id=delivery_id,
        )
        success = await processor.run()

    if success:
        print(f"✅ Successfully processed delivery {delivery_id}")
    else:
        print(f"❌ Failed to process delivery {delivery_id}")

    return success


async def show_help():
    """Show usage instructions."""
    print(
        """
📚 Webhook Processor Usage:

python run_webhooks_advanced.py [command] [args]

Commands:
  all              - Process all due webhooks (default)
  today            - Process webhooks scheduled for today
  delivery <id>    - Process a specific delivery by ID
  help             - Show this help message

Examples:
  python run_webhooks_advanced.py
  python run_webhooks_advanced.py all
  python run_webhooks_advanced.py today
  python run_webhooks_advanced.py delivery 123e4567-e89b-12d3-a456-426614174000
"""
    )


async def main():
    """Main entry point."""
    args = sys.argv[1:]

    if not args or args[0] == "all":
        await process_all()
    elif args[0] == "today":
        await process_for_today()
    elif args[0] == "delivery":
        if len(args) < 2:
            print("❌ Error: delivery command requires an ID")
            print("Usage: python run_webhooks_advanced.py delivery <delivery_id>")
            sys.exit(1)
        await process_single_delivery(args[1])
    elif args[0] == "help":
        await show_help()
    else:
        print(f"❌ Unknown command: {args[0]}")
        await show_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
