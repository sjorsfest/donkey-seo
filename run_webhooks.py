#!/usr/bin/env python3
"""Manual webhook processor - run this to process pending publication webhooks."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.publication_webhook import process_due_publication_webhook_deliveries


async def main():
    """Process all due publication webhooks."""
    print("🚀 Processing publication webhooks...")

    # Process in batches until no more deliveries
    total_processed = 0
    batch_num = 1

    while True:
        print(f"\n📦 Processing batch {batch_num}...")
        processed = await process_due_publication_webhook_deliveries(batch_size=50)

        if processed == 0:
            break

        total_processed += processed
        print(f"✅ Processed {processed} deliveries in batch {batch_num}")
        batch_num += 1

    print(f"\n🎉 Complete! Total deliveries processed: {total_processed}")

    if total_processed == 0:
        print("ℹ️  No pending webhooks found")


if __name__ == "__main__":
    asyncio.run(main())
