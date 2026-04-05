#!/usr/bin/env python3
"""Process all pending photos: detect → extract features.

Run locally (not in Lambda) to process the database in batches.
Requires DATABASE_URL and GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY env vars.

Usage:
    python process.py [--detect-only] [--extract-only] [--batch-size 50]
"""

import argparse
import os
import sys

# Load env from cloud/.env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from db import get_pending_photos, get_detected_photos


def main():
    parser = argparse.ArgumentParser(description="Process pending photos")
    parser.add_argument("--detect-only", action="store_true", help="Only run detection")
    parser.add_argument("--extract-only", action="store_true", help="Only run extraction")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    args = parser.parse_args()

    if not args.extract_only:
        print("=== Detection Phase ===")
        from detector import detect_batch

        while True:
            photos = get_pending_photos(limit=args.batch_size)
            if not photos:
                print("No more pending photos to detect.")
                break
            print(f"\nDetecting batch of {len(photos)} photos...")
            counts = detect_batch(photos)
            print(f"  Results: {counts}")
            if counts.get("SKIPPED", 0) == len(photos):
                print(f"All {len(photos)} photos in batch were skipped (unreadable). Stopping detection.")
                break

    if not args.detect_only:
        print("\n=== Extraction Phase ===")
        from extractor import extract_batch

        while True:
            photos = get_detected_photos(limit=args.batch_size)
            if not photos:
                print("No more detected photos to extract.")
                break
            print(f"\nExtracting batch of {len(photos)} photos...")
            counts = extract_batch(photos)
            print(f"  Results: {counts}")

    print("\nDone.")


if __name__ == "__main__":
    main()
