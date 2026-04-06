#!/usr/bin/env python3
"""Fan out detection/extraction to Lambda for parallel processing.

Usage:
    python fan_out.py detect [--batch-size 20] [--max-concurrent 10]
    python fan_out.py extract [--batch-size 20] [--max-concurrent 10]
    python fan_out.py status
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

import config
config.init()

from db import get_pending_photos, get_detected_photos, get_connection

FUNCTION_NAME = os.environ.get("ML_WORKER_LAMBDA_NAME", "horse-id-ml-worker")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

lambda_client = boto3.client("lambda", region_name=REGION)


def invoke_lambda(task: str, photos: list[dict]) -> dict:
    """Invoke the ML worker Lambda with a batch of photos."""
    payload = {
        "task": task,
        "photos": [
            {
                "id": p["id"],
                "horse_id": p["horse_id"],
                "drive_file_id": p["drive_file_id"],
                "filename": p["filename"],
            }
            for p in photos
        ],
    }

    response = lambda_client.invoke(
        FunctionName=FUNCTION_NAME,
        InvocationType="Event",  # async — returns immediately
        Payload=json.dumps(payload),
    )

    return {"StatusCode": response["StatusCode"], "batch_size": len(photos)}


def get_status():
    """Print current processing status from DB."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT processing_status, detection_result, count(*)
            FROM photos
            GROUP BY processing_status, detection_result
            ORDER BY processing_status, detection_result
        """)
        rows = cur.fetchall()

    print(f"\n{'Status':<20} {'Detection':<15} {'Count':>8}")
    print("-" * 45)
    total = 0
    for status, detection, count in rows:
        print(f"{status or 'NULL':<20} {detection or 'NULL':<15} {count:>8}")
        total += count
    print("-" * 45)
    print(f"{'Total':<36} {total:>8}")

    # Feature counts
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM features")
        feature_count = cur.fetchone()[0]
    print(f"\nFeatures extracted: {feature_count}")


def fan_out(task: str, batch_size: int, max_concurrent: int):
    """Fan out work to Lambda in parallel batches."""
    if task == "detect":
        all_photos = get_pending_photos(limit=10000)
        label = "detection"
    elif task == "extract":
        all_photos = get_detected_photos(limit=10000)
        label = "extraction"
    else:
        raise ValueError(f"Unknown task: {task}")

    if not all_photos:
        print(f"No photos pending {label}.")
        return

    # Split into batches
    batches = [all_photos[i : i + batch_size] for i in range(0, len(all_photos), batch_size)]
    print(f"Fanning out {len(all_photos)} photos for {label} across {len(batches)} batches (max {max_concurrent} concurrent)")

    invoked = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(invoke_lambda, task, batch): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                result = future.result()
                invoked += 1
                if result["StatusCode"] == 202:
                    print(f"  Batch {batch_idx + 1}/{len(batches)}: invoked ({result['batch_size']} photos)")
                else:
                    print(f"  Batch {batch_idx + 1}/{len(batches)}: unexpected status {result['StatusCode']}")
                    errors += 1
            except Exception as e:
                print(f"  Batch {batch_idx + 1}/{len(batches)}: ERROR {e}")
                errors += 1

    print(f"\nDone: {invoked} batches invoked, {errors} errors")
    print("Lambda functions are running asynchronously. Use 'python fan_out.py status' to monitor progress.")


def main():
    parser = argparse.ArgumentParser(description="Fan out ML processing to Lambda")
    parser.add_argument("command", choices=["detect", "extract", "status"])
    parser.add_argument("--batch-size", type=int, default=20, help="Photos per Lambda invocation")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent Lambda invocations")
    args = parser.parse_args()

    if args.command == "status":
        get_status()
    else:
        fan_out(args.command, args.batch_size, args.max_concurrent)


if __name__ == "__main__":
    main()
