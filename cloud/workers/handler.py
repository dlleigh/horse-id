"""Lambda entry point. Routes to detector, extractor, or identifier based on event.task."""

import json
import os

# Initialize config (SSM in Lambda, .env locally) before anything reads env vars
import config
config.init()


def lambda_handler(event, context):
    task = event.get("task")

    if task == "detect":
        from detector import detect_batch
        photos = event.get("photos", [])
        print(f"Detecting {len(photos)} photos...")
        result = detect_batch(photos)
        return {"status": "ok", "task": "detect", "counts": result}

    elif task == "extract":
        from extractor import extract_batch
        photos = event.get("photos", [])
        print(f"Extracting {len(photos)} photos...")
        result = extract_batch(photos)
        return {"status": "ok", "task": "extract", "counts": result}

    elif task == "identify":
        from identifier import identify
        result = identify(
            image_bytes=event.get("image_bytes"),
            drive_file_id=event.get("drive_file_id"),
            herd_id=event.get("herd_id"),
            top_k=event.get("top_k", 5),
            confidence_threshold=event.get("confidence_threshold", 0.8),
        )
        return {"status": "ok", "task": "identify", "predictions": result["predictions"]}

    else:
        return {"status": "error", "message": f"Unknown task: {task}"}
