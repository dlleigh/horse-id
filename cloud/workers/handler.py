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

        # Download image from S3 if s3_key provided
        image_bytes = None
        if event.get("s3_key"):
            import boto3
            s3 = boto3.client("s3")
            resp = s3.get_object(Bucket=event["s3_bucket"], Key=event["s3_key"])
            image_bytes = resp["Body"].read()

        result = identify(
            image_bytes=image_bytes,
            drive_file_id=event.get("drive_file_id"),
            herd_id=event.get("herd_id"),
            top_k=event.get("top_k", 5),
            confidence_threshold=event.get("confidence_threshold", 0.8),
        )
        return {"status": "ok", "task": "identify", "predictions": result["predictions"]}

    elif task == "twilio_identify":
        import requests as http_requests
        from twilio.rest import Client
        from identifier import identify
        from db import get_herd_id_by_name, get_all_herd_names

        media_url = event.get("media_url")
        from_number = event.get("from_number")
        to_number = event.get("to_number")
        herd_name = event.get("herd_name")

        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        twilio_client = Client(account_sid, auth_token)

        def send_sms(body):
            twilio_client.messages.create(body=body, from_=to_number, to=from_number)

        try:
            # Resolve herd name to ID
            herd_id = None
            if herd_name:
                herd_id = get_herd_id_by_name(herd_name)
                if herd_id is None:
                    valid = get_all_herd_names()
                    msg = f"Sorry, I don't recognize the herd '{herd_name}'.\n\nValid herds:\n"
                    for h in valid:
                        msg += f"  • {h}\n"
                    send_sms(msg)
                    return {"status": "ok", "task": "twilio_identify", "error": "invalid_herd"}

            # Download image from Twilio
            print(f"Downloading image from {media_url}")
            resp = http_requests.get(media_url, auth=(account_sid, auth_token), timeout=15)
            resp.raise_for_status()
            image_bytes = resp.content

            # Run identification
            result = identify(
                image_bytes=image_bytes,
                herd_id=herd_id,
                top_k=5,
                confidence_threshold=float(os.environ.get("IDENTIFY_CONFIDENCE_THRESHOLD", "0.0")),
            )

            # Format SMS response
            predictions = result["predictions"]
            msg = "\n"
            if herd_name:
                msg += f"Searched in {herd_name}:\n\n"
            msg += "Horse Identification Results:\n"
            if predictions:
                for p in predictions:
                    msg += f"  {p['horse_name']} - {p['herd_name']} (Confidence: {p['similarity']:.1%})\n"
            else:
                msg += "  No strong match found.\n"

            send_sms(msg)
            print(f"Sent identification results to {from_number}")
            return {"status": "ok", "task": "twilio_identify", "predictions": predictions}

        except Exception as e:
            print(f"twilio_identify error: {e}")
            try:
                send_sms("Sorry, something went wrong during identification. Please try again.")
            except Exception:
                pass
            return {"status": "error", "task": "twilio_identify", "message": str(e)}

    else:
        return {"status": "error", "message": f"Unknown task: {task}"}
