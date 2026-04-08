"""Horse detection using YOLO. Classifies photos as NONE/SINGLE/MULTIPLE."""

import os
import numpy as np
import yaml
try:
    import pi_heif
    pi_heif.register_heif_opener()
except ImportError:
    pass
from PIL import Image
from ultralytics import YOLO

from drive_client import download_image
from db import update_photo_status
from lambda_utils import invoke_extraction


class UnreadableImageError(Exception):
    """Raised when a downloaded image can't be decoded."""
    pass

# Load config
_config_path = os.path.join(os.path.dirname(__file__), "config.yml")
if os.path.exists(_config_path):
    with open(_config_path) as f:
        _config = yaml.safe_load(f)
else:
    _config = {}

_detection_config = _config.get("detection", {})
YOLO_MODEL = _detection_config.get("yolo_model", "yolo11x-seg.pt")
CONFIDENCE_THRESHOLD = _detection_config.get("confidence_threshold", 0.5)
SIZE_RATIO_THRESHOLD = _detection_config.get("size_ratio_for_single_horse", 2.2)

_model = None


def get_model():
    global _model
    if _model is None:
        model_path = os.environ.get("YOLO_MODEL_PATH", YOLO_MODEL)
        _model = YOLO(model_path)
    return _model


def classify_image(image_path: str) -> str:
    """Classify an image as NONE, SINGLE, or MULTIPLE horses.

    This is a simplified version of the full horse_detection_lib logic.
    For the cloud workers, we use the core detection but skip the complex
    depth/edge analysis — the detailed classification from
    horse_detection_lib.py can be imported if needed.
    """
    model = get_model()

    # Validate the image is readable before passing to YOLO
    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as e:
        raise UnreadableImageError(f"Cannot decode image: {e}")

    try:
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    except Exception as e:
        raise UnreadableImageError(f"YOLO failed to process image: {e}")

    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return "NONE"

    # COCO class ID 17 = horse
    classes = result.boxes.cls.cpu().numpy()
    horse_indices = np.where(classes == 17)[0]

    if len(horse_indices) == 0:
        return "NONE"

    if len(horse_indices) == 1:
        return "SINGLE"

    # Multiple horses detected — check size dominance
    horse_boxes = result.boxes[horse_indices]
    areas = (horse_boxes.xywh[:, 2] * horse_boxes.xywh[:, 3]).cpu().numpy()

    # Try importing the full classification library
    try:
        from horse_detection_lib import classify_horse_detection

        img_height, img_width = result.orig_shape
        horse_masks = result.masks[horse_indices]
        classification, _, _, _ = classify_horse_detection(
            horse_boxes, horse_masks, horse_indices, areas,
            img_width, img_height, None
        )
        return classification
    except ImportError:
        # Fallback: simple size ratio check
        sorted_areas = np.sort(areas)[::-1]
        if len(sorted_areas) >= 2 and sorted_areas[1] > 0:
            ratio = sorted_areas[0] / sorted_areas[1]
            if ratio >= SIZE_RATIO_THRESHOLD:
                return "SINGLE"
        return "MULTIPLE"


def detect_batch(photos: list[dict]) -> dict:
    """Run detection on a batch of photos.

    Args:
        photos: list of dicts with keys: id, horse_id, drive_file_id, filename

    Returns:
        dict with counts of each classification
    """
    counts = {"NONE": 0, "SINGLE": 0, "MULTIPLE": 0, "ERROR": 0}
    single_photos = []

    for photo in photos:
        image_path = None
        try:
            update_photo_status(photo["id"], "detecting")
            image_path = download_image(photo["drive_file_id"])

            classification = classify_image(image_path)
            update_photo_status(photo["id"], "detected", classification)
            counts[classification] += 1
            print(f"  {photo['filename']}: {classification}")

            if classification == "SINGLE":
                single_photos.append(photo)

        except UnreadableImageError as e:
            print(f"  ERROR (unreadable) {photo['filename']}: {e}")
            update_photo_status(photo["id"], "error")
            counts["ERROR"] += 1

        except Exception as e:
            print(f"  ERROR {photo['filename']}: {e}")
            update_photo_status(photo["id"], "error")
            counts["ERROR"] += 1

        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    # Chain to extraction for SINGLE-detected photos
    if single_photos:
        invoke_extraction(single_photos)
        print(f"  Dispatched {len(single_photos)} photos for extraction")

    return counts
