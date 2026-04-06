"""Feature extraction using wildlife-mega-L-384 model."""

import os
import numpy as np
import pandas as pd
try:
    import pi_heif
    pi_heif.register_heif_opener()
except ImportError:
    pass
import timm
import torchvision.transforms as T
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import ImageDataset

from drive_client import download_image
from db import update_photo_status, insert_feature

MODEL_NAME = "hf-hub:BVRA/wildlife-mega-L-384"
_default_cache = "/opt/ml/model" if os.path.isdir("/opt/ml") else os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
MODEL_CACHE = os.environ.get("MODEL_CACHE_DIR", _default_cache)
IMAGE_SIZE = 384

_backbone = None
_extractor = None
_transform = None


def get_extractor():
    global _backbone, _extractor, _transform
    if _extractor is None:
        _backbone = timm.create_model(
            MODEL_NAME, num_classes=0, pretrained=True, cache_dir=MODEL_CACHE
        )
        _extractor = DeepFeatures(_backbone, num_workers=0)
        _transform = T.Compose([
            T.Resize([IMAGE_SIZE, IMAGE_SIZE]),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return _extractor, _transform


def extract_single(image_path: str) -> np.ndarray:
    """Extract feature embedding from a single image. Returns 384-dim vector."""
    extractor, transform = get_extractor()

    df = pd.DataFrame([{
        "path": os.path.basename(image_path),
        "identity": -1,
    }])
    dataset = ImageDataset(df, os.path.dirname(image_path), transform=transform)
    features = extractor(dataset)
    return features.features[0]


def extract_batch(photos: list[dict]) -> dict:
    """Extract features for a batch of photos (should be SINGLE-detected only).

    Args:
        photos: list of dicts with keys: id, horse_id, drive_file_id, filename

    Returns:
        dict with extracted/error counts
    """
    counts = {"extracted": 0, "error": 0}

    for photo in photos:
        try:
            update_photo_status(photo["id"], "extracting")
            image_path = download_image(photo["drive_file_id"])

            try:
                embedding = extract_single(image_path)
                insert_feature(photo["id"], photo["horse_id"], embedding.tolist())
                update_photo_status(photo["id"], "ready")
                counts["extracted"] += 1
                print(f"  {photo['filename']}: extracted ({len(embedding)}-dim)")
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

        except Exception as e:
            print(f"  ERROR {photo['filename']}: {e}")
            update_photo_status(photo["id"], "error")
            counts["error"] += 1

    return counts
