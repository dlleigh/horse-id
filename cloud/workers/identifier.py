"""Identify a query image against the database using pgvector similarity."""

import os
import tempfile

from extractor import extract_single
from db import query_similar
from drive_client import download_image


def identify(
    image_path: str = None,
    image_bytes: bytes = None,
    drive_file_id: str = None,
    herd_id: int = None,
    top_k: int = 5,
    confidence_threshold: float = 0.8,
) -> dict:
    """Identify a horse from a query image.

    Provide exactly one of: image_path, image_bytes, or drive_file_id.

    Returns:
        dict with keys: predictions (list of matches), query_embedding
    """
    tmp_path = None

    try:
        if image_path:
            path = image_path
        elif image_bytes:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)
            path = tmp_path
        elif drive_file_id:
            tmp_path = download_image(drive_file_id)
            path = tmp_path
        else:
            raise ValueError("Must provide image_path, image_bytes, or drive_file_id")

        # Extract query embedding
        embedding = extract_single(path)

        # Query pgvector for similar embeddings
        results = query_similar(embedding.tolist(), limit=top_k, herd_id=herd_id)

        # Aggregate by horse (multiple photos per horse)
        horse_scores = {}
        for r in results:
            key = r["horse_id"]
            if key not in horse_scores or r["similarity"] > horse_scores[key]["similarity"]:
                horse_scores[key] = r

        predictions = []
        for r in sorted(horse_scores.values(), key=lambda x: x["similarity"], reverse=True):
            if r["similarity"] >= confidence_threshold:
                predictions.append({
                    "horse_id": r["horse_id"],
                    "horse_name": r["horse_name"],
                    "herd_name": r["herd_name"],
                    "similarity": float(r["similarity"]),
                    "reference_photo_id": r["photo_id"],
                })

        return {
            "predictions": predictions,
            "query_embedding": embedding.tolist(),
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
