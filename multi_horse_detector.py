import os
import pandas as pd
import torch
from torch import amp  # Add this import
import yaml
from tqdm import tqdm

import warnings # Import the warnings library

# Suppress the specific FutureWarning from PyTorch's amp module
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Load Configuration ---
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# --- Use the config values ---
DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
IMAGE_DIR = config['paths']['dataset_dir'].format(data_root=DATA_ROOT)
# It reads the base manifest and writes to the detected_manifest_file
INPUT_MANIFEST_FILE = config['paths']['manifest_file'].format(data_root=DATA_ROOT)
OUTPUT_MANIFEST_FILE = config['detection']['detected_manifest_file'].format(data_root=DATA_ROOT)
YOLO_MODEL = config['detection']['yolo_model']
CONFIDENCE_THRESHOLD = config['detection']['confidence_threshold']
SIZE_RATIO = config['detection']['size_ratio_for_single_horse']


def count_horses(image_path, model):
    """
    Analyzes an image to determine the number of horses detected.
    Returns a tuple: (detection_status, size_ratio).
    detection_status is "NONE", "SINGLE", or "MULTIPLE".
    size_ratio is the ratio of the largest_area to the next_largest_area, or NaN.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}, marking as NONE.")
        return "NONE", float('nan')

    try:
        results = model(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "NONE", float('nan')

    # COCO class for 'horse' is 17
    horse_boxes = [box for box in results.xyxy[0] if int(box[5]) == 17]

    if len(horse_boxes) == 0:
        return "NONE", float('nan')
    elif len(horse_boxes) == 1:
        return "SINGLE", float('nan') # No ratio to calculate for a single horse
    else:
        # Multiple horses detected, check size ratio
        detection_status = "MULTIPLE" # Default for multiple horses
        areas = []
        for box in horse_boxes:
            # box format is [xmin, ymin, xmax, ymax, confidence, class]
            width = (box[2] - box[0]).item()  # Extract Python number
            height = (box[3] - box[1]).item() # Extract Python number
            areas.append(width * height)

        if not areas: # Should not happen if horse_boxes is not empty
            return "MULTIPLE", float('nan')

        largest_area = max(areas)
        other_areas = [area for area in areas if area < largest_area]
        
        calculated_size_ratio = float('nan')
        if other_areas:
            next_largest_area = max(other_areas)
            if next_largest_area > 0: # Avoid division by zero
                calculated_size_ratio = largest_area / next_largest_area

        if not other_areas: # Only one "largest" horse, or all are same size
            # If all areas are the same, there's no "next largest" distinct area.
            # The status remains MULTIPLE unless the size_ratio logic below changes it.
            pass

        # Check if the largest horse is SIZE_RATIO times larger than ALL other horses
        # This logic determines if it should be re-classified as SINGLE
        if all(largest_area >= SIZE_RATIO * other_area for other_area in other_areas):
            detection_status = "SINGLE"
        
        return detection_status, calculated_size_ratio

def main():
    """Updates the manifest file with horse detection information."""
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL, pretrained=True)
    model.conf = CONFIDENCE_THRESHOLD
    print(f"Model '{YOLO_MODEL}' loaded successfully.")

    print(f"Reading manifest file: {INPUT_MANIFEST_FILE}")
    try:
        manifest_df = pd.read_csv(INPUT_MANIFEST_FILE)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {INPUT_MANIFEST_FILE}")
        return

    # Add column if it doesn't exist
    if 'num_horses_detected' not in manifest_df.columns:
        manifest_df['num_horses_detected'] = ''
    if 'size_ratio' not in manifest_df.columns:
        manifest_df['size_ratio'] = pd.NA # Use pandas NA for float columns

    # Use .astype(str) to avoid issues with mixed types (e.g., float NaNs)
    manifest_df['num_horses_detected'] = manifest_df['num_horses_detected'].astype(str)
    manifest_df['size_ratio'] = pd.to_numeric(manifest_df['size_ratio'], errors='coerce')


    print("Analyzing images for horse detection...")
    detection_results = []
    size_ratio_results = []

    for _, row in tqdm(manifest_df.iterrows(), total=manifest_df.shape[0], desc="Processing images"):
        # Only process images that haven't been labeled yet
        if pd.isna(row['num_horses_detected']) or row['num_horses_detected'] in ['', 'nan']:
            image_path = os.path.join(IMAGE_DIR, row['filename'])
            status, ratio = count_horses(image_path, model)
            detection_results.append(status)
            size_ratio_results.append(ratio)
        else:
            detection_results.append(row['num_horses_detected'])
            size_ratio_results.append(row['size_ratio']) # Keep existing value if already processed

    manifest_df['num_horses_detected'] = detection_results
    manifest_df['size_ratio'] = size_ratio_results

    # Save the updated DataFrame to the new CSV file
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST_FILE), exist_ok=True)
    print(f"Saving updated manifest to: {OUTPUT_MANIFEST_FILE}")
    manifest_df.to_csv(OUTPUT_MANIFEST_FILE, index=False)

    # Display a summary
    counts = manifest_df['num_horses_detected'].value_counts()
    print("\n--- Analysis Complete ---")
    print(f"Total images processed: {len(manifest_df)}")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    print(f"  Average size_ratio (for MULTIPLE actual detections): {manifest_df[manifest_df['num_horses_detected'] == 'MULTIPLE']['size_ratio'].mean():.2f}")

    print("-------------------------")


if __name__ == '__main__':
    main()