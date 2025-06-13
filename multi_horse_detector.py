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

    # INPUT_MANIFEST_FILE is config['paths']['manifest_file'] (from parser)
    # OUTPUT_MANIFEST_FILE is config['detection']['detected_manifest_file'] (this script's output)
    print(f"Reading base manifest file: {INPUT_MANIFEST_FILE}")
    try:
        base_manifest_df = pd.read_csv(INPUT_MANIFEST_FILE, dtype={'filename': str})
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {INPUT_MANIFEST_FILE}")
        return

    output_df = base_manifest_df.copy()

    # Initialize detection columns if they don't exist (e.g., from parser's output)
    if 'num_horses_detected' not in output_df.columns:
        output_df['num_horses_detected'] = ''
    if 'size_ratio' not in output_df.columns:
        output_df['size_ratio'] = pd.NA

    # Ensure correct types before potentially filling from previous run
    output_df['num_horses_detected'] = output_df['num_horses_detected'].astype(str)
    output_df['size_ratio'] = pd.to_numeric(output_df['size_ratio'], errors='coerce')

    # Load previous detection results if the output file exists
    if os.path.exists(OUTPUT_MANIFEST_FILE):
        print(f"Loading previous detections from: {OUTPUT_MANIFEST_FILE}")
        previous_detections_df = pd.read_csv(OUTPUT_MANIFEST_FILE, dtype={'filename': str})

        if not previous_detections_df.empty:
            # Create a map for efficient lookup of previous results
            previous_detections_map = {}
            prev_num_horses_col = 'num_horses_detected' if 'num_horses_detected' in previous_detections_df.columns else None
            prev_size_ratio_col = 'size_ratio' if 'size_ratio' in previous_detections_df.columns else None

            for _, prev_row in previous_detections_df.iterrows():
                num_horses = str(prev_row[prev_num_horses_col]) if prev_num_horses_col and pd.notna(prev_row[prev_num_horses_col]) else ''
                size_ratio = pd.to_numeric(prev_row[prev_size_ratio_col], errors='coerce') if prev_size_ratio_col and pd.notna(prev_row[prev_size_ratio_col]) else pd.NA
                if prev_row['filename']: # Ensure filename is not None
                    previous_detections_map[prev_row['filename']] = (num_horses, size_ratio)
            
            # Apply previous detections to output_df
            def apply_previous_detections(row):
                if row['filename'] in previous_detections_map:
                    prev_detection, prev_ratio = previous_detections_map[row['filename']]
                    # Update if current is empty/default or if previous had a more specific value
                    if not str(row['num_horses_detected']).strip() or str(row['num_horses_detected']).lower() in ['nan', '<na>']:
                        row['num_horses_detected'] = prev_detection
                    if pd.isna(row['size_ratio']):
                         row['size_ratio'] = prev_ratio
                return row
            output_df = output_df.apply(apply_previous_detections, axis=1)


    print("Analyzing images for horse detection...")
    for index, row in tqdm(output_df.iterrows(), total=output_df.shape[0], desc="Processing images"):
        current_detection_status = str(row['num_horses_detected']).strip().lower()
        if not current_detection_status or current_detection_status in ['nan', '<na>']: # Process if empty, 'nan', or '<NA>'
            image_path = os.path.join(IMAGE_DIR, row['filename'])
            status, ratio = count_horses(image_path, model)
            output_df.loc[index, 'num_horses_detected'] = status
            output_df.loc[index, 'size_ratio'] = ratio

    # Save the updated DataFrame to the new CSV file
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST_FILE), exist_ok=True)
    print(f"Saving updated manifest to: {OUTPUT_MANIFEST_FILE}")
    output_df.to_csv(OUTPUT_MANIFEST_FILE, index=False)

    # Display a summary
    counts = output_df['num_horses_detected'].value_counts()
    print("\n--- Analysis Complete ---")
    print(f"Total images in manifest: {len(output_df)}")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    print(f"  Average size_ratio (for MULTIPLE actual detections): {output_df[output_df['num_horses_detected'] == 'MULTIPLE']['size_ratio'].mean():.2f}")

    print("-------------------------")


if __name__ == '__main__':
    main()