import os
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO
import json

# Import shared detection library
from horse_detection_lib import classify_horse_detection

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

# Import individual functions from shared library for backward compatibility
from horse_detection_lib import (
    calculate_bbox_overlap,
    calculate_distance_from_center,
    analyze_depth_relationships,
    is_edge_cropped,
    identify_subject_horse
)

def count_horses(image_path, model, confidence_threshold):
    """
    Analyzes an image to determine the number of horses detected using improved logic.
    Returns a tuple: (status, ratio, largest_bbox, largest_mask_str).
    - status: "NONE", "SINGLE", or "MULTIPLE".
    - ratio: Ratio of the largest horse area to the next largest, or NaN.
    - largest_bbox: Pixel coordinates [x, y, width, height] of the largest horse, or None.
    - largest_mask_str: Custom string representation of the largest horse's segmentation mask polygon, or None.
    
    Improved logic considers:
    1. Area ratios (original logic)
    2. Overlap/occlusion analysis
    3. Position relative to image center
    """
    default_return = "NONE", float('nan'), None, None
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}, marking as NONE.")
        return default_return

    try:
        # verbose=False suppresses the console output from YOLO
        results = model(image_path, conf=confidence_threshold, verbose=False)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return default_return

    # Process results
    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return default_return

    # In COCO dataset, the class ID for 'horse' is 17
    horse_indices = np.where(result.boxes.cls.cpu().numpy() == 17)[0]

    if len(horse_indices) == 0:
        return default_return

    # Get image dimensions for pixel coordinate conversion
    img_height, img_width = result.orig_shape
    
    # Filter detections to only include horses
    horse_boxes = result.boxes[horse_indices]
    horse_masks = result.masks[horse_indices]

    # Calculate areas from bounding boxes (xywh format is convenient here)
    areas = (horse_boxes.xywh[:, 2] * horse_boxes.xywh[:, 3]).cpu().numpy()
    
    # Use depth-aware subject identification instead of just largest area
    subject_idx_in_horses = identify_subject_horse(horse_boxes, horse_masks, horse_indices, areas, img_width, img_height)
    subject_original_idx = horse_indices[subject_idx_in_horses]
    
    # Get bounding box in normalized xywh format (x_center, y_center, width, height)
    xywhn = result.boxes.xywhn[subject_original_idx].cpu().numpy().flatten()
    x_center, y_center, width, height = xywhn

    # Convert normalized coordinates to pixel coordinates
    # Convert center x,y to top-left x,y and scale to pixel coordinates
    x_pixel = int((x_center - (width / 2)) * img_width)
    y_pixel = int((y_center - (height / 2)) * img_height)
    width_pixel = int(width * img_width)
    height_pixel = int(height * img_height)
    
    subject_bbox_xywh = [x_pixel, y_pixel, width_pixel, height_pixel]
    subject_mask_xy = horse_masks.xy[subject_idx_in_horses]
    # Create a comma-free string format to avoid quoting issues in CSV.
    # Format: "x1 y1;x2 y2;..." for a single polygon.
    points_as_strings = [f"{point[0]} {point[1]}" for point in subject_mask_xy.tolist()]
    subject_mask_str = ";".join(points_as_strings)

    # Use shared library for classification
    classification, calculated_size_ratio, subject_idx_final, analysis = classify_horse_detection(
        horse_boxes, horse_masks, horse_indices, areas, img_width, img_height
    )
    
    # Ensure we're using the correct subject horse for bbox/mask output
    if classification == "SINGLE" and len(horse_indices) == 1:
        # Single horse case - bbox and mask already set correctly above
        return classification, calculated_size_ratio, subject_bbox_xywh, subject_mask_str
    else:
        # Multiple horse case - use the subject identified by the library
        subject_idx_in_horses = subject_idx_final
        subject_original_idx = horse_indices[subject_idx_in_horses]
        
        # Get bounding box in normalized xywh format (x_center, y_center, width, height)
        xywhn = result.boxes.xywhn[subject_original_idx].cpu().numpy().flatten()
        x_center, y_center, width, height = xywhn

        # Convert normalized coordinates to pixel coordinates
        x_pixel = int((x_center - (width / 2)) * img_width)
        y_pixel = int((y_center - (height / 2)) * img_height)
        width_pixel = int(width * img_width)
        height_pixel = int(height * img_height)
        
        subject_bbox_xywh = [x_pixel, y_pixel, width_pixel, height_pixel]
        subject_mask_xy = horse_masks.xy[subject_idx_in_horses]
        points_as_strings = [f"{point[0]} {point[1]}" for point in subject_mask_xy.tolist()]
        subject_mask_str = ";".join(points_as_strings)
        
        return classification, calculated_size_ratio, subject_bbox_xywh, subject_mask_str

def main():
    """Updates the manifest file with horse detection information."""
    print(f"Loading YOLOv8-Seg model: {YOLO_MODEL}...")
    model = YOLO(YOLO_MODEL)
    print(f"Model '{YOLO_MODEL}' loaded successfully.")

    print(f"Reading base manifest file: {INPUT_MANIFEST_FILE}")
    try:
        base_manifest_df = pd.read_csv(INPUT_MANIFEST_FILE, dtype={'filename': str})
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {INPUT_MANIFEST_FILE}")
        return

    output_df = base_manifest_df.copy()

    # Load previous detection results if the output file exists
    if os.path.exists(OUTPUT_MANIFEST_FILE):
        print(f"Loading previous detections from: {OUTPUT_MANIFEST_FILE}")
        previous_detections_df = pd.read_csv(OUTPUT_MANIFEST_FILE, dtype={'filename': str})

        if not previous_detections_df.empty:
            # --- FIX for dtype mismatch ---
            # Ensure the 'num_horses_detected' column is treated as a string in both
            # DataFrames to prevent a dtype mismatch during the merge.
            if 'num_horses_detected' in output_df.columns:
                output_df['num_horses_detected'] = output_df['num_horses_detected'].fillna('').astype(str)
            if 'num_horses_detected' in previous_detections_df.columns:
                previous_detections_df['num_horses_detected'] = previous_detections_df['num_horses_detected'].fillna('').astype(str)

            # Merge previous results into the current dataframe based on filename
            # This preserves existing data and allows us to only process new/missing rows
            output_df = pd.merge(output_df, previous_detections_df, on=list(base_manifest_df.columns), how='left')

    # Initialize new columns if they don't exist
    new_cols = {
        'num_horses_detected': '',
        'size_ratio': pd.NA,
        'bbox_x': pd.NA,
        'bbox_y': pd.NA,
        'bbox_width': pd.NA,
        'bbox_height': pd.NA,
        'segmentation_mask': ''
    }
    for col, default_val in new_cols.items():
        if col not in output_df.columns:
            output_df[col] = default_val

    # Ensure correct types for columns
    output_df['size_ratio'] = pd.to_numeric(output_df['size_ratio'], errors='coerce')
    for col in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']:
        output_df[col] = pd.to_numeric(output_df[col], errors='coerce')

    print("Analyzing images for horse detection...")
    for index, row in tqdm(output_df.iterrows(), total=output_df.shape[0], desc="Processing images"):
        # Process if 'segmentation_mask' is missing, which indicates a new or unprocessed image
        if pd.isna(row.get('segmentation_mask')) or not row.get('segmentation_mask'):
            image_path = os.path.join(IMAGE_DIR, row['filename'])
            status, ratio, bbox, mask = count_horses(image_path, model, CONFIDENCE_THRESHOLD)

            output_df.loc[index, 'num_horses_detected'] = status
            output_df.loc[index, 'size_ratio'] = ratio
            output_df.loc[index, 'segmentation_mask'] = mask

            if bbox:
                output_df.loc[index, 'bbox_x'] = bbox[0]
                output_df.loc[index, 'bbox_y'] = bbox[1]
                output_df.loc[index, 'bbox_width'] = bbox[2]
                output_df.loc[index, 'bbox_height'] = bbox[3]

    # Save the updated DataFrame to the new CSV file
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST_FILE), exist_ok=True)
    print(f"Saving updated manifest to: {OUTPUT_MANIFEST_FILE}")
    output_df.to_csv(OUTPUT_MANIFEST_FILE, index=False)

    # Display a summary
    counts = output_df['num_horses_detected'].value_counts()
    print("\n--- Analysis Complete ---")
    print(f"Total images in manifest: {len(output_df)}")
    for status, count in counts.items():
        print(f"  - {status}: {count}")
    
    multiple_horse_df = output_df[output_df['num_horses_detected'] == 'MULTIPLE']
    if not multiple_horse_df.empty:
        print(f"  - Average size_ratio (for MULTIPLE detections): {multiple_horse_df['size_ratio'].mean():.2f}")
    print("-------------------------")

if __name__ == '__main__':
    main()