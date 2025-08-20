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
# It reads the normalized manifest and writes to the detected_manifest_file
INPUT_MANIFEST_FILE = config['paths']['normalized_manifest_file'].format(data_root=DATA_ROOT)
OUTPUT_MANIFEST_FILE = config['paths']['detected_manifest_file'].format(data_root=DATA_ROOT)
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
    Returns only the horse detection status: "NONE", "SINGLE", or "MULTIPLE".
    
    Note: Segmentation and bounding box data is no longer returned as it was found
    to hurt performance in README_MASKING_ANALYSIS.md and is not used for reprocessing.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}, marking as NONE.")
        return "NONE"

    try:
        # verbose=False suppresses the console output from YOLO
        results = model(image_path, conf=confidence_threshold, verbose=False)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "NONE"

    # Process results
    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return "NONE"

    # In COCO dataset, the class ID for 'horse' is 17
    horse_indices = np.where(result.boxes.cls.cpu().numpy() == 17)[0]

    if len(horse_indices) == 0:
        return "NONE"

    # Get image dimensions
    img_height, img_width = result.orig_shape
    
    # Filter detections to only include horses
    horse_boxes = result.boxes[horse_indices]
    horse_masks = result.masks[horse_indices]

    # Calculate areas from bounding boxes (xywh format is convenient here)
    areas = (horse_boxes.xywh[:, 2] * horse_boxes.xywh[:, 3]).cpu().numpy()
    
    # Use shared library for classification - only need the classification result
    classification, _, _, _ = classify_horse_detection(
        horse_boxes, horse_masks, horse_indices, areas, img_width, img_height
    )
    
    return classification

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

            # Merge previous results into the current dataframe based on core identification columns
            # This preserves existing data and allows us to only process new/missing rows
            # Use only core columns that should be present in both files to avoid column mismatch errors
            core_merge_columns = ['canonical_id', 'original_canonical_id', 'filename', 'message_id']
            
            # Only use columns that actually exist in both dataframes
            available_merge_columns = [col for col in core_merge_columns 
                                     if col in base_manifest_df.columns and col in previous_detections_df.columns]
            
            if available_merge_columns:
                # Select detection-specific column from previous detections
                detection_columns = ['num_horses_detected']
                columns_to_keep = available_merge_columns + [col for col in detection_columns 
                                                           if col in previous_detections_df.columns]
                
                # Create a subset of previous detections with only the columns we need
                prev_detections_subset = previous_detections_df[columns_to_keep]
                
                # Use combine_first to preserve existing data in base_manifest and only fill missing values
                # First, set index to merge columns for proper alignment
                base_indexed = output_df.set_index(available_merge_columns)
                prev_indexed = prev_detections_subset.set_index(available_merge_columns)
                
                # Use combine_first to preserve base data and only add missing detection data
                combined_indexed = base_indexed.combine_first(prev_indexed)
                
                # Reset index back to regular dataframe
                output_df = combined_indexed.reset_index()
            else:
                print("Warning: No common merge columns found. Skipping merge with previous detections.")

    # Initialize new column if it doesn't exist
    if 'num_horses_detected' not in output_df.columns:
        output_df['num_horses_detected'] = ''

    print("Analyzing images for horse detection...")
    for index, row in tqdm(output_df.iterrows(), total=output_df.shape[0], desc="Processing images"):
        # Only process if the image hasn't been analyzed yet
        # Check for valid detection results in num_horses_detected (NONE, SINGLE, MULTIPLE)
        current_detection = row.get('num_horses_detected', '')
        
        # Process if no valid detection result exists (empty, NaN, or "nan" string)
        if (pd.isna(current_detection) or 
            not current_detection or 
            str(current_detection).strip() == '' or 
            str(current_detection).lower() == 'nan' or
            current_detection not in ['NONE', 'SINGLE', 'MULTIPLE']):
            image_path = os.path.join(IMAGE_DIR, row['filename'])
            status = count_horses(image_path, model, CONFIDENCE_THRESHOLD)
            output_df.loc[index, 'num_horses_detected'] = status

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
    
    # Summary statistics (size_ratio no longer available)
    if 'MULTIPLE' in counts:
        print(f"  - Multiple horse detections: {counts['MULTIPLE']}")
    print("-------------------------")

if __name__ == '__main__':
    main()