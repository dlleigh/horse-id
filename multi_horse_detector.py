import os
import pandas as pd
import torch
import yaml
from tqdm import tqdm

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


def count_horses(image_path, model):
    """
    Analyzes an image to determine the number of horses detected.
    Returns "NONE", "SINGLE", or "MULTIPLE".
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}, marking as NONE.")
        return "NONE"

    try:
        results = model(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "NONE"

    # COCO class for 'horse' is 17
    horse_boxes = [box for box in results.xyxy[0] if int(box[5]) == 17]

    if len(horse_boxes) == 0:
        return "NONE"
    elif len(horse_boxes) == 1:
        return "SINGLE"
    else:
        # Simple check for multiple significant detections
        return "MULTIPLE"

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

    # Use .astype(str) to avoid issues with mixed types (e.g., float NaNs)
    manifest_df['num_horses_detected'] = manifest_df['num_horses_detected'].astype(str)


    print("Analyzing images for horse detection...")
    detection_results = []
    for _, row in tqdm(manifest_df.iterrows(), total=manifest_df.shape[0], desc="Processing images"):
        # Only process images that haven't been labeled yet
        if pd.isna(row['num_horses_detected']) or row['num_horses_detected'] in ['', 'nan']:
            image_path = os.path.join(IMAGE_DIR, row['filename'])
            result = count_horses(image_path, model)
            detection_results.append(result)
        else:
            detection_results.append(row['num_horses_detected'])

    manifest_df['num_horses_detected'] = detection_results

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
    print("-------------------------")


if __name__ == '__main__':
    main()