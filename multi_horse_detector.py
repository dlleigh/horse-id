import os
import pandas as pd
import torch
import cv2
import warnings
from tqdm import tqdm
from pathlib import Path
import math
from enum import Enum

#Filter out the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message='`torch.cuda.amp.autocast.*')

class HorseDetection(str, Enum):
    NONE = "NONE"
    SINGLE = "SINGLE" 
    MULTIPLE = "MULTIPLE"

def is_image_ambiguous(image_path, model):
    """
    Analyzes an image to determine number of horses detected.

    Args:
        image_path (str): The full path to the image file.
        model: The loaded YOLOv5 model.

    Returns:
        HorseDetection: NONE if no horses, SINGLE if one horse, MULTIPLE if multiple horses
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return HorseDetection.NONE

    # Run inference
    try:
        results = model(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return HorseDetection.NONE

    # Get bounding box information for horses (class 17 in COCO)
    horse_boxes = [box for box in results.xyxy[0] if int(box[5]) == 17]

    if len(horse_boxes) == 0:
        return HorseDetection.NONE
    elif len(horse_boxes) == 1:
        return HorseDetection.SINGLE
    else:
        # Sort boxes by area (width * height) in descending order
        horse_boxes.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

        # Calculate areas
        largest_area = (horse_boxes[0][2] - horse_boxes[0][0]) * (horse_boxes[0][3] - horse_boxes[0][1])
        second_largest_area = (horse_boxes[1][2] - horse_boxes[1][0]) * (horse_boxes[1][3] - horse_boxes[1][1])

        # If second horse is more than 50% of largest, mark as multiple
        if second_largest_area / largest_area > 0.5:
            return HorseDetection.MULTIPLE
        return HorseDetection.SINGLE  # Second horse too small to count

def create_html_report(manifest_df, image_dir, detection_type, output_path):
    """
    Creates an HTML file with a grid of images.
    
    Args:
        manifest_df (pd.DataFrame): DataFrame containing image information
        image_dir (str): Directory containing the images
        detection_type (HorseDetection): Type of images to include
        output_path (str): Path to save the HTML file
    """
    # Filter images based on horse_detection
    filtered_df = manifest_df[manifest_df['horse_detection'] == detection_type]
    
    # Calculate grid dimensions
    images_per_row = 4
    total_images = len(filtered_df)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{detection_type.value} Horse Images Report</title>
        <style>
            .image-grid {{
                display: grid;
                grid-template-columns: repeat({images_per_row}, 1fr);
                gap: 10px;
                padding: 10px;
            }}
            .image-container {{
                position: relative;
            }}
            .image-container img {{
                width: 100%;
                height: auto;
            }}
            .image-caption {{
                font-size: 12px;
                text-align: center;
                padding: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>{detection_type.value} Horse Images ({total_images} images)</h1>
        <div class="image-grid">
    """
    
    # Add images to the grid
    for _, row in filtered_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        rel_path = os.path.relpath(image_path, os.path.dirname(output_path))
        html_content += f"""
            <div class="image-container">
                <img src="{rel_path}" alt="{row['filename']}">
                <div class="image-caption">{row['filename']}</div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    """
    Main function to update the manifest file with ambiguity information.
    """
    # --- Configuration ---
    # Update these paths to match your project structure
    data_root = os.path.join(os.environ.get('HOME'), 'google-drive/horseID Project/data')
    image_dir = os.path.join(data_root, 'horse_photos')
    manifest_file = os.path.join(data_root, 'horse_photos_manifest.csv')
    output_manifest_file = 'horse_photos_manifest_multi_horse_detected.csv'
    # -------------------

    print("Loading YOLOv5 model...")
    # Load a pre-trained YOLOv5 model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    print("Model loaded successfully.")

    model.conf = 0.1

    print(f"Reading manifest file: {manifest_file}")
    try:
        manifest_df = pd.read_csv(manifest_file)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_file}")
        return

    # Create a new column to mark ambiguous images
    print("Analyzing images for ambiguity...")
    # Create a list to store results
    horse_detection_results = []
    
    # Process images with progress bar
    for filename in tqdm(manifest_df['filename'], desc="Processing images"):
        result = is_image_ambiguous(os.path.join(image_dir, filename), model)
        horse_detection_results.append(result)
    
    # Assign results to DataFrame
    manifest_df['horse_detection'] = horse_detection_results
    
    # Save the updated DataFrame to a new CSV file
    print(f"Saving updated manifest to: {output_manifest_file}")
    manifest_df.to_csv(output_manifest_file, index=False)

    # Display a summary
    none_count = (manifest_df['horse_detection'] == HorseDetection.NONE).sum()
    single_count = (manifest_df['horse_detection'] == HorseDetection.SINGLE).sum()
    multiple_count = (manifest_df['horse_detection'] == HorseDetection.MULTIPLE).sum()
    total_images = len(manifest_df)
    
    print("\n--- Analysis Complete ---")
    print(f"Total images processed: {total_images}")
    print(f"No horses detected: {none_count} ({none_count/total_images:.1%})")
    print(f"Single horse detected: {single_count} ({single_count/total_images:.1%})")
    print(f"Multiple horses detected: {multiple_count} ({multiple_count/total_images:.1%})")
    print("-------------------------")

    # Generate HTML reports
    print("\nGenerating HTML reports...")
    output_dir = os.path.dirname(output_manifest_file)
    
    # Create HTML report for each detection type
    for detection_type in HorseDetection:
        report_path = os.path.join(output_dir, f'{detection_type.value.lower()}_horse_report.html')
        create_html_report(manifest_df, image_dir, detection_type, report_path)
        print(f"{detection_type.value} report saved to: {report_path}")

if __name__ == '__main__':
    main()