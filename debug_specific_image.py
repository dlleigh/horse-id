#!/usr/bin/env python3
"""
Debug a specific image detection to understand the classification logic.
"""

import os
import numpy as np
import yaml
from ultralytics import YOLO
import sys

# Load configuration
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATA_ROOT = os.path.expanduser(config['paths']['data_root'])
IMAGE_DIR = config['paths']['dataset_dir'].format(data_root=DATA_ROOT)
YOLO_MODEL = config['detection']['yolo_model']
CONFIDENCE_THRESHOLD = config['detection']['confidence_threshold']
SIZE_RATIO = config['detection']['size_ratio_for_single_horse']
CLASSIFICATION_CONFIG = config['detection']['classification']

# Import functions from the main script
from multi_horse_detector import (
    calculate_bbox_overlap, 
    calculate_distance_from_center,
    analyze_depth_relationships,
    identify_subject_horse,
    is_edge_cropped
)

def debug_image(filename):
    """Debug detection for a specific image file."""
    image_path = os.path.join(IMAGE_DIR, filename)
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    print(f"=== DEBUGGING: {filename} ===")
    
    # Load model and run detection
    model = YOLO(YOLO_MODEL)
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]
    
    if result.masks is None or len(result.masks) == 0:
        print("No detections with masks found")
        return
    
    # Get horse detections
    horse_indices = np.where(result.boxes.cls.cpu().numpy() == 17)[0]
    print(f"Number of horses detected: {len(horse_indices)}")
    
    if len(horse_indices) <= 1:
        print(f"Only {len(horse_indices)} horse(s) detected - would be SINGLE")
        return
    
    # Get image dimensions
    img_height, img_width = result.orig_shape
    print(f"Image dimensions: {img_width} x {img_height}")
    
    # Filter detections
    horse_boxes = result.boxes[horse_indices]
    horse_masks = result.masks[horse_indices]
    areas = (horse_boxes.xywh[:, 2] * horse_boxes.xywh[:, 3]).cpu().numpy()
    
    print(f"\nHorse areas: {areas}")
    print(f"Area ratios: {[areas[0]/areas[i] if i > 0 else 1.0 for i in range(len(areas))]}")
    
    # Analyze each horse
    for i in range(len(horse_indices)):
        bbox_xyxy = horse_boxes.xyxy[i].cpu().numpy()
        bbox_xywh = horse_boxes.xywh[i].cpu().numpy()
        
        print(f"\nHorse {i+1}:")
        print(f"  Bbox (xyxy): {bbox_xyxy}")
        print(f"  Bbox (xywh): {bbox_xywh}")
        print(f"  Area: {areas[i]:.2f}")
        print(f"  Area % of image: {(areas[i] / (img_width * img_height)) * 100:.1f}%")
        
        # Position in frame
        x_center_norm = bbox_xywh[0] / img_width
        y_center_norm = bbox_xywh[1] / img_height
        print(f"  Position (normalized): x={x_center_norm:.3f}, y={y_center_norm:.3f}")
        
        # Edge cropping analysis
        is_cropped, severity = is_edge_cropped(bbox_xyxy, img_width, img_height)
        print(f"  Edge cropped: {is_cropped} (severity: {severity:.3f})")
        
        # Edge distances
        left_margin = bbox_xyxy[0]
        top_margin = bbox_xyxy[1]
        right_margin = img_width - bbox_xyxy[2]
        bottom_margin = img_height - bbox_xyxy[3]
        print(f"  Edge margins: L={left_margin:.1f}, T={top_margin:.1f}, R={right_margin:.1f}, B={bottom_margin:.1f}")
    
    # Subject horse identification
    subject_idx = identify_subject_horse(horse_boxes, horse_masks, horse_indices, areas, img_width, img_height)
    print(f"\nIdentified subject horse: Horse {subject_idx + 1}")
    
    # Depth analysis
    depth_scores = analyze_depth_relationships(horse_boxes, horse_masks, horse_indices, img_width, img_height)
    print(f"\nDepth scores: {depth_scores}")
    for i, score in enumerate(depth_scores):
        print(f"  Horse {i+1}: {score:.3f}")
    
    # Classification criteria
    subject_area = areas[subject_idx]
    other_areas = np.delete(areas, subject_idx)
    subject_depth_score = depth_scores[subject_idx]
    other_depth_scores = np.delete(depth_scores, subject_idx)
    
    print(f"\n=== CLASSIFICATION CRITERIA ===")
    
    # Size dominance
    max_other_area = max(other_areas)
    size_dominance = subject_area >= SIZE_RATIO * max_other_area
    size_ratio = subject_area / max_other_area
    print(f"Size dominance: {size_dominance}")
    print(f"  Subject area: {subject_area:.2f}")
    print(f"  Max other area: {max_other_area:.2f}")
    print(f"  Size ratio: {size_ratio:.2f} (need >= {SIZE_RATIO})")
    
    # Depth dominance
    max_other_depth = max(other_depth_scores)
    depth_dominance = subject_depth_score > max_other_depth + CLASSIFICATION_CONFIG['depth_dominance_threshold']
    print(f"Depth dominance: {depth_dominance}")
    print(f"  Subject depth: {subject_depth_score:.3f}")
    print(f"  Max other depth: {max_other_depth:.3f}")
    print(f"  Difference: {subject_depth_score - max_other_depth:.3f} (need > {CLASSIFICATION_CONFIG['depth_dominance_threshold']})")
    
    # Strong size dominance
    strong_size_dominance = size_ratio >= CLASSIFICATION_CONFIG['strong_size_dominance_threshold']
    print(f"Strong size dominance: {strong_size_dominance}")
    print(f"  Size ratio: {size_ratio:.2f} (need >= {CLASSIFICATION_CONFIG['strong_size_dominance_threshold']})")
    
    # Extreme occlusion
    subject_bbox_xyxy = horse_boxes.xyxy[subject_idx].cpu().numpy()
    high_overlap_count = 0
    
    print(f"Extreme occlusion analysis:")
    for i, other_idx in enumerate([idx for j, idx in enumerate(range(len(horse_indices))) if j != subject_idx]):
        other_bbox_xyxy = horse_boxes.xyxy[other_idx].cpu().numpy()
        overlap_ratio = calculate_bbox_overlap(subject_bbox_xyxy, other_bbox_xyxy)
        print(f"  Horse {other_idx+1} overlap with subject: {overlap_ratio:.3f}")
        
        if overlap_ratio > 0.7:
            high_overlap_count += 1
    
    extreme_occlusion = high_overlap_count == len(other_areas)
    print(f"Extreme occlusion: {extreme_occlusion}")
    print(f"  High overlap count: {high_overlap_count}/{len(other_areas)}")
    
    # Final decision
    condition1 = size_dominance and depth_dominance
    condition2 = strong_size_dominance
    condition3 = extreme_occlusion
    
    should_classify_as_single = condition1 or condition2 or condition3
    final_classification = "SINGLE" if should_classify_as_single else "MULTIPLE"
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Classification: {final_classification}")
    print(f"Condition 1 (size AND depth): {condition1}")
    print(f"Condition 2 (strong size): {condition2}")
    print(f"Condition 3 (extreme occlusion): {condition3}")
    print(f"Final result: {should_classify_as_single}")
    
    if not should_classify_as_single:
        print(f"\n=== WHY MULTIPLE? ===")
        print(f"• Size dominance failed: {size_ratio:.2f} < {SIZE_RATIO}")
        print(f"• Strong size dominance failed: {size_ratio:.2f} < {CLASSIFICATION_CONFIG['strong_size_dominance_threshold']}")
        print(f"• Depth dominance failed: {subject_depth_score - max_other_depth:.3f} ≤ {CLASSIFICATION_CONFIG['depth_dominance_threshold']}")
        print(f"• Extreme occlusion failed: {high_overlap_count}/{len(other_areas)} horses highly overlapped")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_specific_image.py <filename>")
        print("Example: python debug_specific_image.py 1971c6c807b75195-IMG_6284.jpg")
        sys.exit(1)
    
    debug_image(sys.argv[1])