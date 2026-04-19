#!/usr/bin/env python3
"""
Interactive tool for tuning multi-horse detection parameters.
Run with: streamlit run tune_detection_parameters.py
"""

import streamlit as st
import os
import numpy as np
import yaml
import warnings
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
import urllib.parse
import sys

# Suppress PyTorch/Streamlit watcher warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import shared detection library
from horse_detection_lib import (
    analyze_depth_relationships,
    identify_subject_horse,
    is_edge_cropped,
    calculate_bbox_overlap,
    classify_horse_detection,
    analyze_classification_criteria
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model():
    config = load_config()
    return YOLO(config['detection']['yolo_model'])

def parse_file_url(file_url):
    """Parse a file:// URL to extract the local file path."""
    if file_url.startswith('file://'):
        # Remove 'file://' prefix and decode URL encoding
        path = urllib.parse.unquote(file_url[7:])
        return path
    return file_url

def analyze_image(image_path, config_params):
    """Analyze an image with the given configuration parameters."""
    if not os.path.exists(image_path):
        return None, f"Image not found: {image_path}"
    
    model = load_model()
    
    try:
        results = model(image_path, conf=config_params['confidence_threshold'], verbose=False)
    except Exception as e:
        return None, f"Error processing image: {e}"
    
    result = results[0]
    
    if result.masks is None or len(result.masks) == 0:
        return None, "No detections with masks found"
    
    # Get horse detections (class 17 in COCO)
    horse_indices = np.where(result.boxes.cls.cpu().numpy() == 17)[0]
    
    if len(horse_indices) == 0:
        return None, "No horses detected"
    
    if len(horse_indices) == 1:
        return {
            'num_horses': 1,
            'classification': 'SINGLE',
            'reason': 'Only one horse detected',
            'image_path': image_path,
            'result': result,
            'horse_indices': horse_indices
        }, None
    
    # Get image dimensions
    img_height, img_width = result.orig_shape
    
    # Filter detections
    horse_boxes = result.boxes[horse_indices]
    horse_masks = result.masks[horse_indices]
    areas = (horse_boxes.xywh[:, 2] * horse_boxes.xywh[:, 3]).cpu().numpy()
    
    # Use shared library for classification with custom parameters
    classification, calculated_size_ratio, subject_idx, analysis = classify_horse_detection(
        horse_boxes, horse_masks, horse_indices, areas, img_width, img_height, config_params
    )
    
    return {
        'num_horses': len(horse_indices),
        'classification': analysis['final_classification'],
        'reason': analysis['reason'],
        'image_path': image_path,
        'result': result,
        'horse_indices': horse_indices,
        'subject_idx': subject_idx,
        'analysis': analysis,
        'img_width': img_width,
        'img_height': img_height
    }, None

# All detection logic is now handled by the shared library

def draw_detection_overlay(image_path, detection_result):
    """Draw bounding box and segmentation mask on the image."""
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    if detection_result['num_horses'] == 1:
        # Single horse - draw the only detection
        horse_idx = detection_result['horse_indices'][0]
        bbox = detection_result['result'].boxes.xyxy[horse_idx].cpu().numpy()
        
        # Draw bounding box
        draw.rectangle(bbox, outline='red', width=3)
        
        # Draw segmentation mask
        mask = detection_result['result'].masks.xy[0]
        if len(mask) > 0:
            mask_points = [(float(x), float(y)) for x, y in mask]
            draw.polygon(mask_points, outline='green', width=2)
    else:
        # Multiple horses - highlight the subject horse
        subject_idx = detection_result['subject_idx']
        
        # Draw all horses in light red
        for i, horse_idx in enumerate(detection_result['horse_indices']):
            bbox = detection_result['result'].boxes.xyxy[horse_idx].cpu().numpy()
            color = 'red' if i == subject_idx else 'orange'
            width = 4 if i == subject_idx else 2
            draw.rectangle(bbox, outline=color, width=width)
        
        # Draw subject horse mask in green
        subject_horse_idx = detection_result['horse_indices'][subject_idx]
        subject_mask_idx = subject_idx  # Index within the horse arrays
        mask = detection_result['result'].masks.xy[subject_mask_idx]
        if len(mask) > 0:
            mask_points = [(float(x), float(y)) for x, y in mask]
            draw.polygon(mask_points, outline='green', width=3)
    
    return image

def main():
    st.set_page_config(page_title="Horse Detection Parameter Tuner", layout="wide")
    
    st.title("🐎 Horse Detection Parameter Tuner")
    st.markdown("Interactively tune multi-horse detection parameters and see results in real-time!")
    
    # Load default config
    default_config = load_config()
    
    # Sidebar for parameters
    st.sidebar.header("📊 Detection Parameters")
    
    # Image input
    st.sidebar.subheader("🖼️ Image Input")
    image_input = st.sidebar.text_input(
        "Image File Path or URL",
        value="",
        help="Enter file:// URL or absolute path to image"
    )
    
    if not image_input:
        st.info("👆 Please enter an image path in the sidebar to get started!")
        st.markdown("**Example:** `file:///Users/username/path/to/image.jpg`")
        return
    
    # Parse file URL if needed
    image_path = parse_file_url(image_input)
    
    # Basic detection parameters
    st.sidebar.subheader("🔍 Basic Detection")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.01, 1.0, 
        default_config['detection']['confidence_threshold'], 0.01
    )
    size_ratio_for_single_horse = st.sidebar.slider(
        "Size Ratio for Single Horse", 1.0, 5.0, 
        default_config['detection']['size_ratio_for_single_horse'], 0.1
    )
    
    # Subject identification parameters
    st.sidebar.subheader("🎯 Subject Identification")
    area_weight = st.sidebar.slider(
        "Area Weight", 0.0, 1.0, 
        default_config['detection']['subject_identification']['area_weight'], 0.05
    )
    depth_weight = st.sidebar.slider(
        "Depth Weight", 0.0, 1.0, 
        default_config['detection']['subject_identification']['depth_weight'], 0.05
    )
    edge_penalty_factor = st.sidebar.slider(
        "Edge Penalty Factor", 0.0, 1.0, 
        default_config['detection']['subject_identification']['edge_penalty_factor'], 0.05
    )
    
    # Depth analysis parameters
    st.sidebar.subheader("🏔️ Depth Analysis")
    vertical_position_weight = st.sidebar.slider(
        "Vertical Position Weight", 0.0, 1.0, 
        default_config['detection']['depth_analysis']['vertical_position_weight'], 0.05
    )
    occlusion_boost_weight = st.sidebar.slider(
        "Occlusion Boost Weight", 0.0, 1.0, 
        default_config['detection']['depth_analysis']['occlusion_boost_weight'], 0.05
    )
    occlusion_penalty_weight = st.sidebar.slider(
        "Occlusion Penalty Weight", 0.0, 1.0, 
        default_config['detection']['depth_analysis']['occlusion_penalty_weight'], 0.05
    )
    overlap_threshold = st.sidebar.slider(
        "Overlap Threshold", 0.0, 0.5, 
        default_config['detection']['depth_analysis']['overlap_threshold'], 0.01
    )
    perspective_score_boost = st.sidebar.slider(
        "Perspective Score Boost", 0.0, 1.0, 
        default_config['detection']['depth_analysis']['perspective_score_boost'], 0.05
    )
    perspective_size_threshold = st.sidebar.slider(
        "Perspective Size Threshold", 0.0, 0.5, 
        default_config['detection']['depth_analysis']['perspective_size_threshold'], 0.01
    )
    center_position_weight = st.sidebar.slider(
        "Center Position Weight", 0.0, 1.0, 
        default_config['detection']['depth_analysis']['center_position_weight'], 0.05
    )
    
    # Classification parameters
    st.sidebar.subheader("⚖️ Classification")
    depth_dominance_threshold = st.sidebar.slider(
        "Depth Dominance Threshold", 0.0, 1.0, 
        default_config['detection']['classification']['depth_dominance_threshold'], 0.01
    )
    strong_size_dominance_threshold = st.sidebar.slider(
        "Strong Size Dominance Threshold", 1.0, 20.0, 
        default_config['detection']['classification']['strong_size_dominance_threshold'], 0.1
    )
    extreme_overlap_threshold = st.sidebar.slider(
        "Extreme Overlap Threshold", 0.0, 1.0, 
        default_config['detection']['classification']['extreme_overlap_threshold'], 0.05
    )
    
    # Edge cropping parameters
    st.sidebar.subheader("✂️ Edge Cropping")
    edge_threshold_pixels = st.sidebar.slider(
        "Edge Threshold (pixels)", 1, 20, 
        default_config['detection']['edge_cropping']['edge_threshold_pixels'], 1
    )
    severity_edge_weight = st.sidebar.slider(
        "Severity Edge Weight", 0.0, 1.0, 
        default_config['detection']['edge_cropping']['severity_edge_weight'], 0.05
    )
    severity_large_object_weight = st.sidebar.slider(
        "Severity Large Object Weight", 0.0, 1.0, 
        default_config['detection']['edge_cropping']['severity_large_object_weight'], 0.05
    )
    severity_close_margin_weight = st.sidebar.slider(
        "Severity Close Margin Weight", 0.0, 1.0, 
        default_config['detection']['edge_cropping']['severity_close_margin_weight'], 0.05
    )
    large_object_width_threshold = st.sidebar.slider(
        "Large Object Width Threshold", 0.0, 1.0, 
        default_config['detection']['edge_cropping']['large_object_width_threshold'], 0.05
    )
    large_object_height_threshold = st.sidebar.slider(
        "Large Object Height Threshold", 0.0, 1.0, 
        default_config['detection']['edge_cropping']['large_object_height_threshold'], 0.05
    )
    close_margin_threshold = st.sidebar.slider(
        "Close Margin Threshold", 0, 10, 
        default_config['detection']['edge_cropping']['close_margin_threshold'], 1
    )
    edge_advantage_significant_threshold = st.sidebar.slider(
        "Edge Advantage Significant Threshold", 0.0, 1.0, 
        default_config['detection']['classification']['edge_advantage_significant_threshold'], 0.01
    )
    edge_advantage_moderate_threshold = st.sidebar.slider(
        "Edge Advantage Moderate Threshold", 0.0, 1.0, 
        default_config['detection']['classification']['edge_advantage_moderate_threshold'], 0.01
    )
    edge_significant_size_reduction = st.sidebar.slider(
        "Edge Significant Size Reduction", 0.0, 1.0, 
        default_config['detection']['classification']['edge_significant_size_reduction'], 0.05
    )
    edge_significant_depth_reduction = st.sidebar.slider(
        "Edge Significant Depth Reduction", 0.0, 1.0, 
        default_config['detection']['classification']['edge_significant_depth_reduction'], 0.05
    )
    edge_significant_scaling_factor = st.sidebar.slider(
        "Edge Significant Scaling Factor", 1, 10, 
        default_config['detection']['classification']['edge_significant_scaling_factor'], 1
    )
    edge_moderate_size_factor = st.sidebar.slider(
        "Edge Moderate Size Factor", 0.0, 1.0, 
        default_config['detection']['classification']['edge_moderate_size_factor'], 0.05
    )
    edge_moderate_depth_threshold = st.sidebar.slider(
        "Edge Moderate Depth Threshold", 0.0, 1.0, 
        default_config['detection']['classification']['edge_moderate_depth_threshold'], 0.01
    )
    
    # Collect all parameters
    config_params = {
        'confidence_threshold': confidence_threshold,
        'size_ratio_for_single_horse': size_ratio_for_single_horse,
        'area_weight': area_weight,
        'depth_weight': depth_weight,
        'edge_penalty_factor': edge_penalty_factor,
        'vertical_position_weight': vertical_position_weight,
        'occlusion_boost_weight': occlusion_boost_weight,
        'occlusion_penalty_weight': occlusion_penalty_weight,
        'overlap_threshold': overlap_threshold,
        'perspective_score_boost': perspective_score_boost,
        'perspective_size_threshold': perspective_size_threshold,
        'center_position_weight': center_position_weight,
        'depth_dominance_threshold': depth_dominance_threshold,
        'strong_size_dominance_threshold': strong_size_dominance_threshold,
        'extreme_overlap_threshold': extreme_overlap_threshold,
        'edge_threshold_pixels': edge_threshold_pixels,
        'severity_edge_weight': severity_edge_weight,
        'severity_large_object_weight': severity_large_object_weight,
        'severity_close_margin_weight': severity_close_margin_weight,
        'large_object_width_threshold': large_object_width_threshold,
        'large_object_height_threshold': large_object_height_threshold,
        'close_margin_threshold': close_margin_threshold,
        'edge_advantage_significant_threshold': edge_advantage_significant_threshold,
        'edge_advantage_moderate_threshold': edge_advantage_moderate_threshold,
        'edge_significant_size_reduction': edge_significant_size_reduction,
        'edge_significant_depth_reduction': edge_significant_depth_reduction,
        'edge_significant_scaling_factor': edge_significant_scaling_factor,
        'edge_moderate_size_factor': edge_moderate_size_factor,
        'edge_moderate_depth_threshold': edge_moderate_depth_threshold
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🖼️ Image Analysis")
        
        # Analyze image
        with st.spinner("Analyzing image..."):
            detection_result, error = analyze_image(image_path, config_params)
        
        if error:
            st.error(f"Error: {error}")
            return
        
        # Display result
        if detection_result['classification'] == 'SINGLE':
            st.success(f"🟢 **Classification: {detection_result['classification']}**")
        else:
            st.info(f"🔵 **Classification: {detection_result['classification']}**")
        
        st.write(f"**Reason:** {detection_result['reason']}")
        
        # Display image with overlay
        overlay_image = draw_detection_overlay(image_path, detection_result)
        st.image(overlay_image, caption="Detection Results (Red=Subject/Only Horse, Orange=Other Horses, Green=Subject Mask)", use_container_width=True)
    
    with col2:
        st.header("📊 Analysis Details")
        
        if detection_result['num_horses'] == 1:
            st.write("**Single horse detected**")
            st.write("No classification analysis needed.")
        else:
            analysis = detection_result['analysis']
            
            # Classification criteria
            st.subheader("🎯 Classification Criteria")
            
            # Condition 1: Size AND Depth Dominance
            condition1_color = "🟢" if analysis['condition1'] else "🔴"
            st.write(f"{condition1_color} **Condition 1:** Size AND Depth Dominance")
            st.write(f"   - Size dominance: {analysis['size_dominance']} (ratio: {analysis['size_ratio']:.2f})")
            st.write(f"   - Depth dominance: {analysis['depth_dominance']} (advantage: {analysis['depth_advantage']:.3f})")
            
            # Condition 2: Strong Size Dominance
            condition2_color = "🟢" if analysis['condition2'] else "🔴"
            st.write(f"{condition2_color} **Condition 2:** Strong Size Dominance")
            st.write(f"   - Strong size: {analysis['strong_size_dominance']} (ratio: {analysis['size_ratio']:.2f})")
            
            # Condition 3: Edge Cropping Advantage
            condition3_color = "🟢" if analysis['condition3'] else "🔴"
            st.write(f"{condition3_color} **Condition 3:** Edge Cropping Advantage")
            st.write(f"   - Edge advantage: {analysis['edge_cropping_advantage']} (severity diff: {analysis['severity_advantage']:.3f})")
            
            # Condition 4: Extreme Occlusion
            condition4_color = "🟢" if analysis['condition4'] else "🔴"
            st.write(f"{condition4_color} **Condition 4:** Extreme Occlusion")
            st.write(f"   - Extreme occlusion: {analysis['extreme_occlusion']} ({analysis['high_overlap_count']}/{analysis['num_other_horses']} horses)")
            
            # Detailed metrics
            st.subheader("📈 Detailed Metrics")
            st.write(f"**Number of horses:** {detection_result['num_horses']}")
            st.write(f"**Subject horse area:** {analysis['subject_area']:.0f}")
            st.write(f"**Largest other area:** {analysis['max_other_area']:.0f}")
            st.write(f"**Subject depth score:** {analysis['subject_depth_score']:.3f}")
            st.write(f"**Max other depth score:** {analysis['max_other_depth']:.3f}")
            st.write(f"**Subject edge severity:** {analysis['subject_severity']:.3f}")
            st.write(f"**Avg other edge severity:** {analysis['avg_other_severity']:.3f}")
    
    # Configuration management buttons
    st.sidebar.subheader("⚙️ Configuration Management")
    
    col_save, col_reset = st.sidebar.columns(2)
    
    with col_save:
        if st.button("💾 Save to Config", help="Save current parameter values to config.yml"):
            # Update config with current parameters
            config = load_config()
            config['detection']['confidence_threshold'] = confidence_threshold
            config['detection']['size_ratio_for_single_horse'] = size_ratio_for_single_horse
            config['detection']['subject_identification']['area_weight'] = area_weight
            config['detection']['subject_identification']['depth_weight'] = depth_weight
            config['detection']['subject_identification']['edge_penalty_factor'] = edge_penalty_factor
            config['detection']['depth_analysis']['vertical_position_weight'] = vertical_position_weight
            config['detection']['depth_analysis']['occlusion_boost_weight'] = occlusion_boost_weight
            config['detection']['depth_analysis']['occlusion_penalty_weight'] = occlusion_penalty_weight
            config['detection']['depth_analysis']['overlap_threshold'] = overlap_threshold
            config['detection']['depth_analysis']['perspective_score_boost'] = perspective_score_boost
            config['detection']['depth_analysis']['perspective_size_threshold'] = perspective_size_threshold
            config['detection']['depth_analysis']['center_position_weight'] = center_position_weight
            config['detection']['classification']['depth_dominance_threshold'] = depth_dominance_threshold
            config['detection']['classification']['strong_size_dominance_threshold'] = strong_size_dominance_threshold
            config['detection']['classification']['extreme_overlap_threshold'] = extreme_overlap_threshold
            config['detection']['edge_cropping']['edge_threshold_pixels'] = edge_threshold_pixels
            config['detection']['edge_cropping']['severity_edge_weight'] = severity_edge_weight
            config['detection']['edge_cropping']['severity_large_object_weight'] = severity_large_object_weight
            config['detection']['edge_cropping']['severity_close_margin_weight'] = severity_close_margin_weight
            config['detection']['edge_cropping']['large_object_width_threshold'] = large_object_width_threshold
            config['detection']['edge_cropping']['large_object_height_threshold'] = large_object_height_threshold
            config['detection']['edge_cropping']['close_margin_threshold'] = close_margin_threshold
            config['detection']['classification']['edge_advantage_significant_threshold'] = edge_advantage_significant_threshold
            config['detection']['classification']['edge_advantage_moderate_threshold'] = edge_advantage_moderate_threshold
            config['detection']['classification']['edge_significant_size_reduction'] = edge_significant_size_reduction
            config['detection']['classification']['edge_significant_depth_reduction'] = edge_significant_depth_reduction
            config['detection']['classification']['edge_significant_scaling_factor'] = edge_significant_scaling_factor
            config['detection']['classification']['edge_moderate_size_factor'] = edge_moderate_size_factor
            config['detection']['classification']['edge_moderate_depth_threshold'] = edge_moderate_depth_threshold
            
            # Save to file
            with open('config.yml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            st.sidebar.success("✅ Configuration saved!")
    
    with col_reset:
        if st.button("🔄 Reset to Config", help="Reset all parameters to current config.yml values"):
            # Clear Streamlit's cache to force reload of config
            st.cache_resource.clear()
            st.sidebar.success("🔄 Parameters reset!")
            st.rerun()

if __name__ == "__main__":
    main()