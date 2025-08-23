#!/usr/bin/env python3
"""
Shared library for horse detection logic.
Contains all the core detection algorithms used by both multi_horse_detector.py
and tune_detection_parameters.py to ensure consistency.
"""

import numpy as np
import yaml
from typing import Tuple, Dict, Any, List


def load_detection_config() -> Dict[str, Any]:
    """Load detection configuration from config.yml"""
    from config_utils import load_config
    config = load_config()
    return {
        'depth_analysis': config['detection']['depth_analysis'],
        'edge_cropping': config['detection']['edge_cropping'],
        'subject_identification': config['detection']['subject_identification'],
        'classification': config['detection']['classification'],
        'size_ratio': config['detection']['size_ratio_for_single_horse']
    }


def calculate_bbox_overlap(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate the overlap ratio of bbox2 with bbox1 (how much of bbox2 is covered by bbox1)."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0  # No overlap
    
    intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    return intersection_area / bbox2_area if bbox2_area > 0 else 0.0


def calculate_distance_from_center(bbox: np.ndarray, img_width: int, img_height: int) -> float:
    """Calculate normalized distance from bbox center to image center."""
    bbox_center_x = bbox[0] + bbox[2] / 2
    bbox_center_y = bbox[1] + bbox[3] / 2
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Normalize by image diagonal
    img_diagonal = np.sqrt(img_width**2 + img_height**2)
    distance = np.sqrt((bbox_center_x - img_center_x)**2 + (bbox_center_y - img_center_y)**2)
    
    return distance / img_diagonal


def analyze_depth_relationships(horse_boxes, horse_masks, horse_indices: np.ndarray, 
                               img_width: int, img_height: int, 
                               config: Dict[str, Any] = None) -> np.ndarray:
    """
    Analyze spatial relationships to determine which horses are in foreground vs background.
    Returns a depth score for each horse (higher = more likely to be foreground).
    
    Args:
        horse_boxes: YOLO boxes for horses
        horse_masks: YOLO masks for horses  
        horse_indices: Indices of horse detections
        img_width: Image width
        img_height: Image height
        config: Configuration dict or None to load from file
    """
    if config is None:
        detection_config = load_detection_config()
        depth_config = detection_config['depth_analysis']
    else:
        depth_config = config
    
    num_horses = len(horse_indices)
    depth_scores = np.zeros(num_horses)
    
    for i in range(num_horses):
        score = 0.0
        
        # Factor 1: Vertical position (lower in image often means closer)
        bbox_i = horse_boxes.xyxy[i].cpu().numpy()
        y_center_i = (bbox_i[1] + bbox_i[3]) / 2
        y_bottom_i = bbox_i[3]
        
        # Normalize y positions
        normalized_y_center = y_center_i / img_height
        normalized_y_bottom = y_bottom_i / img_height
        
        # Horses lower in the frame get higher depth scores
        score += normalized_y_bottom * depth_config['vertical_position_weight']
        
        # Factor 2: Occlusion analysis - which horse is "on top"
        for j in range(num_horses):
            if i == j:
                continue
                
            bbox_j = horse_boxes.xyxy[j].cpu().numpy()
            
            # Check if bboxes overlap
            overlap = calculate_bbox_overlap(bbox_i, bbox_j)
            
            if overlap > depth_config['overlap_threshold']:  # If there's meaningful overlap
                # The horse that appears more complete (less occluded) is likely in front
                y_center_j = (bbox_j[1] + bbox_j[3]) / 2
                
                # If horse i is lower AND overlapping, it's likely in foreground
                if y_center_i > y_center_j:
                    score += depth_config['occlusion_boost_weight'] * overlap
                else:
                    score -= depth_config['occlusion_penalty_weight'] * overlap
        
        # Factor 3: Size relative to position (perspective correction)
        bbox_area = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
        normalized_area = bbox_area / (img_width * img_height)
        
        # Expected size based on position (simple perspective model)
        expected_size_factor = 0.3 + (normalized_y_center * 0.7)  # 0.3 to 1.0
        
        # If actual size matches expected size for position, boost score
        if normalized_area >= expected_size_factor * depth_config['perspective_size_threshold']:
            score += depth_config['perspective_score_boost']
        
        # Factor 4: Central positioning (subjects are often more centered horizontally)
        x_center_i = (bbox_i[0] + bbox_i[2]) / 2
        normalized_x_center = x_center_i / img_width
        center_distance = abs(normalized_x_center - 0.5)  # Distance from horizontal center
        score += (0.5 - center_distance) * depth_config['center_position_weight']  # Closer to center = higher score
        
        depth_scores[i] = score
    
    return depth_scores


def is_edge_cropped(bbox_xyxy: np.ndarray, img_width: int, img_height: int, 
                   config: Dict[str, Any] = None) -> Tuple[bool, float]:
    """
    Determine if a bounding box is significantly cropped by image edges.
    Returns a tuple: (is_significantly_cropped, cropping_severity_score)
    
    Args:
        bbox_xyxy: Bounding box in xyxy format
        img_width: Image width
        img_height: Image height
        config: Configuration dict or None to load from file
    """
    if config is None:
        detection_config = load_detection_config()
        edge_config = detection_config['edge_cropping']
    else:
        edge_config = config
    
    x1, y1, x2, y2 = bbox_xyxy
    
    # Check how close the bbox is to each edge (in pixels)
    left_margin = x1
    top_margin = y1
    right_margin = img_width - x2
    bottom_margin = img_height - y2
    
    # Count how many edges the bbox touches (within threshold)
    edge_threshold = edge_config['edge_threshold_pixels']
    edge_touches = 0
    if left_margin <= edge_threshold:
        edge_touches += 1
    if top_margin <= edge_threshold:
        edge_touches += 1
    if right_margin <= edge_threshold:
        edge_touches += 1
    if bottom_margin <= edge_threshold:
        edge_touches += 1
    
    # Calculate size ratios
    width_ratio = (x2 - x1) / img_width
    height_ratio = (y2 - y1) / img_height
    
    # Calculate cropping severity score (0 = not cropped, 1 = severely cropped)
    severity_score = 0.0
    
    # Multiple edge touches increase severity
    severity_score += edge_touches * edge_config['severity_edge_weight']
    
    # Large objects touching edges are more likely to be cropped
    if edge_touches >= 1:
        if width_ratio > edge_config['large_object_width_threshold']:
            severity_score += edge_config['severity_large_object_weight']
        if height_ratio > edge_config['large_object_height_threshold']:
            severity_score += edge_config['severity_large_object_weight']
    
    # Very close margins indicate severe cropping
    min_margin = min(left_margin, top_margin, right_margin, bottom_margin)
    if min_margin <= edge_config['close_margin_threshold']:
        severity_score += edge_config['severity_close_margin_weight']
    
    # Determine if significantly cropped
    is_significantly_cropped = (edge_touches >= 2) or (edge_touches >= 1 and (
        width_ratio > edge_config['large_object_width_threshold'] or 
        height_ratio > edge_config['large_object_height_threshold']))
    
    return is_significantly_cropped, min(severity_score, 1.0)


def identify_subject_horse(horse_boxes, horse_masks, horse_indices: np.ndarray, 
                          areas: np.ndarray, img_width: int, img_height: int,
                          config: Dict[str, Any] = None) -> int:
    """
    Identify the most likely subject horse considering size, depth, and edge cropping.
    Returns the index of the subject horse within the horses array.
    
    Args:
        horse_boxes: YOLO boxes for horses
        horse_masks: YOLO masks for horses
        horse_indices: Indices of horse detections
        areas: Array of horse areas
        img_width: Image width
        img_height: Image height
        config: Configuration dict or None to load from file
    """
    if config is None:
        detection_config = load_detection_config()
        subject_config = detection_config['subject_identification']
        depth_config = detection_config['depth_analysis']
        edge_config = detection_config['edge_cropping']
    else:
        subject_config = config
        depth_config = config
        edge_config = config
    
    depth_scores = analyze_depth_relationships(horse_boxes, horse_masks, horse_indices, 
                                             img_width, img_height, depth_config)
    
    # Check for edge cropping with severity scores
    edge_cropped = []
    cropping_severities = []
    for i in range(len(horse_indices)):
        bbox_xyxy = horse_boxes.xyxy[i].cpu().numpy()
        is_cropped, severity = is_edge_cropped(bbox_xyxy, img_width, img_height, edge_config)
        edge_cropped.append(is_cropped)
        cropping_severities.append(severity)
    
    # Combine area, depth scores, and edge cropping
    # Normalize areas to 0-1 scale
    max_area = np.max(areas)
    normalized_areas = areas / max_area
    
    # Normalize depth scores to 0-1 scale
    if np.max(depth_scores) > np.min(depth_scores):
        normalized_depth_scores = (depth_scores - np.min(depth_scores)) / (np.max(depth_scores) - np.min(depth_scores))
    else:
        normalized_depth_scores = np.ones_like(depth_scores)
    
    # Penalize based on cropping severity (more nuanced than binary)
    edge_penalty = np.array([1.0 - (severity * subject_config['edge_penalty_factor']) for severity in cropping_severities])
    
    # Combined score: depth weighted heavily, size moderately, with edge penalty
    combined_scores = (subject_config['area_weight'] * normalized_areas + 
                      subject_config['depth_weight'] * normalized_depth_scores) * edge_penalty
    
    return np.argmax(combined_scores)


def analyze_classification_criteria(horse_boxes, horse_masks, horse_indices: np.ndarray,
                                  areas: np.ndarray, subject_idx: int, 
                                  img_width: int, img_height: int,
                                  config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze all classification criteria and return detailed results.
    
    Args:
        horse_boxes: YOLO boxes for horses
        horse_masks: YOLO masks for horses
        horse_indices: Indices of horse detections
        areas: Array of horse areas
        subject_idx: Index of subject horse within horses array
        img_width: Image width
        img_height: Image height
        config: Configuration dict or None to load from file
    
    Returns:
        Dictionary with classification analysis results
    """
    if config is None:
        detection_config = load_detection_config()
        classification_config = detection_config['classification']
        size_ratio = detection_config['size_ratio']
        depth_config = detection_config['depth_analysis']
        edge_config = detection_config['edge_cropping']
    else:
        classification_config = config
        size_ratio = config.get('size_ratio_for_single_horse', 2.2)
        depth_config = config
        edge_config = config
    
    subject_area = areas[subject_idx]
    other_areas = np.delete(areas, subject_idx)
    
    # Get depth scores
    depth_scores = analyze_depth_relationships(horse_boxes, horse_masks, horse_indices, 
                                             img_width, img_height, depth_config)
    subject_depth_score = depth_scores[subject_idx]
    other_depth_scores = np.delete(depth_scores, subject_idx)
    
    # Size analysis
    max_other_area = max(other_areas)
    size_ratio_val = subject_area / max_other_area
    size_dominance = subject_area >= size_ratio * max_other_area
    strong_size_dominance = size_ratio_val >= classification_config['strong_size_dominance_threshold']
    
    # Depth analysis
    max_other_depth = max(other_depth_scores)
    depth_advantage = subject_depth_score - max_other_depth
    depth_dominance = depth_advantage > classification_config['depth_dominance_threshold']
    
    # Edge cropping analysis
    subject_bbox_xyxy = horse_boxes.xyxy[subject_idx].cpu().numpy()
    subject_is_cropped, subject_severity = is_edge_cropped(subject_bbox_xyxy, img_width, img_height, edge_config)
    
    other_cropping_severities = []
    for i, other_idx in enumerate([idx for j, idx in enumerate(range(len(horse_indices))) if j != subject_idx]):
        other_bbox_xyxy = horse_boxes.xyxy[other_idx].cpu().numpy()
        _, severity = is_edge_cropped(other_bbox_xyxy, img_width, img_height, edge_config)
        other_cropping_severities.append(severity)
    
    avg_other_severity = np.mean(other_cropping_severities)
    severity_advantage = avg_other_severity - subject_severity
    
    # Full edge cropping advantage logic
    edge_cropping_advantage = False
    if severity_advantage > classification_config['edge_advantage_significant_threshold']:
        # More lenient requirements based on severity difference
        severity_factor = min(severity_advantage * classification_config['edge_significant_scaling_factor'], 1.0)
        relaxed_size_ratio = size_ratio * (1 - severity_factor * classification_config['edge_significant_size_reduction'])
        relaxed_depth_threshold = classification_config['depth_dominance_threshold'] * (1 - severity_factor * classification_config['edge_significant_depth_reduction'])
        
        relaxed_size_dominance = subject_area >= relaxed_size_ratio * max_other_area
        relaxed_depth_dominance = subject_depth_score > max_other_depth + relaxed_depth_threshold
        
        edge_cropping_advantage = relaxed_size_dominance or relaxed_depth_dominance
    elif severity_advantage > classification_config['edge_advantage_moderate_threshold']:
        # Slightly relaxed requirements
        relaxed_size_dominance = subject_area >= size_ratio * classification_config['edge_moderate_size_factor'] * max_other_area
        relaxed_depth_dominance = subject_depth_score > max_other_depth + classification_config['edge_moderate_depth_threshold']
        edge_cropping_advantage = relaxed_size_dominance or relaxed_depth_dominance
    
    # Extreme occlusion analysis
    high_overlap_count = 0
    for i, other_idx in enumerate([idx for j, idx in enumerate(range(len(horse_indices))) if j != subject_idx]):
        other_bbox_xyxy = horse_boxes.xyxy[other_idx].cpu().numpy()
        overlap_ratio = calculate_bbox_overlap(subject_bbox_xyxy, other_bbox_xyxy)
        if overlap_ratio > classification_config['extreme_overlap_threshold']:
            high_overlap_count += 1
    
    extreme_occlusion = high_overlap_count == len(other_areas)
    
    # Final decision
    condition1 = size_dominance and depth_dominance
    condition2 = strong_size_dominance
    condition3 = edge_cropping_advantage
    condition4 = extreme_occlusion
    
    should_classify_as_single = condition1 or condition2 or condition3 or condition4
    final_classification = "SINGLE" if should_classify_as_single else "MULTIPLE"
    
    # Build reason string
    reasons = []
    if condition1:
        reasons.append("Size AND depth dominance")
    if condition2:
        reasons.append("Strong size dominance")
    if condition3:
        reasons.append("Edge cropping advantage")
    if condition4:
        reasons.append("Extreme occlusion")
    
    if reasons:
        reason = f"Classified as SINGLE due to: {', '.join(reasons)}"
    else:
        reason = "Classified as MULTIPLE - no single-horse criteria met"
    
    return {
        'final_classification': final_classification,
        'reason': reason,
        'size_ratio': size_ratio_val,
        'size_dominance': size_dominance,
        'strong_size_dominance': strong_size_dominance,
        'depth_advantage': depth_advantage,
        'depth_dominance': depth_dominance,
        'severity_advantage': severity_advantage,
        'edge_cropping_advantage': edge_cropping_advantage,
        'extreme_occlusion': extreme_occlusion,
        'condition1': condition1,
        'condition2': condition2,
        'condition3': condition3,
        'condition4': condition4,
        'subject_area': subject_area,
        'max_other_area': max_other_area,
        'subject_depth_score': subject_depth_score,
        'max_other_depth': max_other_depth,
        'subject_severity': subject_severity,
        'avg_other_severity': avg_other_severity,
        'high_overlap_count': high_overlap_count,
        'num_other_horses': len(other_areas)
    }


def classify_horse_detection(horse_boxes, horse_masks, horse_indices: np.ndarray,
                           areas: np.ndarray, img_width: int, img_height: int,
                           config: Dict[str, Any] = None) -> Tuple[str, float, int, Dict[str, Any]]:
    """
    Complete horse detection classification pipeline.
    
    Args:
        horse_boxes: YOLO boxes for horses
        horse_masks: YOLO masks for horses
        horse_indices: Indices of horse detections
        areas: Array of horse areas
        img_width: Image width
        img_height: Image height
        config: Configuration dict or None to load from file
    
    Returns:
        Tuple of (classification, size_ratio, subject_idx, analysis_dict)
    """
    if len(horse_indices) == 1:
        return "SINGLE", float('nan'), 0, {
            'final_classification': 'SINGLE',
            'reason': 'Only one horse detected'
        }
    
    # Identify subject horse
    subject_idx = identify_subject_horse(horse_boxes, horse_masks, horse_indices, 
                                       areas, img_width, img_height, config)
    
    # Analyze classification criteria
    analysis = analyze_classification_criteria(horse_boxes, horse_masks, horse_indices,
                                             areas, subject_idx, img_width, img_height, config)
    
    # Calculate size ratio
    subject_area = areas[subject_idx]
    other_areas = np.delete(areas, subject_idx)
    calculated_size_ratio = float('nan')
    if len(other_areas) > 0:
        next_largest_area = max(other_areas)
        if next_largest_area > 0:
            calculated_size_ratio = subject_area / next_largest_area
    
    return analysis['final_classification'], calculated_size_ratio, subject_idx, analysis