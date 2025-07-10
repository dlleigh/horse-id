# Horse Detection Configuration Guide

This document explains all the detection constants in the `config.yml` file and how they work together to classify images as containing SINGLE or MULTIPLE horses.

## Overview

The multi-horse detector uses a sophisticated algorithm that goes beyond simple horse counting. It analyzes multiple factors to determine whether an image should be classified as containing a single subject horse or multiple horses that are equally important.

## Basic Detection Settings

### `yolo_model: 'yolo11x-seg.pt'`
The YOLO segmentation model used for horse detection. This model provides both bounding boxes and segmentation masks for each detected horse.

### `confidence_threshold: 0.1`
Minimum confidence score (0-1) for YOLO to consider a detection valid. Lower values detect more horses but may include false positives.

### `size_ratio_for_single_horse: 2.2`
Traditional size-based threshold. If the largest horse is at least 2.2 times larger than the second largest, it suggests a single subject horse scenario.

## Depth Analysis System

The depth analysis system determines which horses are in the foreground vs background by analyzing their spatial relationships.

### `vertical_position_weight: 0.3`
Weight for vertical position scoring. Horses lower in the frame (higher Y coordinates) typically appear closer to the camera and receive higher depth scores.

### `occlusion_boost_weight: 0.4`
Boost applied to horses that overlap with others and appear to be in front. When two horses overlap, the one that appears more complete gets a depth advantage.

### `occlusion_penalty_weight: 0.2`
Penalty applied to horses that appear to be behind others in overlapping scenarios.

### `overlap_threshold: 0.1`
Minimum overlap ratio (0-1) between two horse bounding boxes to trigger occlusion analysis. Only meaningful overlaps are considered.

### `perspective_score_boost: 0.2`
Score boost for horses whose size matches their expected size based on their position in the frame (simple perspective correction).

### `perspective_size_threshold: 0.1`
Minimum normalized size threshold for perspective correction calculations.

### `center_position_weight: 0.1`
Weight for horizontal center positioning. Horses closer to the horizontal center of the image receive slightly higher depth scores.

## Edge Cropping Detection

This system identifies horses that are partially cut off by image boundaries, which often indicates they are not the main subject. The logic works in several steps:

### Step 1: Calculate Margins
For each horse bounding box `(x1, y1, x2, y2)`, calculate distance to each edge:
```
left_margin = x1                    # Distance from left edge
top_margin = y1                     # Distance from top edge  
right_margin = img_width - x2       # Distance from right edge
bottom_margin = img_height - y2     # Distance from bottom edge
```

### Step 2: Count Edge Touches
Count how many edges the horse "touches" (within threshold):
- `edge_threshold_pixels: 5` - If margin ≤ 5 pixels, the horse "touches" that edge
- `edge_touches` = count of edges touched (0-4)

**Examples:**
- Horse entirely inside image: `edge_touches = 0`
- Horse touching right edge: `edge_touches = 1`
- Horse in corner: `edge_touches = 2`
- Horse spanning full width: `edge_touches = 2` (left + right)

### Step 3: Calculate Size Ratios
```
width_ratio = horse_width / image_width     # How much of image width the horse occupies
height_ratio = horse_height / image_height  # How much of image height the horse occupies
```

### Step 4: Calculate Severity Score
The severity score (0-1) indicates how severely cropped the horse is:

**Base severity from edge touches:**
- `severity_edge_weight: 0.25` - Base score per edge touched
- `severity_score += edge_touches × 0.25`

**Additional penalty for large objects:**
- `large_object_width_threshold: 0.8` - If horse occupies >80% of image width
- `large_object_height_threshold: 0.8` - If horse occupies >80% of image height
- `severity_large_object_weight: 0.4` - Additional penalty for large objects at edges

**Logic:** Large objects touching edges are more likely to be cropped subjects rather than naturally small background objects.

**Extreme proximity penalty:**
- `close_margin_threshold: 1` - If any margin ≤ 1 pixel
- `severity_close_margin_weight: 0.3` - Additional penalty for extremely close margins

### Step 5: Determine if Significantly Cropped
A horse is considered "significantly cropped" if:
- It touches 2+ edges, OR
- It touches 1+ edges AND is a large object (width >80% OR height >80%)

### Severity Score Examples

**Example 1: Small background horse near right edge**
- `edge_touches = 1`, `width_ratio = 0.2`, `height_ratio = 0.3`
- `severity_score = 1 × 0.25 = 0.25`
- `is_significantly_cropped = False` (only 1 edge, not large)

**Example 2: Large foreground horse cut off at bottom**
- `edge_touches = 1`, `width_ratio = 0.6`, `height_ratio = 0.9`
- `severity_score = 1 × 0.25 + 0.4 = 0.65` (large object penalty)
- `is_significantly_cropped = True` (1 edge + large object)

**Example 3: Horse in corner**
- `edge_touches = 2`, `width_ratio = 0.4`, `height_ratio = 0.4`
- `severity_score = 2 × 0.25 = 0.5`
- `is_significantly_cropped = True` (2 edges)

## How Edge Cropping Affects Classification

The edge cropping system helps in two ways:

### 1. Subject Identification
Horses with high cropping severity get penalized in subject identification:
```
subject_score = (area_weight × area + depth_weight × depth) × (1 - edge_penalty_factor × severity)
```
- `edge_penalty_factor: 0.6` - Cropped horses get reduced subject scores

### 2. Edge Cropping Advantage
If other horses are significantly more cropped than the subject horse, relaxed classification criteria apply:

**Significant Advantage** (severity difference > 0.15):
- Size requirement reduced by 70%
- Depth requirement reduced by 70%
- `relaxed_size_ratio = 2.2 × (1 - 0.7 × severity_factor) = 0.66` (much more lenient)

**Moderate Advantage** (severity difference > 0.05):
- Size requirement: `2.2 × 0.6 = 1.32`
- Depth requirement: `0.15 → 0.1`

**Real-world scenario:** Subject horse in center (severity=0.1), competing horse mostly cropped out (severity=0.8)
- Severity advantage: `0.8 - 0.1 = 0.7` (significant)
- Relaxed requirements make it easier to classify as SINGLE

## Subject Horse Identification

This system combines multiple factors to identify which horse is the main subject of the image.

### `area_weight: 0.6`
Weight for area (size) in the subject identification score. Larger horses are more likely to be the subject.

### `depth_weight: 0.4`
Weight for depth analysis in the subject identification score. Horses with higher depth scores (more likely in foreground) are favored.

### `edge_penalty_factor: 0.6`
Penalty factor (0-1) applied to horses that are edge-cropped. Higher values mean more penalty for cropped horses.

**Subject Score Formula:**
```
subject_score = (area_weight × normalized_area + depth_weight × normalized_depth) × (1 - edge_penalty_factor × cropping_severity)
```

## Classification Decision Logic

The final SINGLE vs MULTIPLE classification uses several criteria that can independently trigger a SINGLE classification.

### `depth_dominance_threshold: 0.15`
Required depth score advantage for a horse to be considered "depth dominant." The subject horse must have a depth score at least 0.15 higher than the next best horse.

### `strong_size_dominance_threshold: 1.8`
**Critical threshold:** If the subject horse is at least 1.8 times larger than the largest other horse, it can be classified as SINGLE even without depth dominance. This handles cases with clear foreground subjects and distant background horses.

### `extreme_overlap_threshold: 0.7`
Overlap ratio threshold for "extreme occlusion." If the subject horse overlaps with ALL other horses by at least 70%, the image is classified as SINGLE.

## Edge Cropping Advantage System

This system provides more lenient classification when other horses are significantly edge-cropped.

### `edge_advantage_significant_threshold: 0.15`
Severity difference threshold for "significant" edge cropping advantage. If other horses are cropped 0.15 more severely than the subject, relaxed criteria apply.

### `edge_advantage_moderate_threshold: 0.05`
Severity difference threshold for "moderate" edge cropping advantage with slightly relaxed criteria.

### `edge_significant_size_reduction: 0.7`
Factor by which size requirements are reduced when significant edge advantage is detected.

### `edge_significant_depth_reduction: 0.7`
Factor by which depth requirements are reduced when significant edge advantage is detected.

### `edge_significant_scaling_factor: 3`
Scaling factor for severity advantage calculations in significant edge cropping scenarios.

### `edge_moderate_size_factor: 0.6`
Size factor for moderate edge advantage scenarios.

### `edge_moderate_depth_threshold: 0.1`
Depth threshold for moderate edge advantage scenarios.

## Classification Decision Tree

An image is classified as **SINGLE** if ANY of these conditions are met:

1. **Traditional Criteria:** `(size_dominance AND depth_dominance)`
   - Size dominance: Subject ≥ 2.2× larger than next largest
   - Depth dominance: Subject depth score ≥ 0.15 higher than next best

2. **Strong Size Dominance:** `subject_area / largest_other_area ≥ 1.8`
   - Overrides depth requirements for clear foreground/background scenarios

3. **Edge Cropping Advantage:** Other horses are significantly more cropped than subject
   - Relaxed size/depth requirements based on cropping severity difference

4. **Extreme Occlusion:** Subject horse overlaps ≥ 70% with ALL other horses
   - Handles cases where subject horse almost completely hides others

Otherwise, the image is classified as **MULTIPLE**.

## Tuning Guidelines

- **Increase `strong_size_dominance_threshold`** if too many images with distant background horses are classified as SINGLE
- **Decrease `strong_size_dominance_threshold`** if images with clear foreground subjects are classified as MULTIPLE
- **Adjust `area_weight` vs `depth_weight`** to balance size vs position in subject identification
- **Modify `depth_dominance_threshold`** to make depth requirements more/less strict
- **Tune edge cropping thresholds** based on your image dataset characteristics

The current configuration is optimized for horse photography where there's typically one clear subject horse with possible distant background horses.