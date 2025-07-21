# Horse Detection Algorithm Test Scenarios - Visual Guide

This document provides visual representations of the test scenarios covered in `tests/test_detection_algorithms.py`.

## 📊 Test Overview

The test suite covers **25 test scenarios** across **6 algorithm categories**:

| Algorithm Category | Tests | Purpose |
|-------------------|-------|---------|
| **Bounding Box Overlap** | 5 | Calculate intersection/overlap ratios |
| **Distance from Center** | 3 | Measure centrality of detections |
| **Edge Cropping Detection** | 4 | Identify partially visible horses |
| **Depth Analysis** | 2 | Determine foreground/background horses |
| **Subject Identification** | 1 | Select the primary horse |
| **Classification Pipeline** | 2 | Overall SINGLE/MULTIPLE classification |
| **Config Loading** | 2 | Validate configuration handling |

---

## 🔲 1. Bounding Box Overlap Tests

### Test Case 1: No Overlap
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────┐                         ┌─────────┐       │
│  │         │                         │         │       │
│  │ Bbox1   │                         │ Bbox2   │       │ 400x400
│  │ (0,0    │                         │(200,200)│       │ image
│  │ 100x100)│                         │ 100x100)│       │
│  └─────────┘                         └─────────┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `overlap = 0.0` (no intersection)

### Test Case 2: Complete Overlap
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────┐                                           │
│  │ Bbox1 & │  ← Both boxes exactly the same            │ 400x400
│  │ Bbox2   │    (0,0, 100x100)                         │ image
│  │         │                                           │
│  └─────────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `overlap = 1.0` (perfect overlap)

### Test Case 3: Partial Overlap
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────┐                                           │
│  │ Bbox1   │                                           │ 400x400
│  │ (0,0    │┌─────────┐                                │ image
│  │ 100x100)││ Bbox2   │                                │
│  └─────────┘│ (50,50  │                                │
│             │ 100x100)│                                │
│             └─────────┘                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `overlap = 0.25` (intersection area / bbox2 area)

### Test Case 4: Inner Bounding Box
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────────────────┐                               │
│  │ Bbox1 (0,0, 200x200)│                               │ 400x400
│  │                     │                               │ image
│  │   ┌─────────┐       │                               │
│  │   │ Bbox2   │       │                               │
│  │   │(50,50   │       │                               │
│  │   │100x100) │       │                               │
│  │   └─────────┘       │                               │
│  └─────────────────────┘                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `overlap = 1.0` (bbox2 completely inside bbox1)

### Test Case 5: Zero Area Edge Case
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────┐                                           │
│  │ Bbox1   │                                           │ 400x400
│  │ (0,0    │ ●← Bbox2 (50,50,50,50)                   │ image
│  │ 100x100)│   Zero area (width=0, height=0)          │
│  └─────────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `overlap = 0.0` (zero area bbox)

---

## 📐 2. Distance from Center Tests

### Test Case 1: Centered Bounding Box
```
┌─────────────────────────────────────────────────────────┐
│                     500x480 image                      │
│                                                         │
│                                                         │
│                     ┌─────┐                            │
│                     │     │ ← Bbox center at           │
│                     │ ●   │   (250, 240) = image center│
│                     │     │                            │
│                     └─────┘                            │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `distance = 0.0` (perfect center)

### Test Case 2: Corner Bounding Box
```
┌─────────────────────────────────────────────────────────┐
│ ┌─────┐               500x480 image                     │
│ │ ●   │ ← Bbox center at (25, 25)                      │
│ └─────┘                                                 │
│                                                         │
│                                                         │
│                     ●← Image center (250, 240)         │
│                                                         │
│                                                         │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `distance = normalized_euclidean_distance`

### Test Case 3: XYWH Format Validation
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
│                                                         │
│                                                         │
│                     ┌─────────┐                        │
│                     │         │ ← Bbox (100,100,200,200)│
│                     │    ●    │   Center at (200,200)   │
│                     │         │   = Image center        │
│                     └─────────┘                        │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `distance = 0.0` (validates XYWH interpretation)

---

## ✂️ 3. Edge Cropping Detection Tests

### Test Case 1: Not Cropped (Safe Zone)
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
│                                                         │
│                                                         │
│            ┌─────────────────┐                         │
│            │                 │ ← Bbox well inside      │
│            │   Not Cropped   │   (100,100,200x200)     │
│            │                 │                         │
│            └─────────────────┘                         │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `is_cropped = False, severity = 0.0`

### Test Case 2: Single Edge Touch (Small Object)
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
┌─────────┐                                               │
│ Touching│ ← Bbox touching left edge                     │
│ Left    │   (0,100,100x200) - small object             │
│ Edge    │                                               │
└─────────┘                                               │
│                                                         │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `is_cropped = False, severity > 0.0` (not severe enough)

### Test Case 3: Multiple Edge Touch
```
┌─────────────────────────────────────────────────────────┐
┌─────────┐               400x400 image                   │
│ Touches │ ← Bbox touching top AND left edges            │
│ Top &   │   (0,0,100x100)                               │
│ Left    │                                               │
└─────────┘                                               │
│                                                         │
│                                                         │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `is_cropped = True, severity > 0.0` (multiple edges = cropped)

### Test Case 4: Large Object Single Edge
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
┌─────────────────────────────────────┐                   │
│                                     │ ← Large bbox      │
│         Large Object                │   (0,50,300x200)  │
│         Touching Left Edge          │   touching left    │
│                                     │                   │
└─────────────────────────────────────┘                   │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `is_cropped = True, severity > 0.0` (large object = cropped)

---

## 🔍 4. Depth Analysis Tests

### Test Case 1: Single Horse Depth
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
│                                                         │
│                                                         │
│            ┌─────────────────┐                         │
│            │                 │ ← Single horse          │
│            │   Horse #0      │   gets baseline depth   │
│            │                 │   score                 │
│            └─────────────────┘                         │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** Single positive depth score

### Test Case 2: Multiple Horses Depth Comparison
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
│ ┌─────┐                                                 │
│ │Horse│ ← Horse #1: Top-left (50,50,50x50)             │
│ │ #1  │   Lower depth score (background)               │
│ └─────┘                                                 │
│                                                         │
│                                                         │
│                  ┌─────┐                               │
│                  │Horse│ ← Horse #0: Bottom-center     │
│                  │ #0  │   (175,300,50x80)             │
│                  └─────┘   Higher depth score (foreground)│
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `depth_scores[0] > depth_scores[1]` (bottom-center wins)

**Depth Scoring Factors:**
- **Vertical Position** (0.3 weight): Lower = higher score
- **Center Position** (0.1 weight): More centered = higher score  
- **Occlusion Analysis**: Overlap relationships
- **Perspective Size**: Larger objects tend to be closer

---

## 🎯 5. Subject Identification Test

### Test Case: Largest Horse Selection
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
│                                                         │
│    ┌─────────────────────────────────┐                 │
│    │                                 │                 │
│    │         Horse #0                │ ← Large horse   │
│    │         Area: 10,000            │   (100,100,     │
│    │         (100x100 bbox)          │    100x100)     │
│    │                                 │                 │
│    └─────────────────────────────────┘                 │
│                                                         │
│                               ┌───────┐                │
│                               │Horse#1│ ← Small horse  │
│                               │Area:  │   (300,300,    │
│                               │2,500  │    50x50)      │
│                               └───────┘                │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:** `subject_idx = 0` (larger horse selected as primary subject)

**Selection Criteria:**
- **Area Weight** (0.3): Larger horses preferred
- **Depth Weight** (0.7): Foreground horses preferred  
- **Edge Penalty**: Cropped horses penalized

---

## 🏗️ 6. Classification Pipeline Tests

### Test Case 1: Single Horse Classification
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
│                                                         │
│                                                         │
│            ┌─────────────────┐                         │
│            │                 │ ← Only one horse        │
│            │   Single Horse  │   detected              │
│            │   Area: 10,000  │                         │
│            └─────────────────┘                         │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:**
- `classification = "SINGLE"`
- `size_ratio = NaN` (no comparison possible)
- `subject_idx = 0`
- `analysis['reason'] = "Only one horse detected"`

### Test Case 2: Multiple Horses Classification
```
┌─────────────────────────────────────────────────────────┐
│                     400x400 image                      │
│                                                         │
│    ┌─────────────────────────────────┐                 │
│    │         Horse #0                │ ← Large horse   │
│    │         Area: 10,000            │   (5x larger)   │
│    └─────────────────────────────────┘                 │
│                                                         │
│                               ┌───────┐                │
│                               │Horse#1│ ← Small horse  │
│                               │Area:  │   Area: 2,000  │
│                               │2,000  │                │
│                               └───────┘                │
└─────────────────────────────────────────────────────────┘
```
**Expected Result:**
- `classification = "SINGLE" or "MULTIPLE"` (depends on algorithm decision)
- `size_ratio = 5.0` (10,000 / 2,000)
- `subject_idx = 0 or 1`
- Complex analysis based on size dominance, depth, and edge factors

**Classification Logic:**
1. **Size Dominance**: Ratio > 4.0 → Strong dominance
2. **Depth Analysis**: Foreground vs background positioning
3. **Edge Cropping**: Penalty for partially visible horses
4. **Final Decision**: SINGLE (if dominant horse) or MULTIPLE (if comparable)

---

## ⚙️ 7. Configuration Loading Tests

### Test Case 1: Successful Config Loading
```yaml
detection:
  depth_analysis:
    vertical_position_weight: 0.3
    overlap_threshold: 0.1
  edge_cropping:
    edge_threshold_pixels: 10
    severity_edge_weight: 0.25
  subject_identification:
    area_weight: 0.3
    depth_weight: 0.7
  classification:
    strong_size_dominance_threshold: 4.0
  size_ratio_for_single_horse: 2.2
```
**Expected Result:** All configuration sections properly loaded and accessible

### Test Case 2: Missing Config File
**Expected Result:** `FileNotFoundError` raised when config file not found

---

## 🔢 Algorithm Parameters Summary

| Algorithm | Key Parameters | Weights |
|-----------|---------------|---------|
| **Depth Analysis** | `vertical_position_weight: 0.3`<br>`center_position_weight: 0.1`<br>`occlusion_boost_weight: 0.2` | Vertical position most important |
| **Edge Cropping** | `edge_threshold_pixels: 10`<br>`large_object_width_threshold: 0.7`<br>`severity_edge_weight: 0.25` | Size-dependent thresholds |
| **Subject ID** | `area_weight: 0.3`<br>`depth_weight: 0.7`<br>`edge_penalty_factor: 0.5` | Depth dominates over size |
| **Classification** | `strong_size_dominance_threshold: 4.0`<br>`depth_dominance_threshold: 0.1` | Size ratio critical for decisions |

This comprehensive test suite ensures robust horse detection across all scenarios: single horses, multiple horses, edge cases, cropping situations, depth relationships, and configuration handling! 🐎