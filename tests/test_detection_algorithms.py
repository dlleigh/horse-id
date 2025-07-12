import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from horse_detection_lib import (
    calculate_bbox_overlap,
    calculate_distance_from_center,
    analyze_depth_relationships,
    is_edge_cropped,
    identify_subject_horse,
    classify_horse_detection,
    load_detection_config
)


class TestBboxOverlap:
    """Test bounding box overlap calculations."""
    
    def test_no_overlap(self):
        """Test bounding boxes with no overlap."""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([200, 200, 300, 300])
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 0.0
    
    def test_complete_overlap(self):
        """Test completely overlapping bounding boxes."""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([0, 0, 100, 100])
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 1.0
    
    def test_partial_overlap(self):
        """Test partially overlapping bounding boxes."""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([50, 50, 150, 150])
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        # Intersection is 50x50 = 2500, bbox2 area is 100x100 = 10000
        expected = 2500 / 10000
        assert overlap == expected
    
    def test_bbox2_inside_bbox1(self):
        """Test bbox2 completely inside bbox1."""
        bbox1 = np.array([0, 0, 200, 200])
        bbox2 = np.array([50, 50, 150, 150])
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 1.0
    
    def test_edge_case_zero_area(self):
        """Test edge case with zero area bbox."""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([50, 50, 50, 50])  # Zero area
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 0.0


class TestDistanceFromCenter:
    """Test distance from center calculations."""
    
    def test_centered_bbox(self):
        """Test bbox at image center."""
        bbox = np.array([225, 190, 50, 100])  # Center at (250, 240)
        distance = calculate_distance_from_center(bbox, 500, 480)
        assert distance == 0.0
    
    def test_corner_bbox(self):
        """Test bbox at image corner."""
        bbox = np.array([0, 0, 50, 50])  # Center at (25, 25)
        distance = calculate_distance_from_center(bbox, 500, 480)
        # Distance from (25, 25) to (250, 240) normalized by diagonal
        img_diagonal = np.sqrt(500**2 + 480**2)
        expected_distance = np.sqrt((25 - 250)**2 + (25 - 240)**2) / img_diagonal
        assert abs(distance - expected_distance) < 1e-10
    
    def test_bbox_format_xywh(self):
        """Test with different bbox format interpretation."""
        bbox = np.array([100, 100, 200, 200])  # x, y, w, h
        distance = calculate_distance_from_center(bbox, 400, 400)
        # Center should be at (100 + 200/2, 100 + 200/2) = (200, 200)
        # Image center is (200, 200), so distance should be 0
        assert distance == 0.0


class TestEdgeCropping:
    """Test edge cropping detection."""
    
    def test_not_cropped(self):
        """Test bbox not touching any edges."""
        bbox = np.array([100, 100, 200, 200])  # Well inside image
        is_cropped, severity = is_edge_cropped(bbox, 400, 400)
        assert not is_cropped
        assert severity == 0.0
    
    def test_single_edge_touch(self):
        """Test bbox touching single edge."""
        bbox = np.array([0, 100, 100, 200])  # Touching left edge
        config = {
            'edge_threshold_pixels': 10,
            'severity_edge_weight': 0.25,
            'large_object_width_threshold': 0.7,
            'large_object_height_threshold': 0.7,
            'severity_large_object_weight': 0.3,
            'close_margin_threshold': 5,
            'severity_close_margin_weight': 0.2
        }
        is_cropped, severity = is_edge_cropped(bbox, 400, 400, config)
        assert not is_cropped  # Single edge touch with small object
        assert severity > 0.0
    
    def test_multiple_edge_touch(self):
        """Test bbox touching multiple edges."""
        bbox = np.array([0, 0, 100, 100])  # Touching top and left edges
        config = {
            'edge_threshold_pixels': 10,
            'severity_edge_weight': 0.25,
            'large_object_width_threshold': 0.7,
            'large_object_height_threshold': 0.7,
            'severity_large_object_weight': 0.3,
            'close_margin_threshold': 5,
            'severity_close_margin_weight': 0.2
        }
        is_cropped, severity = is_edge_cropped(bbox, 400, 400, config)
        assert is_cropped  # Multiple edge touches
        assert severity > 0.0
    
    def test_large_object_single_edge(self):
        """Test large object touching single edge."""
        bbox = np.array([0, 50, 300, 200])  # Large object touching left edge
        config = {
            'edge_threshold_pixels': 10,
            'severity_edge_weight': 0.25,
            'large_object_width_threshold': 0.7,
            'large_object_height_threshold': 0.7,
            'severity_large_object_weight': 0.3,
            'close_margin_threshold': 5,
            'severity_close_margin_weight': 0.2
        }
        is_cropped, severity = is_edge_cropped(bbox, 400, 400, config)
        assert is_cropped  # Large object touching edge
        assert severity > 0.0


class TestDepthAnalysis:
    """Test depth relationship analysis."""
    
    def test_single_horse_depth(self):
        """Test depth analysis with single horse."""
        mock_boxes = Mock()
        mock_boxes.xyxy = [Mock()]
        mock_boxes.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 200, 200])
        
        mock_masks = Mock()
        horse_indices = np.array([0])
        
        config = {
            'vertical_position_weight': 0.3,
            'overlap_threshold': 0.1,
            'occlusion_boost_weight': 0.2,
            'occlusion_penalty_weight': 0.1,
            'perspective_size_threshold': 0.1,
            'perspective_score_boost': 0.1,
            'center_position_weight': 0.1
        }
        
        depth_scores = analyze_depth_relationships(
            mock_boxes, mock_masks, horse_indices, 400, 400, config
        )
        
        assert len(depth_scores) == 1
        assert depth_scores[0] > 0  # Should have positive depth score
    
    def test_multiple_horses_depth(self):
        """Test depth analysis with multiple horses."""
        mock_boxes = Mock()
        mock_boxes.xyxy = [Mock(), Mock()]
        # Horse 1 much lower in image AND centered (should have higher depth score)
        mock_boxes.xyxy[0].cpu.return_value.numpy.return_value = np.array([175, 300, 225, 380])  # Bottom center
        # Horse 2 higher in image and off-center
        mock_boxes.xyxy[1].cpu.return_value.numpy.return_value = np.array([50, 50, 100, 100])   # Top left
        
        mock_masks = Mock()
        horse_indices = np.array([0, 1])
        
        config = {
            'vertical_position_weight': 0.3,
            'overlap_threshold': 0.1,
            'occlusion_boost_weight': 0.2,
            'occlusion_penalty_weight': 0.1,
            'perspective_size_threshold': 0.1,
            'perspective_score_boost': 0.1,
            'center_position_weight': 0.1
        }
        
        depth_scores = analyze_depth_relationships(
            mock_boxes, mock_masks, horse_indices, 400, 400, config
        )
        
        assert len(depth_scores) == 2
        # Horse at bottom center should have significantly higher score than horse at top left
        assert depth_scores[0] > depth_scores[1]


class TestSubjectIdentification:
    """Test subject horse identification."""
    
    def test_identify_largest_horse(self):
        """Test identifying the largest horse as subject."""
        mock_boxes = Mock()
        mock_boxes.xyxy = [Mock(), Mock()]
        mock_boxes.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 200, 200])
        mock_boxes.xyxy[1].cpu.return_value.numpy.return_value = np.array([300, 300, 350, 350])
        
        mock_masks = Mock()
        horse_indices = np.array([0, 1])
        areas = np.array([10000, 2500])  # First horse is much larger
        
        config = {
            'area_weight': 0.3,
            'depth_weight': 0.7,
            'edge_penalty_factor': 0.5,
            'vertical_position_weight': 0.3,
            'overlap_threshold': 0.1,
            'occlusion_boost_weight': 0.2,
            'occlusion_penalty_weight': 0.1,
            'perspective_size_threshold': 0.1,
            'perspective_score_boost': 0.1,
            'center_position_weight': 0.1,
            'edge_threshold_pixels': 10,
            'severity_edge_weight': 0.25,
            'large_object_width_threshold': 0.7,
            'large_object_height_threshold': 0.7,
            'severity_large_object_weight': 0.3,
            'close_margin_threshold': 5,
            'severity_close_margin_weight': 0.2
        }
        
        subject_idx = identify_subject_horse(
            mock_boxes, mock_masks, horse_indices, areas, 400, 400, config
        )
        
        assert subject_idx == 0  # Should identify the larger horse


class TestClassificationPipeline:
    """Test complete horse detection classification."""
    
    def test_single_horse_classification(self):
        """Test classification with single horse."""
        mock_boxes = Mock()
        mock_masks = Mock()
        horse_indices = np.array([0])
        areas = np.array([10000])
        
        classification, size_ratio, subject_idx, analysis = classify_horse_detection(
            mock_boxes, mock_masks, horse_indices, areas, 400, 400
        )
        
        assert classification == "SINGLE"
        assert np.isnan(size_ratio)
        assert subject_idx == 0
        assert analysis['final_classification'] == 'SINGLE'
        assert 'Only one horse detected' in analysis['reason']
    
    @patch('horse_detection_lib.load_detection_config')
    def test_multiple_horses_classification(self, mock_load_config):
        """Test classification with multiple horses."""
        mock_load_config.return_value = {
            'depth_analysis': {
                'vertical_position_weight': 0.3,
                'overlap_threshold': 0.1,
                'occlusion_boost_weight': 0.2,
                'occlusion_penalty_weight': 0.1,
                'perspective_size_threshold': 0.1,
                'perspective_score_boost': 0.1,
                'center_position_weight': 0.1
            },
            'edge_cropping': {
                'edge_threshold_pixels': 10,
                'severity_edge_weight': 0.25,
                'large_object_width_threshold': 0.7,
                'large_object_height_threshold': 0.7,
                'severity_large_object_weight': 0.3,
                'close_margin_threshold': 5,
                'severity_close_margin_weight': 0.2
            },
            'subject_identification': {
                'area_weight': 0.3,
                'depth_weight': 0.7,
                'edge_penalty_factor': 0.5
            },
            'classification': {
                'strong_size_dominance_threshold': 4.0,
                'depth_dominance_threshold': 0.1,
                'edge_advantage_significant_threshold': 0.3,
                'edge_significant_scaling_factor': 2.0,
                'edge_significant_size_reduction': 0.3,
                'edge_significant_depth_reduction': 0.5,
                'edge_advantage_moderate_threshold': 0.15,
                'edge_moderate_size_factor': 0.8,
                'edge_moderate_depth_threshold': 0.05,
                'extreme_overlap_threshold': 0.8
            },
            'size_ratio': 2.2
        }
        
        mock_boxes = Mock()
        mock_boxes.xyxy = [Mock(), Mock()]
        mock_boxes.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 200, 200])
        mock_boxes.xyxy[1].cpu.return_value.numpy.return_value = np.array([300, 300, 350, 350])
        
        mock_masks = Mock()
        horse_indices = np.array([0, 1])
        areas = np.array([10000, 2000])  # First horse is 5x larger
        
        classification, size_ratio, subject_idx, analysis = classify_horse_detection(
            mock_boxes, mock_masks, horse_indices, areas, 400, 400
        )
        
        assert classification in ["SINGLE", "MULTIPLE"]
        assert not np.isnan(size_ratio)
        assert size_ratio == 5.0
        assert subject_idx in [0, 1]
        assert 'final_classification' in analysis


class TestConfigLoading:
    """Test configuration loading."""
    
    @patch('horse_detection_lib.open')
    @patch('horse_detection_lib.yaml.safe_load')
    def test_load_detection_config_success(self, mock_yaml_load, mock_open):
        """Test successful config loading."""
        mock_config = {
            'detection': {
                'depth_analysis': {'test': 'value'},
                'edge_cropping': {'test': 'value'},
                'subject_identification': {'test': 'value'},
                'classification': {'test': 'value'},
                'size_ratio_for_single_horse': 2.2
            }
        }
        mock_yaml_load.return_value = mock_config
        
        config = load_detection_config()
        
        assert 'depth_analysis' in config
        assert 'edge_cropping' in config
        assert 'subject_identification' in config
        assert 'classification' in config
        assert 'size_ratio' in config
        assert config['size_ratio'] == 2.2
    
    @patch('horse_detection_lib.open', side_effect=FileNotFoundError)
    def test_load_detection_config_file_not_found(self, mock_open):
        """Test config loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_detection_config()