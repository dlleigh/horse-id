import pytest
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import yaml
import sys

# Mock ML/CV dependencies that might not be available in test environment
class MockModule:
    def __getattr__(self, name):
        return Mock()

# Mock heavy ML dependencies to avoid installation conflicts
ML_MODULES = [
    'torch', 'torchvision', 'timm', 'ultralytics', 
    'wildlife_tools', 'wildlife_datasets', 'kornia',
    'albumentations', 'cv2', 'sklearn'
]

for module_name in ML_MODULES:
    if module_name not in sys.modules:
        sys.modules[module_name] = MockModule()
        # Mock common submodules
        sys.modules[f'{module_name}.transforms'] = MockModule()
        sys.modules[f'{module_name}.models'] = MockModule()
        sys.modules[f'{module_name}.inference'] = MockModule()
        sys.modules[f'{module_name}.features'] = MockModule()
        sys.modules[f'{module_name}.similarity'] = MockModule()
        sys.modules[f'{module_name}.data'] = MockModule()
        # Mock deeper nested modules for wildlife_tools
        if module_name == 'wildlife_tools':
            sys.modules[f'{module_name}.similarity.calibration'] = MockModule()
            sys.modules[f'{module_name}.similarity.wildfusion'] = MockModule()


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'paths': {
            'data_root': '/tmp/test_data',
            'dataset_dir': '{data_root}/dataset',
            'manifest_file': '{data_root}/manifest.csv',
            'merged_manifest_file': '{data_root}/merged_manifest.csv',
            'features_dir': '{data_root}/features',
            'calibration_dir': '{data_root}/calibration',
            'merge_results_file': '{data_root}/merge_results.csv',
            'temp_dir': '{data_root}/temp'
        },
        'detection': {
            'yolo_model': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'detected_manifest_file': '{data_root}/detected_manifest.csv',
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
            'size_ratio_for_single_horse': 2.2
        },
        'similarity': {
            'merge_threshold': 0.7,
            'inference_threshold': 0.6,
            'master_horse_location_file': '{data_root}/master_horses.xlsx'
        },
        's3': {
            'bucket_name': 'test-bucket'
        },
        'twilio': {
            'account_sid': 'test_sid',
            'auth_token': 'test_token'
        },
        'gmail': {
            'token_file': 'token.json',
            'credentials_file': 'credentials.json',
            'scopes': ['https://www.googleapis.com/auth/gmail.readonly']
        }
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(sample_config, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_yolo_results():
    """Mock YOLO detection results."""
    mock_result = Mock()
    mock_result.orig_shape = (480, 640)  # height, width
    
    # Mock boxes
    mock_boxes = Mock()
    mock_boxes.cls = Mock()
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([17, 17, 0])  # Two horses, one non-horse
    mock_boxes.xywh = Mock()
    mock_boxes.xywh.__getitem__ = Mock(return_value=Mock())
    mock_boxes.xyxy = Mock()
    mock_boxes.xywhn = Mock()
    mock_result.boxes = mock_boxes
    
    # Mock masks
    mock_masks = Mock()
    mock_masks.__len__ = Mock(return_value=3)
    mock_masks.xy = [
        np.array([[100, 100], [200, 100], [200, 200], [100, 200]]),  # Horse 1
        np.array([[300, 300], [400, 300], [400, 400], [300, 400]]),  # Horse 2
        np.array([[50, 50], [60, 50], [60, 60], [50, 60]])           # Non-horse
    ]
    mock_result.masks = mock_masks
    
    return [mock_result]


@pytest.fixture
def sample_manifest_df():
    """Sample manifest DataFrame for testing."""
    return pd.DataFrame({
        'filename': ['horse1.jpg', 'horse2.jpg', 'horse3.jpg'],
        'horse_name': ['Thunder', 'Lightning', 'Storm'],
        'email_date': ['20230101', '20230102', '20230103'],
        'status': ['ACTIVE', 'ACTIVE', 'EXCLUDE'],
        'num_horses_detected': ['SINGLE', 'SINGLE', 'MULTIPLE']
    })


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 S3 client."""
    mock_client = Mock()
    mock_client.download_file = Mock(return_value=None)
    return mock_client


@pytest.fixture
def mock_twilio_client():
    """Mock Twilio client."""
    mock_client = Mock()
    mock_message = Mock()
    mock_message.sid = 'test_message_sid'
    mock_client.messages.create.return_value = mock_message
    return mock_client


@pytest.fixture
def sample_horse_boxes():
    """Sample horse bounding boxes for testing."""
    mock_boxes = Mock()
    
    # Create mock xyxy tensor
    mock_xyxy = Mock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([100, 100, 200, 200])
    mock_boxes.xyxy = [mock_xyxy, mock_xyxy]
    
    # Create mock xywh tensor
    mock_xywh = Mock()
    mock_xywh.__getitem__ = Mock(return_value=Mock())
    mock_boxes.xywh = mock_xywh
    
    return mock_boxes


@pytest.fixture
def sample_horse_masks():
    """Sample horse masks for testing."""
    mock_masks = Mock()
    mock_masks.xy = [
        np.array([[100, 100], [200, 100], [200, 200], [100, 200]]),
        np.array([[300, 300], [400, 300], [400, 400], [300, 400]])
    ]
    return mock_masks