import pytest
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, mock_open
import sys

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from horse_id import load_config, setup_paths
from horse_detection_lib import load_detection_config


class TestConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_config_success(self, sample_config):
        """Test successful configuration loading."""
        with patch('horse_id.os.path.exists', return_value=True):
            with patch('horse_id.open', mock_open(read_data='test')):
                with patch('horse_id.yaml.safe_load', return_value=sample_config):
                    config = load_config()
        
        assert config == sample_config
        assert 'paths' in config
        assert 'detection' in config
        assert 'similarity' in config
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with missing file."""
        with patch('horse_id.os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match='Configuration file.*not found'):
                load_config()
    
    def test_load_config_yaml_parse_error(self):
        """Test configuration loading with YAML parsing error."""
        with patch('horse_id.os.path.exists', return_value=True):
            with patch('horse_id.open', mock_open(read_data='invalid: yaml: content:')):
                with patch('horse_id.yaml.safe_load', side_effect=yaml.YAMLError('Invalid YAML')):
                    with pytest.raises(yaml.YAMLError):
                        load_config()
    
    def test_load_config_file_read_error(self):
        """Test configuration loading with file read error."""
        with patch('horse_id.os.path.exists', return_value=True):
            with patch('horse_id.open', side_effect=IOError('Permission denied')):
                with pytest.raises(IOError):
                    load_config()
    
    def test_load_config_empty_file(self):
        """Test configuration loading with empty file."""
        with patch('horse_id.os.path.exists', return_value=True):
            with patch('horse_id.open', mock_open(read_data='')):
                with patch('horse_id.yaml.safe_load', return_value=None):
                    config = load_config()
        
        assert config is None


class TestDetectionConfigLoading:
    """Test detection configuration loading functionality."""
    
    def test_load_detection_config_success(self, sample_config):
        """Test successful detection configuration loading."""
        with patch('horse_detection_lib.open', mock_open(read_data='test')):
            with patch('horse_detection_lib.yaml.safe_load', return_value=sample_config):
                config = load_detection_config()
        
        assert 'depth_analysis' in config
        assert 'edge_cropping' in config
        assert 'subject_identification' in config
        assert 'classification' in config
        assert 'size_ratio' in config
        assert config['size_ratio'] == 2.2
    
    def test_load_detection_config_file_not_found(self):
        """Test detection configuration loading with missing file."""
        with patch('horse_detection_lib.open', side_effect=FileNotFoundError('File not found')):
            with pytest.raises(FileNotFoundError):
                load_detection_config()
    
    def test_load_detection_config_yaml_error(self):
        """Test detection configuration loading with YAML error."""
        with patch('horse_detection_lib.open', mock_open(read_data='invalid: yaml:')):
            with patch('horse_detection_lib.yaml.safe_load', side_effect=yaml.YAMLError('Invalid YAML')):
                with pytest.raises(yaml.YAMLError):
                    load_detection_config()
    
    def test_load_detection_config_missing_keys(self):
        """Test detection configuration loading with missing keys."""
        incomplete_config = {
            'detection': {
                'depth_analysis': {'test': 'value'}
                # Missing other required keys
            }
        }
        
        with patch('horse_detection_lib.open', mock_open(read_data='test')):
            with patch('horse_detection_lib.yaml.safe_load', return_value=incomplete_config):
                # Should raise KeyError when required keys are missing
                with pytest.raises(KeyError):
                    load_detection_config()


class TestPathSetup:
    """Test path setup functionality."""
    
    def test_setup_paths_success(self, sample_config):
        """Test successful path setup."""
        with patch('horse_id.os.path.expanduser', side_effect=lambda x: x):
            manifest_file, features_dir, bucket_name = setup_paths(sample_config)
        
        assert manifest_file == '/tmp/test_data/merged_manifest.csv'
        assert features_dir == '/tmp/test_data/features'
        assert bucket_name == 'test-bucket'
    
    def test_setup_paths_with_env_override(self, sample_config):
        """Test path setup with environment variable override."""
        with patch.dict(os.environ, {'HORSE_ID_DATA_ROOT': '/env/override'}):
            with patch('horse_id.os.path.expanduser', side_effect=lambda x: x):
                manifest_file, features_dir, bucket_name = setup_paths(sample_config)
        
        assert manifest_file == '/env/override/merged_manifest.csv'
        assert features_dir == '/env/override/features'
        assert bucket_name == 'test-bucket'
    
    def test_setup_paths_home_directory_expansion(self, sample_config):
        """Test path setup with home directory expansion."""
        sample_config['paths']['data_root'] = '~/test_data'
        
        with patch('horse_id.os.path.expanduser', return_value='/home/user/test_data'):
            manifest_file, features_dir, bucket_name = setup_paths(sample_config)
        
        assert manifest_file == '/home/user/test_data/merged_manifest.csv'
        assert features_dir == '/home/user/test_data/features'
        assert bucket_name == 'test-bucket'
    
    def test_setup_paths_missing_paths_section(self):
        """Test path setup with missing paths section."""
        config = {
            's3': {'bucket_name': 'test-bucket'}
            # Missing paths section
        }
        
        with pytest.raises(ValueError, match='Missing path configuration'):
            setup_paths(config)
    
    def test_setup_paths_missing_s3_section(self):
        """Test path setup with missing S3 section."""
        config = {
            'paths': {
                'data_root': '/tmp/test',
                'merged_manifest_file': '{data_root}/merged_manifest.csv',
                'features_dir': '{data_root}/features'
            }
            # Missing s3 section
        }
        
        with pytest.raises(ValueError, match='Missing path configuration'):
            setup_paths(config)
    
    def test_setup_paths_missing_manifest_file(self, sample_config):
        """Test path setup with missing manifest file configuration."""
        del sample_config['paths']['merged_manifest_file']
        
        with pytest.raises(ValueError, match='Missing path configuration'):
            setup_paths(sample_config)
    
    def test_setup_paths_missing_features_dir(self, sample_config):
        """Test path setup with missing features directory configuration."""
        del sample_config['paths']['features_dir']
        
        with pytest.raises(ValueError, match='Missing path configuration'):
            setup_paths(sample_config)
    
    def test_setup_paths_missing_bucket_name(self, sample_config):
        """Test path setup with missing bucket name configuration."""
        del sample_config['s3']['bucket_name']
        
        with pytest.raises(ValueError, match='Missing path configuration'):
            setup_paths(sample_config)
    
    def test_setup_paths_format_error(self, sample_config):
        """Test path setup with string formatting error."""
        sample_config['paths']['merged_manifest_file'] = '{missing_var}/merged_manifest.csv'
        
        with pytest.raises(ValueError, match='Missing path configuration'):
            setup_paths(sample_config)
    
    def test_setup_paths_expanduser_error(self, sample_config):
        """Test path setup with expanduser error."""
        with patch('horse_id.os.path.expanduser', side_effect=Exception('Expansion error')):
            with pytest.raises(RuntimeError, match='Error setting up paths'):
                setup_paths(sample_config)


class TestConfigValidation:
    """Test configuration validation functionality."""
    
    def test_config_structure_validation(self, sample_config):
        """Test that configuration has required structure."""
        # Test that all required top-level keys exist
        required_keys = ['paths', 'detection', 'similarity', 's3', 'twilio', 'gmail']
        
        for key in required_keys:
            assert key in sample_config, f"Missing required config key: {key}"
    
    def test_paths_config_validation(self, sample_config):
        """Test that paths configuration has required keys."""
        paths_config = sample_config['paths']
        required_keys = [
            'data_root', 'dataset_dir', 'manifest_file', 'merged_manifest_file',
            'features_dir', 'calibration_dir', 'merge_results_file', 'temp_dir'
        ]
        
        for key in required_keys:
            assert key in paths_config, f"Missing required paths config key: {key}"
    
    def test_detection_config_validation(self, sample_config):
        """Test that detection configuration has required keys."""
        detection_config = sample_config['detection']
        required_keys = [
            'yolo_model', 'confidence_threshold', 'detected_manifest_file',
            'depth_analysis', 'edge_cropping', 'subject_identification',
            'classification', 'size_ratio_for_single_horse'
        ]
        
        for key in required_keys:
            assert key in detection_config, f"Missing required detection config key: {key}"
    
    def test_similarity_config_validation(self, sample_config):
        """Test that similarity configuration has required keys."""
        similarity_config = sample_config['similarity']
        required_keys = ['merge_threshold', 'inference_threshold', 'master_horse_location_file']
        
        for key in required_keys:
            assert key in similarity_config, f"Missing required similarity config key: {key}"
    
    def test_s3_config_validation(self, sample_config):
        """Test that S3 configuration has required keys."""
        s3_config = sample_config['s3']
        required_keys = ['bucket_name']
        
        for key in required_keys:
            assert key in s3_config, f"Missing required S3 config key: {key}"
    
    def test_twilio_config_validation(self, sample_config):
        """Test that Twilio configuration has required keys."""
        twilio_config = sample_config['twilio']
        required_keys = ['account_sid', 'auth_token']
        
        for key in required_keys:
            assert key in twilio_config, f"Missing required Twilio config key: {key}"
    
    def test_gmail_config_validation(self, sample_config):
        """Test that Gmail configuration has required keys."""
        gmail_config = sample_config['gmail']
        required_keys = ['token_file', 'credentials_file', 'scopes']
        
        for key in required_keys:
            assert key in gmail_config, f"Missing required Gmail config key: {key}"
    
    def test_config_data_types(self, sample_config):
        """Test that configuration values have correct data types."""
        # Test numeric values
        assert isinstance(sample_config['detection']['confidence_threshold'], float)
        assert isinstance(sample_config['detection']['size_ratio_for_single_horse'], float)
        assert isinstance(sample_config['similarity']['merge_threshold'], float)
        assert isinstance(sample_config['similarity']['inference_threshold'], float)
        
        # Test string values
        assert isinstance(sample_config['paths']['data_root'], str)
        assert isinstance(sample_config['detection']['yolo_model'], str)
        assert isinstance(sample_config['s3']['bucket_name'], str)
        
        # Test list values
        assert isinstance(sample_config['gmail']['scopes'], list)
        assert len(sample_config['gmail']['scopes']) > 0
    
    def test_config_value_ranges(self, sample_config):
        """Test that configuration values are within valid ranges."""
        # Test confidence threshold is between 0 and 1
        conf_threshold = sample_config['detection']['confidence_threshold']
        assert 0.0 <= conf_threshold <= 1.0, f"Confidence threshold {conf_threshold} not in valid range [0.0, 1.0]"
        
        # Test similarity thresholds are between 0 and 1
        merge_threshold = sample_config['similarity']['merge_threshold']
        assert 0.0 <= merge_threshold <= 1.0, f"Merge threshold {merge_threshold} not in valid range [0.0, 1.0]"
        
        inference_threshold = sample_config['similarity']['inference_threshold']
        assert 0.0 <= inference_threshold <= 1.0, f"Inference threshold {inference_threshold} not in valid range [0.0, 1.0]"
        
        # Test size ratio is positive
        size_ratio = sample_config['detection']['size_ratio_for_single_horse']
        assert size_ratio > 0.0, f"Size ratio {size_ratio} must be positive"