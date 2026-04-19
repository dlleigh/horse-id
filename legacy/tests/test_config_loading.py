import pytest
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, mock_open
import sys

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_utils


class TestConfigUtils:
    """Test configuration utilities functionality."""
    
    def test_load_config_success(self, sample_config):
        """Test successful configuration loading."""
        with patch('config_utils.os.path.exists', return_value=True):
            with patch('config_utils.open', mock_open(read_data='test')):
                with patch('config_utils.yaml.safe_load', return_value=sample_config):
                    config = config_utils.load_config()
        
        assert config == sample_config
        assert 'paths' in config
        assert 'detection' in config
        assert 'similarity' in config
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with missing base config file."""
        with patch('config_utils.os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match='Base configuration file not found'):
                config_utils.load_config()
    
    def test_load_config_with_local_override(self, sample_config):
        """Test configuration loading with local config override."""
        local_config = {
            'paths': {
                'data_root': '/custom/path'
            },
            'detection': {
                'confidence_threshold': 0.7
            }
        }
        
        expected_config = sample_config.copy()
        expected_config['paths']['data_root'] = '/custom/path'
        expected_config['detection']['confidence_threshold'] = 0.7
        
        def mock_exists(path):
            return True  # Both config.yml and config.local.yml exist
        
        def mock_open_files(filename, mode='r'):
            if 'config.local.yml' in filename:
                return mock_open(read_data='local')()
            else:
                return mock_open(read_data='base')()
        
        def mock_yaml_load(f):
            if f.read() == 'local':
                return local_config
            else:
                return sample_config
        
        with patch('config_utils.os.path.exists', side_effect=mock_exists):
            with patch('config_utils.open', side_effect=mock_open_files):
                with patch('config_utils.yaml.safe_load', side_effect=mock_yaml_load):
                    config = config_utils.load_config()
        
        assert config['paths']['data_root'] == '/custom/path'
        assert config['detection']['confidence_threshold'] == 0.7
        # Original values should still be present
        assert config['similarity']['merge_threshold'] == sample_config['similarity']['merge_threshold']
    
    def test_load_config_with_env_override(self, sample_config):
        """Test configuration loading with environment variable override."""
        with patch('config_utils.os.path.exists', return_value=True):
            with patch('config_utils.open', mock_open(read_data='test')):
                with patch('config_utils.yaml.safe_load', return_value=sample_config):
                    with patch.dict('config_utils.os.environ', {'HORSE_ID_DATA_ROOT': '/env/override/path'}):
                        config = config_utils.load_config()
        
        assert config['paths']['data_root'] == '/env/override/path'
    
    def test_load_config_yaml_parse_error(self):
        """Test configuration loading with YAML parsing error."""
        with patch('config_utils.os.path.exists', return_value=True):
            with patch('config_utils.open', mock_open(read_data='invalid: yaml: content:')):
                with patch('config_utils.yaml.safe_load', side_effect=yaml.YAMLError('Invalid YAML')):
                    with pytest.raises(yaml.YAMLError):
                        config_utils.load_config()
    
    def test_get_data_root(self, sample_config):
        """Test getting data root from config."""
        with patch('config_utils.os.path.expanduser') as mock_expanduser:
            mock_expanduser.return_value = '/expanded/path'
            data_root = config_utils.get_data_root(sample_config)
            mock_expanduser.assert_called_once_with(sample_config['paths']['data_root'])
            assert data_root == '/expanded/path'
    
    def test_get_data_root_with_env_override(self, sample_config):
        """Test getting data root with environment variable override."""
        with patch.dict('config_utils.os.environ', {'HORSE_ID_DATA_ROOT': '/env/path'}):
            with patch('config_utils.os.path.expanduser') as mock_expanduser:
                mock_expanduser.return_value = '/expanded/env/path'
                config = config_utils.load_config()
                data_root = config_utils.get_data_root(config)
                mock_expanduser.assert_called_with('/env/path')
                assert data_root == '/expanded/env/path'
    
    def test_get_config_value(self, sample_config):
        """Test getting configuration values using dot notation."""
        # Test existing path
        value = config_utils.get_config_value('detection.confidence_threshold', sample_config)
        assert value == 0.5
        
        # Test nested path
        value = config_utils.get_config_value('detection.depth_analysis.vertical_position_weight', sample_config)
        assert value == 0.3
        
        # Test non-existent path with default
        value = config_utils.get_config_value('non.existent.path', sample_config, default='default_value')
        assert value == 'default_value'
        
        # Test non-existent path without default
        value = config_utils.get_config_value('non.existent.path', sample_config)
        assert value is None
    
    def test_validate_data_root(self, sample_config):
        """Test data root validation."""
        with patch('config_utils.os.path.isdir', return_value=True):
            with patch('config_utils.os.access', return_value=True):
                assert config_utils.validate_data_root(sample_config) is True
        
        # Test non-existent directory
        with patch('config_utils.os.path.isdir', return_value=False):
            assert config_utils.validate_data_root(sample_config) is False
        
        # Test insufficient permissions
        with patch('config_utils.os.path.isdir', return_value=True):
            with patch('config_utils.os.access', return_value=False):
                assert config_utils.validate_data_root(sample_config) is False
    
    def test_create_data_directories(self, sample_config):
        """Test creating data directories."""
        with patch('config_utils.os.makedirs') as mock_makedirs:
            with patch('config_utils.get_data_root', return_value='/test/data'):
                config_utils.create_data_directories(sample_config)
                
                # Should create data root
                mock_makedirs.assert_any_call('/test/data', exist_ok=True)
                
                # Should create other directories
                mock_makedirs.assert_any_call('/test/data/calibrations', exist_ok=True)
                mock_makedirs.assert_any_call('/test/data/tmp', exist_ok=True)
                mock_makedirs.assert_any_call('/test/data/features', exist_ok=True)
    
    def test_get_paths_from_config(self, sample_config):
        """Test getting all paths with expansion."""
        with patch('config_utils.get_data_root', return_value='/test/data'):
            paths = config_utils.get_paths_from_config(sample_config)
            
            assert paths['data_root'] == '/test/data'
            assert paths['dataset_dir'] == '/test/data/horse_photos'
            assert paths['manifest_file'] == '/test/data/horse_photos_manifest.csv'
            assert paths['calibration_dir'] == '/test/data/calibrations'
    
    def test_deep_merge(self):
        """Test deep merging of dictionaries."""
        base = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            },
            'e': 4
        }
        
        overlay = {
            'b': {
                'c': 20,  # Override
                'f': 6    # New key
            },
            'g': 7        # New top-level key
        }
        
        result = config_utils._deep_merge(base, overlay)
        
        assert result['a'] == 1      # Unchanged
        assert result['b']['c'] == 20  # Overridden
        assert result['b']['d'] == 3   # Unchanged
        assert result['b']['f'] == 6   # New
        assert result['e'] == 4      # Unchanged
        assert result['g'] == 7      # New
    
    def test_load_and_expand_config(self, sample_config):
        """Test backward compatibility function."""
        with patch('config_utils.load_config', return_value=sample_config):
            with patch('config_utils.get_data_root', return_value='/expanded/path'):
                config, data_root = config_utils.load_and_expand_config()
                
                assert config == sample_config
                assert data_root == '/expanded/path'


class TestConfigIntegration:
    """Integration tests for configuration loading."""
    
    def test_real_config_file_loading(self):
        """Test loading actual config files (if they exist)."""
        # This test only runs if config.yml actually exists
        if os.path.exists('config.yml'):
            config = config_utils.load_config()
            assert 'paths' in config
            assert 'data_root' in config['paths']
            assert isinstance(config['paths']['data_root'], str)
        else:
            pytest.skip("config.yml not found - skipping integration test")
    
    def test_environment_variable_override_integration(self):
        """Test that environment variables actually override config."""
        if os.path.exists('config.yml'):
            # Test with environment variable
            with patch.dict(os.environ, {'HORSE_ID_DATA_ROOT': '/integration/test/path'}):
                config = config_utils.load_config()
                assert config['paths']['data_root'] == '/integration/test/path'
        else:
            pytest.skip("config.yml not found - skipping integration test")