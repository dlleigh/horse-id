#!/usr/bin/env python3
"""
Configuration utilities for the Horse ID system.

Provides flexible configuration loading with support for:
- Environment variable overrides
- Local configuration file overlays
- Path expansion and validation
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_file: str = 'config.yml', local_config_file: str = 'config.local.yml') -> Dict[str, Any]:
    """
    Load configuration with support for local overrides and environment variables.
    
    Loading order (later sources override earlier ones):
    1. Base config.yml
    2. config.local.yml (if exists)
    3. HORSE_ID_DATA_ROOT environment variable (overrides paths.data_root)
    
    Args:
        config_file: Path to base configuration file
        local_config_file: Path to local configuration override file
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If base config file doesn't exist
        yaml.YAMLError: If config files contain invalid YAML
    """
    # Load base configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Base configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Overlay local configuration if it exists
    if os.path.exists(local_config_file):
        with open(local_config_file, 'r') as f:
            local_config = yaml.safe_load(f)
            if local_config:
                config = _deep_merge(config, local_config)
    
    # Apply environment variable overrides
    env_data_root = os.environ.get('HORSE_ID_DATA_ROOT')
    if env_data_root:
        if 'paths' not in config:
            config['paths'] = {}
        config['paths']['data_root'] = env_data_root
    
    # Expand user paths
    if 'paths' in config and 'data_root' in config['paths']:
        config['paths']['data_root'] = os.path.expanduser(config['paths']['data_root'])
    
    return config


def get_data_root(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the data root directory with all overrides applied.
    
    Args:
        config: Optional pre-loaded config dict. If None, loads config fresh.
        
    Returns:
        Expanded data root path
    """
    if config is None:
        config = load_config()
    
    return config['paths']['data_root']


def get_config_value(key_path: str, config: Optional[Dict[str, Any]] = None, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation (e.g., 'detection.confidence_threshold').
    
    Args:
        key_path: Dot-separated path to configuration value
        config: Optional pre-loaded config dict. If None, loads config fresh.
        default: Default value if key path not found
        
    Returns:
        Configuration value or default
    """
    if config is None:
        config = load_config()
    
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def validate_data_root(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate that the data root directory exists and is accessible.
    
    Args:
        config: Optional pre-loaded config dict. If None, loads config fresh.
        
    Returns:
        True if data root is valid, False otherwise
    """
    try:
        data_root = get_data_root(config)
        return os.path.isdir(data_root) and os.access(data_root, os.R_OK | os.W_OK)
    except Exception:
        return False


def create_data_directories(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Create all configured data directories if they don't exist.
    
    Args:
        config: Optional pre-loaded config dict. If None, loads config fresh.
        
    Raises:
        OSError: If directories cannot be created
    """
    if config is None:
        config = load_config()
    
    data_root = get_data_root(config)
    
    # Create data root
    os.makedirs(data_root, exist_ok=True)
    
    # Create other configured directories
    paths = config.get('paths', {})
    for key, path_template in paths.items():
        if key != 'data_root' and isinstance(path_template, str):
            # Expand template with data_root
            expanded_path = path_template.format(data_root=data_root)
            
            # Create directory if it's a directory path (not a file)
            if key.endswith('_dir') or key in ['dataset_dir', 'calibration_dir', 'temp_dir', 'features_dir']:
                os.makedirs(expanded_path, exist_ok=True)
            else:
                # For file paths, create parent directory
                parent_dir = os.path.dirname(expanded_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with overlay values taking precedence.
    
    Args:
        base: Base dictionary
        overlay: Dictionary to overlay on base
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_paths_from_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Get all file paths from config with data_root expansion applied.
    
    Args:
        config: Optional pre-loaded config dict. If None, loads config fresh.
        
    Returns:
        Dictionary of expanded file paths
    """
    if config is None:
        config = load_config()
    
    data_root = get_data_root(config)
    paths = config.get('paths', {})
    expanded_paths = {}
    
    for key, path_template in paths.items():
        if isinstance(path_template, str):
            expanded_paths[key] = path_template.format(data_root=data_root)
        else:
            expanded_paths[key] = path_template
    
    return expanded_paths


# Convenience function for backward compatibility
def load_and_expand_config() -> tuple[Dict[str, Any], str]:
    """
    Load config and return both config dict and expanded data_root.
    
    Returns:
        Tuple of (config_dict, data_root_path)
    """
    config = load_config()
    data_root = get_data_root(config)
    return config, data_root