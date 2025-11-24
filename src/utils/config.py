"""
Configuration management utilities.

Provides functions to load, merge, and validate configuration files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_config(
    config_path: Optional[str] = None,
    default_config_path: str = "configs/default.yaml",
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration with support for defaults and overrides.
    
    Args:
        config_path: Path to experiment-specific config file
        default_config_path: Path to default config file
        overrides: Dictionary of override values
        
    Returns:
        Final merged configuration dictionary
    """
    # Load default config
    if os.path.exists(default_config_path):
        config = load_yaml(default_config_path)
        logger.info(f"Loaded default config from: {default_config_path}")
    else:
        config = {}
        logger.warning(f"Default config not found: {default_config_path}")
    
    # Load experiment-specific config and merge
    if config_path and os.path.exists(config_path):
        experiment_config = load_yaml(config_path)
        config = merge_configs(config, experiment_config)
        logger.info(f"Loaded experiment config from: {config_path}")
    
    # Apply command-line overrides
    if overrides:
        config = merge_configs(config, overrides)
        logger.info(f"Applied {len(overrides)} override(s)")
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save location
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved configuration to: {save_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['experiment', 'data', 'model', 'training', 'logging', 'hardware']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    # Validate data config
    if 'dataset_name' not in config['data']:
        raise ValueError("Missing required parameter: data.dataset_name")
    
    # Validate model config
    if 'name' not in config['model']:
        raise ValueError("Missing required parameter: model.name")
    
    # Validate training config
    if 'epochs' not in config['training']:
        raise ValueError("Missing required parameter: training.epochs")
    
    logger.info("Configuration validation passed")


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get value from nested configuration dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "training.learning_rate")
        default: Default value if key not found
        
    Returns:
        Value at key path or default
        
    Example:
        >>> config = {'training': {'learning_rate': 0.001}}
        >>> get_nested_value(config, 'training.learning_rate')
        0.001
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set value in nested configuration dictionary using dot notation.
    
    Args:
        config: Configuration dictionary (modified in-place)
        key_path: Dot-separated key path (e.g., "training.learning_rate")
        value: Value to set
        
    Example:
        >>> config = {'training': {}}
        >>> set_nested_value(config, 'training.learning_rate', 0.001)
        >>> config
        {'training': {'learning_rate': 0.001}}
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
