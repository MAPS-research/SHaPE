"""
Configuration utilities for API clients.
Handles environment variable loading and common helper functions.
"""

import os
from typing import Dict


def load_env_manually(env_file: str = '.env') -> Dict[str, str]:
    """
    Parse .env file and return configuration dictionary.
    
    Args:
        env_file: Path to the .env file
        
    Returns:
        Dictionary of configuration key-value pairs
    """
    config = {}
    if not os.path.exists(env_file):
        print(f'Warning: {env_file} file not found')
        return config
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Skip empty values if key already exists
                if key:
                    if key in config and not value:
                        continue
                    config[key] = value
        print(f'Loaded {len(config)} configuration items from {env_file}')
    except Exception as e:
        print(f'Error reading {env_file}: {e}')
    
    return config


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Absolute path to the project root
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the code directory, then to project root
    return os.path.dirname(os.path.dirname(current_dir))


def get_env_path() -> str:
    """
    Get the default .env file path.
    
    Returns:
        Path to the .env file in project root
    """
    return os.path.join(get_project_root(), '.env')

