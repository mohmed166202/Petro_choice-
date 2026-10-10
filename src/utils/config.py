"""
config.py - Unified YAML Configuration Loader & Validator
---------------------------------------------------------
Loads configuration files, validates essential fields, and prevents
hardcoded experiment parameters across the codebase.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and returns a validated dictionary.
    
    Args:
        config_path (str): File path to the YAML configuration.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Validation of critical sections
    required_sections = ["experiment", "data", "model"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config '{config_path}' missing mandatory section: '{section}'")
            
    return config


if __name__ == "__main__":
    # Self-test
    test_cfg = "configs/base.yaml"
    if Path(test_cfg).exists():
        cfg = load_config(test_cfg)
        print(f"Loaded config successfully: {cfg['experiment']['name']}")
