"""
Lightweight YAML-based configuration loader for the project.

This module provides a simple interface for retrieving configuration values stored under `config/<name>.yaml`.

Configuration files must exist in the `config/` directory and follow the `<name>.yaml` naming convention.
"""


import yaml
from typing import Dict, Any


def _load_config(config: str) -> Dict:
    """
    Loads a config object.
    Args:
        config: str - config enumerator
    Returns:
        config dict
    """
    with open(f"config/{config}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config(config: str, key: str) -> Any:
    """
    Loads config by a key.
    Args:
        config: str - config enumerator
        key: str - key to index into the config object
    Returns:
        config value pointed to by key
    """
    return _load_config(config)[key]