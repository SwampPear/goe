import yaml
from typing import Dict, Any


def _load_config(config: str) -> Dict:
    """Loads a config object."""
    path = f"config/{config}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config(config: str, key: str) -> Any:
    return _load_config(config)[key]