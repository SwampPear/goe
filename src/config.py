import yaml
from typing import Dict, Any


def load_config(config: str) -> Dict:
    """Loads a config object."""
    path = f"config/{config}.yaml"

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config(config: str, key: str) -> Any:
    cfg = load_config(f"data")
    return cfg['urls']