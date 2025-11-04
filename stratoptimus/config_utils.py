"""Configuration utilities for loading and validating config files."""
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: Dictionary containing configuration parameters.
    :raises FileNotFoundError: If the config file doesn't exist.
    :raises yaml.YAMLError: If the YAML file is malformed.
    :raises ValueError: If the config file is empty or invalid.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not config_file.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config).__name__}")

        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading configuration from {config_path}: {e}") from e