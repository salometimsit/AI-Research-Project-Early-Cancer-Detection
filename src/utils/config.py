"""Configuration loading and access utilities."""

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path, optional
        Path to a YAML config file.  Falls back to ``configs/default.yaml``
        when *None*.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create all output directories referenced under ``cfg['paths']``."""
    for key, dir_path in cfg.get("paths", {}).items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
