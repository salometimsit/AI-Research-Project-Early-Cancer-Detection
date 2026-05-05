"""Configuration loading, seed management and directory utilities.

Pipeline stage: cross-cutting; called by every script.

Inputs
------
* A YAML file (default: ``configs/default.yaml``).
* The integer seed read from ``cfg["seed"]``.

Outputs
-------
* :func:`load_config` -> parsed dict.
* :func:`ensure_dirs` -> creates ``paths.*`` output directories on disk.
* :func:`set_seed` -> seeds ``random``, ``numpy``, ``torch``, CUDA and
  ``PYTHONHASHSEED``.

Failure modes
-------------
* Missing or malformed YAML -> standard PyYAML errors.
* ``ensure_dirs`` skips path entries that look like files (have a
  suffix), so ``labels.csv`` is not turned into a directory.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
)
_DEFAULT_VIZ_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "visualization.yaml"
)


_REQUIRED_SECTIONS: tuple[str, ...] = ("paths", "seed", "preprocessing", "swin_vit")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        path: Path to the YAML config file. Accepts either a ``str`` or a
            :class:`pathlib.Path`. When ``None`` (the default) the project's
            canonical ``configs/default.yaml`` (resolved relative to this
            module) is used.

    Returns:
        The parsed YAML document as a nested ``dict``. Top-level keys are
        the config sections (``paths``, ``preprocessing``, ``training``,
        etc.). Returns an empty ``dict`` if the file is empty.

    Raises:
        FileNotFoundError: The resolved config path does not exist.
        PermissionError: The file exists but cannot be read.
        yaml.YAMLError: The file is not valid YAML (parser/scanner error).
        UnicodeDecodeError: The file is not valid UTF-8.
        KeyError: A mandatory config section (paths, seed, preprocessing,
            swin_vit) is absent from the loaded document.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for key in _REQUIRED_SECTIONS:
        if key not in cfg:
            raise KeyError(
                f"Missing mandatory config section: '{key}' in {cfg_path}"
            )
    return cfg


# Suffixes we treat as "this is a real file, not a directory" when walking
# ``paths`` recursively. The presence of one of these in ``Path(value).suffix``
# makes ``ensure_dirs`` create the *parent* directory instead of the path
# itself. Anything else (e.g. ``models.v2/``) is treated as a directory even
# if it contains a dot in the final component.
_FILE_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".json", ".jsonl", ".npz", ".pkl", ".pth", ".yaml", ".yml",
     ".log", ".png", ".nii", ".gz", ".txt", ".md",
     ".dcm", ".dicom", ".dicm", ".mhd", ".raw"}
)


def _walk_path_values(node: Any) -> list[str]:
    """Recursively flatten string leaves under a nested ``paths`` block."""
    if isinstance(node, str):
        return [node] if node else []
    if isinstance(node, dict):
        out: list[str] = []
        for value in node.values():
            out.extend(_walk_path_values(value))
        return out
    if isinstance(node, (list, tuple)):
        out = []
        for value in node:
            out.extend(_walk_path_values(value))
        return out
    return []


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create every output directory referenced under ``cfg['paths']``.

    Walks ``cfg['paths']`` recursively and ensures each path leaf exists on
    disk. Entries whose final suffix is in :data:`_FILE_SUFFIXES` are treated
    as files; their parent directory is created instead. All other entries
    are treated as directories and created with ``parents=True,
    exist_ok=True``.

    Args:
        cfg: The full parsed config dictionary. Only ``cfg['paths']`` is
            consulted; if missing, the function is a no-op. The ``paths``
            block may be arbitrarily nested (dicts, lists, tuples).

    Returns:
        None. The function only has side effects on the filesystem.

    Raises:
        PermissionError: The process lacks permission to create one of the
            requested directories.
        OSError: A lower-level OS error occurred while creating a directory
            (e.g. ``ENOSPC``, invalid path).
    """
    for raw in _walk_path_values(cfg.get("paths", {})):
        path = Path(raw)
        if path.suffix.lower() in _FILE_SUFFIXES:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def set_seed(cfg: dict[str, Any]) -> None:
    """Seed every RNG involved in the pipeline for full reproducibility.

    Reads ``seed``, ``training.deterministic`` and ``training.cudnn_benchmark``
    from the config dict so the YAML is the single source of truth for all
    reproducibility knobs.

    Beyond the obvious ``random`` / ``numpy`` / ``torch`` seeds, this also:

    * Sets ``PYTHONHASHSEED`` so dict / set ordering is stable.
    * Honours ``training.deterministic`` (default ``True``) to control cuDNN
      deterministic mode — set to ``False`` to trade reproducibility for speed.
    * Calls ``torch.use_deterministic_algorithms(True)`` so any CUDA op
      that has only a non-deterministic implementation raises rather than
      silently introducing run-to-run drift.
    * Sets ``CUBLAS_WORKSPACE_CONFIG`` (required by PyTorch when
      deterministic algorithms are enabled on CUDA >= 10.2).

    Falls back gracefully if optional dependencies are missing; any failure
    while applying the optional knobs is logged at ``WARNING`` level rather
    than swallowed silently.

    Warning:
        The deterministic flags set here are **process-wide and effectively
        non-reversible** for the lifetime of the Python interpreter:

        * ``torch.use_deterministic_algorithms(True)`` mutates global
          PyTorch state. Any subsequent op that lacks a deterministic
          implementation will raise ``RuntimeError`` (with
          ``warn_only=True`` we downgrade this to a warning, but the global
          flag itself stays on).
        * ``torch.backends.cudnn.deterministic = True`` and
          ``benchmark = False`` disable the cuDNN autotuner and can
          measurably slow down 3D convolutional models such as SwinViT.
        * ``CUBLAS_WORKSPACE_CONFIG`` is written to ``os.environ`` and
          inherited by every child process spawned afterwards.
        * ``PYTHONHASHSEED`` is set after interpreter start, so it only
          affects code paths that read it explicitly; it will not retro-
          actively change the hash seed of the running interpreter.

        Call ``set_seed`` exactly once, as early as possible in ``main()``,
        and do not expect a later call with a different ``seed`` to undo
        the deterministic-mode side effects above.

    Args:
        cfg: The full parsed config dict (output of :func:`load_config`).
            Reads ``cfg["seed"]``, ``cfg["training"]["deterministic"]`` and
            ``cfg["training"]["cudnn_benchmark"]``.

    Returns:
        None. Side effects are confined to global RNG state, ``os.environ``
        and PyTorch backend flags.
    """
    seed = int(cfg.get("seed", 42))
    training = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    is_deterministic = bool(training.get("deterministic", True))
    use_benchmark = bool(training.get("cudnn_benchmark", False))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError as exc:  # pragma: no cover
        _logger.warning("NumPy not available; skipping np.random seeding (%s).", exc)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        _logger.warning("PyTorch not available; skipping torch seeding (%s).", exc)
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = is_deterministic
        torch.backends.cudnn.benchmark = use_benchmark
    except AttributeError as exc:  # pragma: no cover -- non-CUDA torch builds
        _logger.warning(
            "Could not set cuDNN deterministic flags (likely a non-CUDA "
            "torch build): %s.",
            exc,
        )

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError) as exc:  # pragma: no cover
        _logger.warning(
            "Could not enable torch.use_deterministic_algorithms(True); "
            "results may have minor run-to-run drift: %s.",
            exc,
        )


def load_viz_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the optional visualization config (dpi, figsize, etc.).

    This config is **not** required for the pipeline to run. When the file
    is absent an empty dict is returned and all visualization functions fall
    back to their built-in defaults.

    Args:
        path: Path to the YAML visualization config. Defaults to
            ``configs/visualization.yaml`` (resolved relative to this module).

    Returns:
        Parsed YAML document as a nested ``dict``, or ``{}`` if the file
        does not exist.

    Raises:
        yaml.YAMLError: The file exists but is not valid YAML.
        UnicodeDecodeError: The file exists but is not valid UTF-8.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_VIZ_CONFIG_PATH
    if not cfg_path.exists():
        _logger.debug("Visualization config not found at %s; using defaults.", cfg_path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        viz_cfg = yaml.safe_load(fh) or {}
    _logger.debug("Loaded visualization config from %s.", cfg_path)
    return viz_cfg
