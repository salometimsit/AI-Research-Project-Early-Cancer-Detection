"""Disk I/O helpers for TCAV artefacts.

Side effects
------------
* ``save_cavs_npz`` writes ``tcav/cavs.npz`` (or any path you pass) with
  one array per concept name -- safe to inspect via ``np.load``.
* ``save_cav_accuracies`` writes a JSON table of CAV training metadata
  (training accuracy, positive/negative arm sizes).

Both helpers create parent directories on demand and log the artefact
path at INFO level so downstream notebooks / scripts can pick them up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger
from src.utils.tcav.types import CAVRecord

logger = get_logger(__name__)


def save_cavs_npz(records: dict[str, CAVRecord], path: Path | str) -> Path:
    """Persist CAV vectors to a single ``.npz`` keyed by concept name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: rec.vector for name, rec in records.items()}
    np.savez_compressed(path, **payload)
    logger.info("Saved %d CAVs -> %s", len(records), path)
    return path


def save_cav_accuracies(
    records: dict[str, CAVRecord],
    path: Path | str,
    params: dict | None = None,
    p_vals: dict[str, float] | None = None,
    *,
    is_significant: dict[str, bool] | None = None,
) -> Path:
    """Persist per-CAV training metadata as a JSON table.

    Parameters
    ----------
    records:
        CAV records keyed by concept name.
    path:
        Destination JSON path.
    params:
        Hyperparameters used during CAV training (e.g. ``{"C": 0.1}``).
        Saved under ``"config"`` for each concept to ensure reproducibility.
    p_vals:
        Per-concept permutation p-values from
        :meth:`TCAV3D.compute_significance`. Saved alongside ``is_significant``
        so reviewers can verify statistical validity.
    is_significant:
        Optional per-concept flags (e.g. Bonferroni-corrected). When
        provided, overrides the default ``p < significance_threshold`` rule.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _p = p_vals or {}
    _params = params or {}
    sig_th = float(_params.get("significance_threshold", 0.05))
    _sig = is_significant or {}
    data = {
        name: {
            "train_accuracy": rec.train_accuracy,
            "n_positive": rec.n_positive,
            "n_negative": rec.n_negative,
            "config": _params,
            "p_value": _p.get(name, float("nan")),
            "is_significant": (
                bool(_sig[name])
                if name in _sig
                else bool(_p.get(name, 1.0) < sig_th)
            ),
        }
        for name, rec in records.items()
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved complete TCAV metadata (including P-values) -> %s", path)
    return path
