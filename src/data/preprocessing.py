"""HU windowing and liver-only Z-score normalization.

Pipeline stage: C + D in the A->M flow.

Inputs
------
* ``data/processed/<PID>/before.nii.gz`` (raw NIfTI from DICOMLoader).
* ``data/processed/<PID>/before_liver.nii.gz`` (liver mask).
* ``data_config.preprocessing`` block with HU window / Z-score settings.

Outputs
-------
* ``data/processed/<PID>/before_norm.nii.gz`` -- normalized volume (same
  affine + header as the input by default).
* Returned ``zscore_stats`` dict with ``mean`` and ``std`` measured
  inside the liver mask -- the caller (``cropping.crop_patient``)
  records this in ``crop_metadata.json`` so attention heatmaps can be
  reverse-mapped.

Failure modes
-------------
* Empty liver mask -> ``ValueError`` from ``ZScoreNormalizer.normalize``.
* HU window with ``upper <= lower`` -> ``ValueError`` from ``HUWindower``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HUWindower:
    """Clip Hounsfield units to a fixed soft-tissue / liver range."""

    def __init__(self, lower: int, upper: int) -> None:
        """Build a windower with the given inclusive HU range.

        Args:
            lower: Lower HU bound (inclusive).
            upper: Upper HU bound (inclusive); must exceed ``lower``.

        Raises:
            ValueError: If ``upper <= lower``.
        """
        if upper <= lower:
            raise ValueError("HU window upper must exceed lower.")
        self.lower = int(lower)
        self.upper = int(upper)

    def apply(self, volume: np.ndarray) -> np.ndarray:
        """Clip ``volume`` element-wise to ``[lower, upper]``.

        Args:
            volume: HU values from a CT scan (any shape / dtype).

        Returns:
            ``float32`` array of the same shape with values clipped to
            the configured HU range.
        """
        return np.clip(volume, self.lower, self.upper).astype(np.float32)


class ZScoreNormalizer:
    """Z-score normalization using statistics computed inside the liver mask."""

    def normalize(
        self,
        volume: np.ndarray,
        liver_mask: np.ndarray,
        std_epsilon: float = 1e-6,
        std_fallback: str = "one",
        background_fill_value: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Apply Z-score normalization driven by liver-mask statistics.

        Parameters
        ----------
        volume:
            HU-windowed CT volume.
        liver_mask:
            Binary mask aligned with ``volume`` (non-zero = liver voxel).
        std_epsilon:
            Minimum allowed standard deviation before fallback is applied.
        std_fallback:
            Strategy for tiny liver std: ``"one"`` or ``"global"``.
        background_fill_value:
            Value assigned outside the liver mask after normalization.

        Returns
        -------
        tuple
            Tuple of ``(normalized_volume, stats)`` where ``stats`` contains
            the ``mean`` and ``std`` measured inside the liver mask. These
            are persisted in ``crop_metadata.json`` to support reverse
            mapping of attention heatmaps.
        """
        liver_voxels = volume[liver_mask > 0]
        if liver_voxels.size == 0:
            raise ValueError("Liver mask is empty; cannot compute Z-score.")
        mean = float(np.mean(liver_voxels))
        std = float(np.std(liver_voxels))
        if std < std_epsilon:
            if std_fallback == "global":
                logger.warning(
                    "Liver std < %.2e; falling back to global std.",
                    std_epsilon,
                )
                std = float(np.std(volume) + std_epsilon)
            else:
                logger.warning(
                    "Liver std < %.2e; falling back to std=1.0.",
                    std_epsilon,
                )
                std = 1.0
        normalized = (volume - mean) / std
        normalized[liver_mask == 0] = float(background_fill_value)
        stats = {"mean": mean, "std": std, "mask_voxels": float(liver_voxels.size)}
        return normalized.astype(np.float32), stats


def preprocess_volume(
    volume_path: Path | str,
    mask_path: Path | str,
    pre_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float], Path]:
    """Run HU windowing + Z-score normalization for a single patient.

    The normalized volume is saved as ``before_norm.nii.gz`` by default so
    the raw ``before.nii.gz`` remains untouched.

    Backward compatibility:
        Older call sites may still pass the full ``default.yaml`` config
        (which contains a top-level ``preprocessing`` key). In that case we
        automatically read ``pre_cfg["preprocessing"]``.
    """
    volume_path = Path(volume_path)
    mask_path = Path(mask_path)
    if "preprocessing" in pre_cfg and isinstance(pre_cfg["preprocessing"], dict):
        pre_cfg = pre_cfg["preprocessing"]
    hu_cfg = pre_cfg["hu_window"]
    zscore_cfg = pre_cfg.get("zscore", {})
    output_cfg = pre_cfg.get("output", {})

    nii = nib.load(str(volume_path))
    volume = nii.get_fdata().astype(np.float32)
    mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)

    windower = HUWindower(hu_cfg["lower"], hu_cfg["upper"])
    windowed = windower.apply(volume)

    if bool(zscore_cfg.get("enabled", True)):
        normalizer = ZScoreNormalizer()
        normalized, stats = normalizer.normalize(
            windowed,
            mask,
            std_epsilon=float(zscore_cfg.get("std_epsilon", 1e-6)),
            std_fallback=str(zscore_cfg.get("std_fallback", "one")),
            background_fill_value=float(zscore_cfg.get("background_fill_value", 0.0)),
        )
    else:
        normalized = windowed
        stats = {
            "mean": float(np.mean(windowed)),
            "std": float(np.std(windowed)),
            "mask_voxels": float(np.count_nonzero(mask)),
        }

    normalized_filename = str(output_cfg.get("normalized_filename", "before_norm.nii.gz"))
    out_path = volume_path.parent / normalized_filename
    out = nib.Nifti1Image(normalized, affine=nii.affine, header=nii.header.copy())
    nib.save(out, str(out_path))
    logger.info(
        "Saved normalized volume -> %s (HU=[%d,%d], mean=%.3f, std=%.3f)",
        out_path,
        windower.lower,
        windower.upper,
        stats["mean"],
        stats["std"],
    )
    return normalized, stats, out_path
