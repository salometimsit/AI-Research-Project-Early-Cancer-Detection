"""Crop the liver bounding box and persist crop metadata.

Pipeline stage: F in the A->M flow.

Inputs
------
* ``before.nii.gz`` (the normalized CT) and ``before_liver.nii.gz``
  (binary liver mask) under ``data/processed/<PID>/``.
* Optional Z-score statistics dict from
  :func:`src.data.preprocessing.preprocess_volume`, saved alongside the
  crop metadata.

Outputs
-------
* ``data/processed/<PID>/before_cropped.nii.gz`` -- the volume cropped
  tightly to the liver bounding box.
* ``data/processed/<PID>/crop_metadata.json`` -- ``{"bbox": {"start",
  "stop"}, "original_shape", "cropped_shape", "zscore"}``. The bbox is
  required to reverse-map SwinViT attention heatmaps onto the original
  CT.

Failure modes
-------------
* Empty liver mask -> ``ValueError`` from ``crop_to_bbox``.
* Volume / mask shape mismatch -> ``ValueError`` (same call site).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiverCropper:
    """Compute the liver 3D bounding box and crop the volume.

    Axis-order convention:
        Both ``volume`` and ``mask`` must share the same axis order (the
        order returned by ``nibabel.load(...).get_fdata()``, typically
        ``(X, Y, Z)`` aka ``(W, H, D)`` for radiology NIfTI). The bbox
        ``start`` / ``stop`` arrays use that same axis order, and
        downstream consumers (e.g. ``plot_attention_heatmap``) must
        respect it.
    """

    def crop_to_bbox(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        padding: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Crop the volume to the liver bounding box.

        Args:
            volume: 3D HU-normalised CT volume.
            mask: 3D binary liver mask aligned with ``volume`` (same
                shape and same axis order).
            padding: Number of voxels to expand the bbox on each side.

        Returns:
            Tuple ``(cropped, metadata)`` where ``cropped`` is the
            contiguous 3D crop (``float32``) and ``metadata`` is a dict
            with keys ``bbox`` (``start`` / ``stop`` per axis,
            half-open), ``original_shape`` and ``cropped_shape``. The
            bbox is required to reverse-map attention heatmaps back to
            the original CT.

        Raises:
            ValueError: If shapes differ, the mask isn't 3D, or the
                liver mask is empty.
        """
        # Both inputs must be 3D and identically shaped. Without this
        # invariant, the bbox indices would be ambiguous between
        # (D, H, W) and (W, H, D) interpretations.
        assert volume.ndim == 3, f"volume must be 3D, got {volume.ndim}D"
        assert mask.ndim == 3, f"mask must be 3D, got {mask.ndim}D"
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}.")
        if volume.shape != mask.shape:
            raise ValueError(
                f"Volume {volume.shape} and mask {mask.shape} must match."
            )
        nonzero = np.argwhere(mask > 0)
        if nonzero.size == 0:
            raise ValueError("Liver mask is empty; cannot compute bbox.")
        mins = np.maximum(nonzero.min(axis=0) - padding, 0)
        maxs = np.minimum(nonzero.max(axis=0) + 1 + padding, volume.shape)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
        cropped = volume[slices]
        metadata: dict[str, Any] = {
            "bbox": {
                "start": [int(v) for v in mins.tolist()],
                "stop": [int(v) for v in maxs.tolist()],
            },
            "original_shape": list(volume.shape),
            "cropped_shape": list(cropped.shape),
            "axis_order": "nibabel-native (matches volume.shape order)",
        }
        return cropped.astype(np.float32), metadata

    def save_metadata(self, metadata: dict[str, Any], output_path: Path | str) -> None:
        """Persist the crop metadata JSON to ``output_path``.

        Args:
            metadata: Dict produced by :meth:`crop_to_bbox`.
            output_path: Destination path; parent directories are
                created on demand.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Saved crop metadata -> %s", output_path)


def crop_patient(
    patient_dir: Path | str,
    zscore_stats: dict[str, float] | None = None,
    padding: int | None = None,
    volume_filename: str = "before.nii.gz",
) -> tuple[Path, Path]:
    """Convenience helper: crop ``before.nii.gz`` using ``before_liver.nii.gz``.

    Returns the paths of the cropped NIfTI and the crop_metadata JSON.
    """
    if padding is None:
        raise ValueError(
            "padding is required. Pass cropping.padding from configs/data.yaml."
        )
    if not volume_filename:
        raise ValueError("volume_filename must be a non-empty filename.")
    patient_dir = Path(patient_dir)
    volume_path = patient_dir / volume_filename
    mask_path = patient_dir / "before_liver.nii.gz"
    cropped_path = patient_dir / "before_cropped.nii.gz"
    metadata_path = patient_dir / "crop_metadata.json"

    nii = nib.load(str(volume_path))
    mask_nii = nib.load(str(mask_path))
    volume = nii.get_fdata().astype(np.float32)
    mask = mask_nii.get_fdata().astype(np.uint8)

    cropper = LiverCropper()
    cropped, meta = cropper.crop_to_bbox(volume, mask, padding=int(padding))
    if zscore_stats is not None:
        meta["zscore"] = zscore_stats
    meta["padding"] = int(padding)

    # Preserve orientation/spacing from the source affine while translating
    # the origin to the cropped volume start index in voxel space.
    start = np.asarray(meta["bbox"]["start"], dtype=np.float64)
    start_h = np.append(start, 1.0)
    world_start = nii.affine @ start_h
    new_affine = nii.affine.copy()
    new_affine[:3, 3] = world_start[:3]

    cropped_nii = nib.Nifti1Image(cropped, affine=new_affine, header=nii.header.copy())
    nib.save(cropped_nii, str(cropped_path))
    cropper.save_metadata(meta, metadata_path)

    logger.info(
        "Cropped %s to bbox start=%s stop=%s",
        patient_dir.name,
        meta["bbox"]["start"],
        meta["bbox"]["stop"],
    )
    return cropped_path, metadata_path
