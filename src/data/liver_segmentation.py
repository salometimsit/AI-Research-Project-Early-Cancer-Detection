"""Liver segmentation wrapper around TotalSegmentator.

Pipeline stage: E in the A->M flow.

Inputs
------
* A NIfTI volume produced by :mod:`src.data.dicom_loader`.
* ``preprocessing.liver_segmentation.gpu`` flag from the config.

Outputs
-------
* ``data/processed/<PID>/before_liver.nii.gz`` -- a binary mask
  (uint8) with ``1`` inside the liver, ``0`` elsewhere.

Side effects
------------
* TotalSegmentator downloads its weights on first use (~1.5 GB).
* Uses a temporary directory for the raw multi-class segmentation.

Failure modes
-------------
* Missing TotalSegmentator package -> ``ImportError`` at first call.
* Volume too small for the U-Net -> upstream error from
  TotalSegmentator (logged, not retried).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiverSegmentor:
    """Run TotalSegmentator and produce a binary liver mask."""

    def __init__(
        self,
        gpu: bool = True,
        roi_subset: tuple[str, ...] = ("liver",),
        ml: bool = False,
        fast: bool = False,
        require_non_empty_mask: bool = True,
    ) -> None:
        """Initialise the segmentor.

        Args:
            gpu: If True (default), TotalSegmentator runs on GPU;
                otherwise it falls back to CPU. Has no effect if no
                CUDA device is visible to the underlying library.
            roi_subset: Organs to request from TotalSegmentator.
            ml: TotalSegmentator ``ml`` flag.
            fast: TotalSegmentator ``fast`` flag.
            require_non_empty_mask: If True, raises when produced mask is empty.
        """
        self.gpu = gpu
        self.roi_subset = roi_subset
        self.ml = ml
        self.fast = fast
        self.require_non_empty_mask = require_non_empty_mask
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError:
            logger.error(
                "TotalSegmentator not installed. Install with: pip install TotalSegmentator"
            )
            raise
        self._ts_func = totalsegmentator

    def _run_totalsegmentator(self, nifti_path: Path, out_dir: Path) -> Path:
        """Invoke TotalSegmentator and return the path to ``liver.nii.gz``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self._ts_func(
            input=str(nifti_path),
            output=str(out_dir),
            roi_subset=list(self.roi_subset),
            ml=self.ml,
            fast=self.fast,
            device="gpu" if self.gpu else "cpu",
        )
        liver_path = out_dir / "liver.nii.gz"
        if not liver_path.exists():
            raise FileNotFoundError(
                f"TotalSegmentator did not produce liver.nii.gz in {out_dir}"
            )
        return liver_path

    def segment(self, nifti_path: Path | str) -> Path:
        """Segment the liver from ``nifti_path`` and persist a binary mask.

        The mask is written next to the input as ``before_liver.nii.gz``.
        """
        nifti_path = Path(nifti_path)
        out_path = nifti_path.parent / "before_liver.nii.gz"
        original_nii = nib.load(str(nifti_path))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            liver_raw = self._run_totalsegmentator(nifti_path, tmp_dir)
            liver_nii = nib.load(str(liver_raw))
            mask = (liver_nii.get_fdata() > 0).astype(np.uint8)
            if mask.shape != original_nii.shape:
                raise ValueError(
                    f"Segmentation shape {mask.shape} does not match source "
                    f"shape {original_nii.shape} for {nifti_path.name}."
                )
            if self.require_non_empty_mask and np.count_nonzero(mask) == 0:
                logger.error(
                    "TotalSegmentator produced an empty liver mask for %s",
                    nifti_path.name,
                )
                raise ValueError(
                    f"Empty liver mask detected for patient {nifti_path.parent.name}"
                )
            binary = nib.Nifti1Image(
                mask,
                affine=original_nii.affine,
                header=original_nii.header.copy(),
            )
            nib.save(binary, str(out_path))

        logger.info("Saved liver mask -> %s", out_path)
        return out_path

    def segment_all(self, processed_dir: Path | str) -> None:
        """Segment every patient under ``processed_dir`` that has a NIfTI volume."""
        processed_dir = Path(processed_dir)
        for patient_dir in sorted(processed_dir.iterdir()):
            volume_path = patient_dir / "before.nii.gz"
            if not volume_path.exists():
                continue
            try:
                self.segment(volume_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Liver segmentation failed for %s: %s", patient_dir.name, exc)
