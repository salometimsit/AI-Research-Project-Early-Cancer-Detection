"""PyTorch dataset, weighted sampler, and 3D augmentation pipeline.

Pipeline stage: feeds J (SwinViT) and L (Cross-Attention Fusion).

Inputs
------
* ``data/processed/<PID>/before_cropped.nii.gz`` for every patient
  listed in ``data/labels.csv``.
* ``swin_vit.img_size`` from the config (target spatial resolution).
* ``augmentation.*`` for ``get_3d_augmentation``.

Outputs
-------
* Per-call dict ``{"image": tensor (1, D, H, W), "label": int,
  "patient_id": str}``.
* :class:`torch.utils.data.WeightedRandomSampler` for class-balanced
  training mini-batches (``get_weighted_sampler``).
* MONAI ``Compose`` of 3D random transforms (``get_3d_augmentation``).

Side effects
------------
* Logs and skips any patient whose cropped volume is missing.

Failure modes
-------------
* No usable samples -> ``RuntimeError`` raised by ``HCCDataset.__init__``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.preprocessing import ZScoreNormalizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HCCDataset(Dataset):
    """Cropped liver volumes + binary HCC labels.

    Each item is a ``(volume_tensor, label, patient_id)`` triple where
    ``volume_tensor`` has shape ``(1, D, H, W)`` (channel-first).
    """

    def __init__(
        self,
        processed_dir: Path | str,
        labels_csv: Path | str,
        transform: Any = None,
        target_size: tuple[int, int, int] | None = None,
        allow_missing: bool = True,
        augmentation_seed_base: int | None = None,
        *,
        zscore_enabled: bool = False,
        zscore_scope: str = "liver_only",
        zscore_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the HCC dataset from processed crops + labels.

        Args:
            processed_dir: Directory that contains per-patient subdirectories.
            labels_csv: CSV with at least ``patient_id`` and ``label`` columns.
            transform: Optional MONAI/PyTorch transform pipeline.
            target_size: Expected `(D, H, W)` shape for each cropped volume.
                When provided, a mismatch raises ``ValueError`` (no implicit
                runtime resize is performed in the dataset).
            allow_missing: If ``True``, samples with missing cropped volumes are
                skipped after a one-time preflight report. If ``False``, missing
                files trigger ``FileNotFoundError``.
            augmentation_seed_base: Optional base seed for deterministic
                transform random-state control. If provided and the transform
                supports ``set_random_state``, each item uses
                ``base + epoch * len(dataset) + idx``.
            zscore_enabled: If ``True``, apply liver-mask Z-score in
                ``__getitem__`` using ``before_liver.nii.gz`` and
                ``crop_metadata.json`` (same geometry as preprocessing). Use
                when ``before_cropped.nii.gz`` holds HU-windowed intensities only
                (``preprocessing.zscore.enabled: false``). If the pipeline
                already wrote Z-scored crops, keep this ``False`` to avoid
                double-normalisation.
            zscore_scope: Only ``"liver_only"`` is implemented (must match
                preprocessing); other values log a warning and behave as
                ``"liver_only"``.
            zscore_cfg: Optional ``preprocessing.zscore`` dict (``std_epsilon``,
                ``std_fallback``, ``background_fill_value``) aligned with
                ``configs/data.yaml``.
        """
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.target_size = target_size
        self.allow_missing = allow_missing
        self.augmentation_seed_base = augmentation_seed_base
        self._zscore_enabled = bool(zscore_enabled)
        self._zscore_scope = str(zscore_scope)
        self._zscore_cfg: dict[str, Any] = dict(zscore_cfg or {})
        self._epoch = 0

        if self._zscore_scope != "liver_only":
            logger.warning(
                "HCCDataset zscore_scope=%r is not supported; using liver_only.",
                self._zscore_scope,
            )
            self._zscore_scope = "liver_only"
        if self._zscore_enabled:
            logger.debug(
                "HCCDataset: runtime liver Z-score enabled (params from zscore_cfg keys=%s).",
                sorted(self._zscore_cfg.keys()),
            )

        labels_df = pd.read_csv(labels_csv)
        records: list[tuple[str, Path, int]] = [
            (
                str(row["patient_id"]),
                self.processed_dir / str(row["patient_id"]) / "before_cropped.nii.gz",
                int(row["label"]),
            )
            for _, row in labels_df.iterrows()
        ]
        missing = [(pid, vol_path) for pid, vol_path, _ in records if not vol_path.exists()]
        if missing and not self.allow_missing:
            missing_str = ", ".join(f"{pid}:{path}" for pid, path in missing[:10])
            raise FileNotFoundError(
                "Missing cropped volumes during dataset preflight: "
                f"{missing_str}"
            )
        if missing:
            preview = ", ".join(f"{pid}" for pid, _ in missing[:10])
            logger.warning(
                "Dataset preflight: %d/%d cropped volumes missing. "
                "Skipping missing patients (first up to 10): %s",
                len(missing),
                len(records),
                preview,
            )
        missing_set = {pid for pid, _ in missing}
        self.samples = [
            (pid, vol_path, label)
            for pid, vol_path, label in records
            if pid not in missing_set
        ]

        if not self.samples:
            raise RuntimeError("HCCDataset has no usable samples.")
        self.labels = np.array([s[2] for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume(self, path: Path) -> np.ndarray:
        """Load one cropped volume and enforce shape expectations."""
        volume = nib.load(str(path)).get_fdata().astype(np.float32)
        if self.target_size is not None and volume.shape != self.target_size:
            logger.warning(
                "Resizing %s from %s to target_size=%s in HCCDataset.",
                path.name,
                volume.shape,
                self.target_size,
            )
            volume = _resize_volume(volume, self.target_size, is_mask=False)
        return volume

    def _apply_runtime_zscore(self, volume: np.ndarray, vol_path: Path) -> np.ndarray:
        """Z-score ``volume`` using a liver mask crop matching ``vol_path``."""
        patient_dir = vol_path.parent
        meta_path = patient_dir / "crop_metadata.json"
        mask_path = patient_dir / "before_liver.nii.gz"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Runtime Z-score requires crop metadata: {meta_path}"
            )
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Runtime Z-score requires full-field liver mask: {mask_path}"
            )
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        bbox = meta["bbox"]
        start, stop = bbox["start"], bbox["stop"]
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        mask_full = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
        mask_crop = mask_full[slices].astype(np.uint8)
        if mask_crop.shape != volume.shape:
            raise ValueError(
                f"Liver mask crop shape {mask_crop.shape} != volume {volume.shape} "
                f"for {vol_path}"
            )
        cfg = self._zscore_cfg
        normalizer = ZScoreNormalizer()
        out, _stats = normalizer.normalize(
            volume,
            mask_crop,
            std_epsilon=float(cfg.get("std_epsilon", 1e-6)),
            std_fallback=str(cfg.get("std_fallback", "one")),
            background_fill_value=float(cfg.get("background_fill_value", 0.0)),
        )
        return out.astype(np.float32)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch used for deterministic augmentation seeds."""
        self._epoch = int(epoch)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pid, vol_path, label = self.samples[idx]
        volume = self._load_volume(vol_path)
        if self._zscore_enabled:
            volume = self._apply_runtime_zscore(volume, vol_path)
        sample: dict[str, Any] = {
            "image": volume[np.newaxis, ...],
            "label": label,
            "patient_id": pid,
        }
        if self.transform is not None:
            if (
                self.augmentation_seed_base is not None
                and hasattr(self.transform, "set_random_state")
            ):
                seed = int(
                    self.augmentation_seed_base
                    + self._epoch * len(self.samples)
                    + idx
                )
                self.transform.set_random_state(seed=seed)
            sample = self.transform(sample)
        if not isinstance(sample["image"], torch.Tensor):
            sample["image"] = torch.from_numpy(np.asarray(sample["image"])).float()
        sample["label"] = torch.tensor(int(sample["label"]), dtype=torch.long)
        return sample


def _resize_volume(
    volume: np.ndarray,
    size: tuple[int, int, int],
    is_mask: bool = False,
) -> np.ndarray:
    """Resize a 3D image/mask volume with interpolation by data type.

    Args:
        volume: Input 3D array.
        size: Target shape `(D, H, W)`.
        is_mask: If ``True``, use nearest-neighbour interpolation to preserve
            binary/discrete labels. Otherwise use trilinear interpolation.

    Returns:
        The resized 3D array as ``float32``.
    """
    tensor = torch.from_numpy(volume)[None, None].float()
    mode = "nearest" if is_mask else "trilinear"
    align_corners = None if is_mask else False
    resized = torch.nn.functional.interpolate(
        tensor, size=size, mode=mode, align_corners=align_corners
    )
    return resized.squeeze().numpy().astype(np.float32)


def get_weighted_sampler(dataset: HCCDataset) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that oversamples the minority class."""
    labels = dataset.labels
    class_counts = np.bincount(labels)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    logger.debug("WeightedRandomSampler weights per class: %s", class_weights)
    return sampler


def get_3d_augmentation(cfg: dict[str, Any]) -> Any:
    """Build a MONAI 3D augmentation transform pipeline from config.

    Every probability, range, and axis selector is read from the
    ``augmentation`` block of ``cfg``; nothing is hardcoded. The canonical
    source for these keys is ``configs/data.yaml``.

    Args:
        cfg: Parsed YAML configuration dict.

    Returns:
        A MONAI :class:`Compose` transform that operates on the
        ``"image"`` key of each sample dict.
    """
    from monai.transforms import (
        Compose,
        EnsureTyped,
        RandAdjustContrastd,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandScaleIntensityd,
        RandShiftIntensityd,
    )

    aug = cfg.get("augmentation", {})
    transforms = [EnsureTyped(keys=["image"])]

    rotation = float(aug.get("random_rotation_degrees", 0))
    if rotation > 0:
        rad = float(np.deg2rad(rotation))
        transforms.append(
            RandAffined(
                keys=["image"],
                prob=float(aug.get("random_rotation_prob", 0.5)),
                rotate_range=(rad, rad, rad),
                mode="bilinear",
            )
        )

    if aug.get("elastic_deformation", False):
        from monai.transforms import Rand3DElasticd

        transforms.append(
            Rand3DElasticd(
                keys=["image"],
                prob=float(aug.get("elastic_prob", 0.3)),
                sigma_range=tuple(aug.get("elastic_sigma_range", (5, 7))),
                magnitude_range=tuple(aug.get("elastic_magnitude_range", (50, 100))),
            )
        )

    if bool(aug.get("enable_intensity_shift", False)) and aug.get("intensity_shift"):
        transforms.append(
            RandShiftIntensityd(
                keys=["image"],
                offsets=float(aug["intensity_shift"]),
                prob=float(aug.get("intensity_shift_prob", 0.5)),
            )
        )

    if bool(aug.get("enable_intensity_scale", False)) and aug.get("intensity_scale"):
        transforms.append(
            RandScaleIntensityd(
                keys=["image"],
                factors=float(aug["intensity_scale"]),
                prob=float(aug.get("intensity_scale_prob", 0.5)),
            )
        )
    if bool(aug.get("enable_contrast_adjust", False)):
        transforms.append(
            RandAdjustContrastd(
                keys=["image"],
                prob=float(aug.get("contrast_prob", 0.3)),
                gamma=tuple(aug.get("contrast_gamma", (0.8, 1.2))),
            )
        )

    if aug.get("random_flip", False):
        flip_axes = list(aug.get("random_flip_axes", [0, 1, 2]))
        transforms.append(
            RandFlipd(
                keys=["image"],
                prob=float(aug.get("random_flip_prob", 0.5)),
                spatial_axis=flip_axes,
            )
        )

    if aug.get("gaussian_noise", True):
        transforms.append(
            RandGaussianNoised(
                keys=["image"],
                prob=float(aug.get("gaussian_noise_prob", 0.2)),
                std=float(aug.get("gaussian_noise_std", 0.01)),
            )
        )
    return Compose(transforms)
