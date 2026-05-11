"""PyRadiomics extraction over the full liver volume.

Pipeline stage: G in the A->M flow.

Inputs
------
* ``data/processed/<PID>/before_cropped.nii.gz`` and
  ``before_liver.nii.gz`` for every labelled patient (names overridable via
  ``radiomics.io`` in ``configs/features.yaml``).
* ``radiomics`` block in ``configs/features.yaml`` (see that file for the
  full schema: PyRadiomics tuning, mask threshold, I/O filenames, tqdm
  labels).

Outputs
-------
* ``data/raw_radiomics_features.csv`` -- one row per patient, one
  column per PyRadiomics feature plus ``patient_id`` and ``label``.
  Rows are appended incrementally during ``extract_all`` to limit RAM.

Failure modes
-------------
* Missing input volumes / masks -> the patient is skipped with a
  warning (logged) so a partial run still proceeds.
* Mask shape mismatch vs volume, or foreground voxel count below
  ``min_mask_voxels`` -> skipped with a warning.
* PyRadiomics raising on a degenerate mask -> the failure is caught and
  the patient is skipped; the rest of the run continues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_RAD_KEYS: Final[tuple[str, ...]] = (
    "feature_classes",
    "bin_width",
    "normalize",
    "min_mask_voxels",
    "label",
    "io",
    "progress",
)


class RadiomicsExtractor:
    """Configurable PyRadiomics extractor over the whole liver mask."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialise the extractor from the ``radiomics`` block in features config.

        Args:
            cfg: Sub-dict ``features_config['radiomics']``. Must define every
                key in ``_REQUIRED_RAD_KEYS``, optional ``resampled_pixel_spacing``
                and ``interpolator`` (interpolator required when spacing is set).

        Raises:
            KeyError: If a required key is absent.
            ValueError: If ``resampled_pixel_spacing`` is set but
                ``interpolator`` is missing.
        """
        missing = [k for k in _REQUIRED_RAD_KEYS if k not in cfg]
        if missing:
            raise KeyError(
                "radiomics config missing required keys "
                f"{sorted(missing)} — set under configs/features.yaml."
            )
        io = cfg["io"]
        if not isinstance(io, dict):
            raise TypeError("radiomics['io'] must be a dict.")
        progress = cfg["progress"]
        if not isinstance(progress, dict):
            raise TypeError("radiomics['progress'] must be a dict.")
        for k in ("volume_filename", "mask_filename"):
            if k not in io:
                raise KeyError(f"radiomics.io missing required key '{k}'.")
        for k in ("desc", "unit"):
            if k not in progress:
                raise KeyError(f"radiomics.progress missing required key '{k}'.")

        self.cfg = cfg
        self.feature_classes: list[str] = list(cfg["feature_classes"])
        self.bin_width: float = float(cfg["bin_width"])
        self.normalize: bool = bool(cfg["normalize"])
        spacing_raw = cfg.get("resampled_pixel_spacing")
        self.resampled_pixel_spacing: list[float] | None = None
        if spacing_raw is not None:
            self.resampled_pixel_spacing = [float(x) for x in list(spacing_raw)]
        self.interpolator: str | None = (
            str(cfg["interpolator"]) if cfg.get("interpolator") is not None else None
        )
        if self.resampled_pixel_spacing is not None and self.interpolator is None:
            raise ValueError(
                "radiomics.interpolator is required when "
                "resampled_pixel_spacing is set (features.yaml)."
            )
        self.min_mask_voxels: int = int(cfg["min_mask_voxels"])
        self.label: int = int(cfg["label"])
        self._volume_filename: str = str(io["volume_filename"])
        self._mask_filename: str = str(io["mask_filename"])
        self._progress_desc: str = str(progress["desc"])
        self._progress_unit: str = str(progress["unit"])
        self._extractor = self._build_extractor()

    def _build_extractor(self) -> Any:
        """Construct the underlying ``RadiomicsFeatureExtractor`` instance."""
        from radiomics.featureextractor import RadiomicsFeatureExtractor

        params: dict[str, Any] = {
            "binWidth": self.bin_width,
            "normalize": self.normalize,
            "label": self.label,
        }
        if self.resampled_pixel_spacing is not None:
            params["resampledPixelSpacing"] = self.resampled_pixel_spacing
        if self.interpolator is not None:
            params["interpolator"] = self.interpolator
        extractor = RadiomicsFeatureExtractor(**params)
        extractor.disableAllFeatures()
        for fc in self.feature_classes:
            extractor.enableFeatureClassByName(fc)
        return extractor

    def _mask_ok_for_radiomics(
        self,
        volume_path: Path | str,
        mask_path: Path | str,
    ) -> bool:
        """Return False if the mask is empty, too small, or shape-mismatched vs volume."""
        vol_path = Path(volume_path)
        m_path = Path(mask_path)
        vol_nii = nib.load(str(vol_path))
        mask_nii = nib.load(str(m_path))
        if vol_nii.shape != mask_nii.shape:
            logger.warning(
                "Skipping radiomics: mask shape %s != volume shape %s (%s).",
                mask_nii.shape,
                vol_nii.shape,
                m_path,
            )
            return False
        mask_data = np.asanyarray(mask_nii.dataobj)
        n_fg = int(np.count_nonzero(mask_data))
        if n_fg < self.min_mask_voxels:
            logger.warning(
                "Skipping radiomics: mask %s has %d foreground voxels "
                "(minimum %d).",
                m_path,
                n_fg,
                self.min_mask_voxels,
            )
            return False
        return True

    def extract_patient(
        self,
        volume_path: Path | str,
        mask_path: Path | str,
    ) -> dict[str, float]:
        """Run PyRadiomics on a single patient and return numeric features."""
        if not self._mask_ok_for_radiomics(volume_path, mask_path):
            return {}
        result = self._extractor.execute(str(volume_path), str(mask_path))
        features: dict[str, float] = {}
        for key, value in result.items():
            if key.startswith("diagnostics_"):
                continue
            try:
                features[key] = float(value)
            except (TypeError, ValueError):
                continue
        return features

    def extract_all(
        self,
        processed_dir: Path | str,
        labels_csv: Path | str,
        output_csv: Path | str,
    ) -> pd.DataFrame:
        """Extract radiomics for every patient that has a cropped volume + mask.

        ``output_csv`` is appended row-by-row so RAM stays bounded and partial
        progress survives interruptions. The returned DataFrame is read back
        from that CSV (same columns as on disk).
        """
        processed_dir = Path(processed_dir)
        labels_df = pd.read_csv(labels_csv)
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if output_csv.exists():
            output_csv.unlink()

        column_order: list[str] | None = None
        rows_written = 0

        for _, row in tqdm(
            list(labels_df.iterrows()),
            desc=self._progress_desc,
            unit=self._progress_unit,
        ):
            pid = str(row["patient_id"])
            patient_dir = processed_dir / pid
            volume_path = patient_dir / self._volume_filename
            mask_path = patient_dir / self._mask_filename
            if not volume_path.exists() or not mask_path.exists():
                logger.warning("Skipping radiomics for %s (missing inputs).", pid)
                continue
            try:
                feats = self.extract_patient(volume_path, mask_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Radiomics failed for %s: %s", pid, exc)
                continue
            if not feats:
                continue
            feats["patient_id"] = pid
            feats["label"] = int(row["label"])

            if column_order is None:
                feature_keys = sorted(
                    k for k in feats if k not in ("patient_id", "label")
                )
                column_order = ["patient_id", "label", *feature_keys]

            row_out: dict[str, Any] = {c: feats.get(c) for c in column_order}
            df_single = pd.DataFrame([row_out])
            mode = "w" if rows_written == 0 else "a"
            header = rows_written == 0
            df_single.to_csv(output_csv, mode=mode, index=False, header=header)
            rows_written += 1

        if rows_written == 0:
            df = pd.DataFrame()
            logger.warning(
                "No radiomics rows written; CSV not created or empty -> %s",
                output_csv,
            )
        else:
            df = pd.read_csv(output_csv)
            logger.info(
                "Saved radiomics features (%d patients) -> %s",
                rows_written,
                output_csv,
            )
        return df
