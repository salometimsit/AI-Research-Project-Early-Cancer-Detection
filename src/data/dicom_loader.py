"""DICOM ingestion: phase filtering and conversion to NIfTI.

Pipeline stage: A + B in the A->M flow.

Inputs
------
* ``data/raw/<PID>/before/`` -- one folder of DICOM files per patient.
* ``preprocessing.target_phase`` from the config (e.g. ``portal_venous``).

Outputs
-------
* ``data/processed/<PID>/before.nii.gz`` -- a single NIfTI volume per
  patient, holding the matched contrast phase.

Side effects
------------
* Logs every series with ``ACCEPTED`` / ``REJECTED`` for the audit trail.

Failure modes
-------------
* No matching series found -> ``RuntimeError``. Inspect the log to see
  which series descriptions were rejected and adjust ``target_phase``
  or the per-phase keyword list.
* Multiple matching series -> the first acceptable one is kept; future
  refinements should rely on slice count or thickness if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import SimpleITK as sitk

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SeriesCandidate:
    """A single DICOM series candidate for phase filtering and ranking."""

    series_dir: Path
    series_id: str
    files: list[str]
    description: str
    slice_thickness_mm: float | None


class DICOMLoader:
    """Filter DICOM series by contrast phase and convert to NIfTI."""

    def __init__(
        self,
        raw_dir: Path | str,
        target_phase: str,
        processed_dir: Path | str,
        loader_cfg: dict[str, Any],
    ) -> None:
        """Initialise the loader.

        Args:
            raw_dir: Directory containing per-patient DICOM folders
                (``<raw_dir>/<PID>/before/``).
            target_phase: Contrast phase key from
                ``loader_cfg['phase_keywords']``. Unknown values fall back
                to a normalized whole-word match on ``target_phase``.
            processed_dir: Output directory; ``<PID>/before.nii.gz`` is
                written under it.
            loader_cfg: DICOM loader config block from ``configs/data.yaml``.
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_phase = target_phase.lower()
        self.loader_cfg = loader_cfg
        self.phase_keywords = self._parse_phase_keywords(loader_cfg)
        self._is_fallback = self.target_phase not in self.phase_keywords

        metadata_keys = loader_cfg.get("metadata_keys", {})
        self.description_keys = tuple(metadata_keys.get("description", ()))
        self.protocol_keys = tuple(metadata_keys.get("protocol", ()))
        self.thickness_keys = tuple(metadata_keys.get("slice_thickness", ()))
        if not self.description_keys and not self.protocol_keys:
            raise KeyError(
                "Missing dicom_loader.metadata_keys.description/protocol in data config."
            )

        matching_cfg = loader_cfg.get("matching", {})
        self.normalize_separators = bool(matching_cfg.get("normalize_separators", True))
        self.use_word_boundaries = bool(matching_cfg.get("use_word_boundaries", True))

        selection_cfg = loader_cfg.get("series_selection", {})
        self.prefer_thinner_slices = bool(
            selection_cfg.get("prefer_thinner_slices", True)
        )
        self.thickness_tolerance_mm = float(
            selection_cfg.get("thickness_tolerance_mm", 0.1)
        )
        self.fallback_to_max_slices_when_missing_metadata = bool(
            selection_cfg.get("fallback_to_max_slices_when_missing_metadata", True)
        )

        if self._is_fallback:
            logger.warning(
                "Unknown target phase '%s'; falling back to normalized word match.",
                self.target_phase,
            )
        logger.debug(
            "DICOMLoader initialized (raw=%s, target_phase=%s, fallback=%s)",
            self.raw_dir,
            self.target_phase,
            self._is_fallback,
        )

    @staticmethod
    def _parse_phase_keywords(loader_cfg: dict[str, Any]) -> dict[str, tuple[str, ...]]:
        """Parse and normalize the phase->keywords mapping from config."""
        raw = loader_cfg.get("phase_keywords", {})
        if not isinstance(raw, dict) or not raw:
            raise KeyError("Missing required dicom_loader.phase_keywords in data config.")
        parsed: dict[str, tuple[str, ...]] = {}
        for phase, values in raw.items():
            phase_key = str(phase).lower()
            if isinstance(values, (list, tuple)):
                parsed[phase_key] = tuple(str(v).lower() for v in values if str(v).strip())
            else:
                parsed[phase_key] = (str(values).lower(),)
        return parsed

    def _normalize_text(self, text: str) -> str:
        """Normalize separators and collapse whitespace for robust matching."""
        normalized = text.lower()
        if self.normalize_separators:
            normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _matches_phase(self, description: str) -> bool:
        """Return True if ``description`` matches the configured phase.

        Uses the configured keyword list when available, otherwise a
        normalized whole-word fallback derived from ``target_phase``.
        """
        text = self._normalize_text(description)
        keywords = self.phase_keywords.get(
            self.target_phase, (self.target_phase.replace("_", " "),)
        )
        for kw in keywords:
            normalized_kw = self._normalize_text(kw)
            if not normalized_kw:
                continue
            if self.use_word_boundaries:
                pattern = rf"\b{re.escape(normalized_kw)}\b"
                if re.search(pattern, text):
                    return True
            elif normalized_kw in text:
                return True
        return False

    def _read_series_metadata(self, first_file: str) -> tuple[str, float | None]:
        """Read DICOM headers only (no pixel data) for phase filtering."""
        reader = sitk.ImageFileReader()
        reader.SetFileName(first_file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        description_parts: list[str] = []
        for key in self.description_keys + self.protocol_keys:
            if reader.HasMetaDataKey(key):
                description_parts.append(reader.GetMetaData(key))
        description = self._normalize_text(" ".join(description_parts))

        slice_thickness_mm: float | None = None
        for key in self.thickness_keys:
            if not reader.HasMetaDataKey(key):
                continue
            try:
                slice_thickness_mm = float(reader.GetMetaData(key))
                break
            except ValueError:
                continue
        return description, slice_thickness_mm

    def _select_best_series(self, accepted: list[SeriesCandidate]) -> SeriesCandidate:
        """Rank accepted series using clinically relevant tie-breakers."""
        if not accepted:
            raise RuntimeError("No accepted series candidates to select from.")

        def _series_key(candidate: SeriesCandidate) -> tuple[float, int, str, str]:
            # Prefer smaller slice thickness when metadata exists.
            if self.prefer_thinner_slices and candidate.slice_thickness_mm is not None:
                tol = max(self.thickness_tolerance_mm, 1e-6)
                thickness_bucket = round(candidate.slice_thickness_mm / tol) * tol
                thickness_key = thickness_bucket
            elif self.prefer_thinner_slices:
                thickness_key = (
                    1e9 if self.fallback_to_max_slices_when_missing_metadata else -1e9
                )
            else:
                thickness_key = 0.0
            return (
                thickness_key,
                -len(candidate.files),  # tie-breaker: prefer more slices
                candidate.series_dir.as_posix(),
                candidate.series_id,
            )

        best = sorted(accepted, key=_series_key)[0]
        logger.info(
            "Selected best series dir=%s sid=%s slices=%d thickness_mm=%s",
            best.series_dir,
            best.series_id,
            len(best.files),
            best.slice_thickness_mm,
        )
        return best

    def filter_by_phase(self, patient_dir: Path) -> list[SeriesCandidate]:
        """Return DICOM series candidates matching ``target_phase``.

        Each candidate series is identified by SeriesInstanceUID via
        ``GDCMSeriesFileNames``. The phase decision uses
        ``SeriesDescription`` / ``ProtocolName`` metadata.
        """
        accepted: list[SeriesCandidate] = []
        reader = sitk.ImageSeriesReader()

        for series_dir in sorted(p for p in patient_dir.rglob("*") if p.is_dir()):
            try:
                series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
            except RuntimeError:
                continue
            for sid in series_ids:
                files = reader.GetGDCMSeriesFileNames(str(series_dir), sid)
                if not files:
                    continue
                description, thickness_mm = self._read_series_metadata(files[0])
                accepted_phase = self._matches_phase(description)
                decision = "ACCEPTED" if accepted_phase else "REJECTED"
                fallback_marker = " (fallback match)" if self._is_fallback else ""
                reason = (
                    "phase keyword matched"
                    if accepted_phase
                    else "no normalized keyword/token match"
                )
                logger.info(
                    "Series in %s sid=%s (desc='%s', thickness_mm=%s, slices=%d) -> "
                    "%s%s | reason=%s",
                    series_dir.name,
                    sid,
                    description,
                    thickness_mm,
                    len(files),
                    decision,
                    fallback_marker,
                    reason,
                )
                if accepted_phase:
                    accepted.append(
                        SeriesCandidate(
                            series_dir=series_dir,
                            series_id=sid,
                            files=files,
                            description=description,
                            slice_thickness_mm=thickness_mm,
                        )
                    )
        return accepted

    def convert_to_nifti(self, patient_id: str) -> Path:
        """Convert the matched series for a patient into a single NIfTI.

        Returns the path to ``data/processed/<PID>/before.nii.gz``.
        """
        patient_dir = self.raw_dir / patient_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory missing: {patient_dir}")

        accepted = self.filter_by_phase(patient_dir)
        if not accepted:
            raise RuntimeError(
                f"No '{self.target_phase}' series found for patient {patient_id}"
            )

        best_series = self._select_best_series(accepted)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(best_series.files)
        volume = reader.Execute()

        out_dir = self.processed_dir / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "before.nii.gz"
        sitk.WriteImage(volume, str(out_path))
        logger.info("Saved NIfTI volume for %s -> %s", patient_id, out_path)
        return out_path

    def process_all_patients(self, patient_ids: Iterable[str] | None = None) -> None:
        """Process every patient directory under ``raw_dir`` (or a subset)."""
        if patient_ids is None:
            patient_ids = [p.name for p in self.raw_dir.iterdir() if p.is_dir()]
        for pid in patient_ids:
            try:
                self.convert_to_nifti(pid)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to convert patient %s: %s", pid, exc)
