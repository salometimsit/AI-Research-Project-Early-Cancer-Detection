"""Phase 2: DICOM -> NIfTI -> HU windowing + Z-score -> liver mask -> crop.

Usage
-----
    python scripts/run_preprocessing.py --config configs/default.yaml
    python scripts/run_preprocessing.py --config configs/default.yaml \\
        --patient P001 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cropping import crop_patient  # noqa: E402
from src.data.dicom_loader import DICOMLoader  # noqa: E402
from src.data.liver_segmentation import LiverSegmentor  # noqa: E402
from src.data.preprocessing import preprocess_volume  # noqa: E402
from src.utils.config import (  # noqa: E402
    ensure_dirs,
    load_config,
    load_data_config,
    set_seed,
)
from src.utils.logger import (  # noqa: E402
    get_logger,
    log_dict,
    log_section,
    setup_logger,
)
from src.utils.run_metadata import record_failure, record_run  # noqa: E402


def _patient_ids(raw_dir: Path) -> list[str]:
    """Return sorted patient directory names under ``raw_dir``.

    Hidden / tooling directories (names starting with ``.``) are ignored so
    paths like ``.ipynb_checkpoints`` are not treated as patient IDs.
    """
    return [
        p.name
        for p in sorted(raw_dir.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_data_config(args.data_config)
    ensure_dirs(cfg)
    set_seed(cfg)

    logs_dir = Path(cfg["paths"]["logs_dir"])
    setup_logger(
        "hcc",
        log_file=logs_dir / "preprocessing.log",
        console_level=args.log_level,
    )
    logger = get_logger("hcc.preprocessing")
    record_run("preprocessing", cfg, vars(args))

    log_section(logger, f"Preprocessing | seed={cfg.get('seed')}")
    log_dict(logger, "config.preprocessing", cfg["preprocessing"])

    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    target_phase = cfg["preprocessing"]["target_phase"]
    dicom_loader_cfg = data_cfg.get("dicom_loader")
    if not isinstance(dicom_loader_cfg, dict):
        raise KeyError("Missing required key data_config['dicom_loader'].")

    loader = DICOMLoader(
        raw_dir,
        target_phase,
        processed_dir,
        loader_cfg=dicom_loader_cfg,
    )
    liver_seg_cfg = data_cfg.get("liver_segmentation")
    if not isinstance(liver_seg_cfg, dict):
        raise KeyError("Missing required key data_config['liver_segmentation'].")
    pre_data_cfg = data_cfg.get("preprocessing")
    if not isinstance(pre_data_cfg, dict):
        raise KeyError("Missing required key data_config['preprocessing'].")
    segmentor = LiverSegmentor(
        gpu=bool(cfg["preprocessing"]["liver_segmentation"].get("gpu", True)),
        roi_subset=tuple(liver_seg_cfg.get("roi_subset", ["liver"])),
        ml=bool(liver_seg_cfg.get("ml", False)),
        fast=bool(liver_seg_cfg.get("fast", False)),
        require_non_empty_mask=bool(liver_seg_cfg.get("require_non_empty_mask", True)),
    )

    patient_ids = [args.patient] if args.patient else _patient_ids(raw_dir)
    if not patient_ids:
        logger.error("No patient directories found under %s", raw_dir)
        sys.exit(1)

    succeeded = 0
    failed = 0
    for pid in patient_ids:
        log_section(logger, f"Patient {pid}", char="-")
        try:
            final_crop = processed_dir / pid / "before_cropped.nii.gz"
            if final_crop.exists():
                logger.info("Patient %s already preprocessed. Skipping.", pid)
                succeeded += 1
                continue
            volume_path = loader.convert_to_nifti(pid)
            mask_path = segmentor.segment(volume_path)
            _, stats, normalized_path = preprocess_volume(volume_path, mask_path, pre_data_cfg)
            cropping_cfg = data_cfg.get("cropping", {})
            if "padding" not in cropping_cfg:
                raise KeyError(
                    "Missing required key data_config['cropping']['padding']."
                )
            padding = int(cropping_cfg["padding"])
            _, _, _ = crop_patient(
                volume_path.parent,
                zscore_stats=stats,
                padding=padding,
                volume_filename=normalized_path.name,
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error("Preprocessing failed for %s: %s", pid, exc)
            record_failure("preprocessing", pid, exc, logs_dir=logs_dir)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log_section(logger, f"Done | ok={succeeded} fail={failed}")


if __name__ == "__main__":
    main()
