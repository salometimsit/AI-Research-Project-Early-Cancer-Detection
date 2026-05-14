"""Download per-patient DICOM zips from Google Drive into ``data/raw/``.

Usage
-----
    python scripts/download_data.py --config configs/default.yaml
    python scripts/download_data.py --config configs/default.yaml --patient P001
    python scripts/download_data.py --config configs/default.yaml --list

Notes
-----
    Corporate or university networks may block ``gdown`` traffic to Google;
    use VPN, proxy, or manual download if downloads fail with network errors.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import ensure_dirs, load_config  # noqa: E402
from src.utils.logger import get_logger, log_section, setup_logger  # noqa: E402
from src.utils.run_metadata import record_failure, record_run  # noqa: E402


def _download_patient(
    patient_id: str,
    file_id: str,
    raw_dir: Path,
    *,
    gdown_fuzzy: bool,
) -> bool:
    import gdown

    logger = get_logger(__name__)
    target_dir = raw_dir / patient_id / "before"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / patient_id / f"{patient_id}.zip"

    logger.info("Downloading %s (id=%s)", patient_id, file_id)
    try:
        gdown.download(
            id=file_id,
            output=str(zip_path),
            quiet=False,
            fuzzy=gdown_fuzzy,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed for %s: %s", patient_id, exc)
        return False

    if not zip_path.exists():
        logger.error("Expected zip missing for %s: %s", patient_id, zip_path)
        return False

    logger.info("Extracting %s -> %s", zip_path.name, target_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                logger.error(
                    "Corrupted zip member for %s: %s",
                    patient_id,
                    bad_member,
                )
                return False
            zf.extractall(target_dir)
    except zipfile.BadZipFile:
        logger.error("File %s is not a valid ZIP for %s", zip_path, patient_id)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Extraction failed for %s: %s", patient_id, exc)
        return False

    zip_path.unlink(missing_ok=True)

    dicom_files = list(target_dir.rglob("*.dcm"))
    if not dicom_files:
        logger.warning("No .dcm files found in %s after extraction.", target_dir)
        return False
    logger.info("Patient %s ready (%d DICOM files).", patient_id, len(dicom_files))
    return True


def _list_status(patient_files: dict[str, str], raw_dir: Path) -> None:
    print(f"{'patient_id':<15} {'file_id':<40} downloaded")
    print("-" * 70)
    for pid, fid in patient_files.items():
        patient_before = raw_dir / pid / "before"
        if patient_before.exists():
            num_dicoms = len(list(patient_before.rglob("*.dcm")))
            status = f"yes ({num_dicoms} .dcm)" if num_dicoms else "no"
        else:
            status = "no"
        print(f"{pid:<15} {fid:<40} {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Drive DICOM downloader")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--patient", type=str, default=None, help="Download a single patient by ID")
    parser.add_argument("--list", action="store_true", help="Show download status only")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logs_dir = Path(cfg["paths"].get("logs_dir", "logs"))
    setup_logger(
        "hcc",
        log_file=logs_dir / "download.log",
        console_level=args.log_level,
    )
    logger = get_logger("hcc.download")
    record_run("download", cfg, vars(args))
    log_section(logger, "Google Drive download")

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    drive_cfg = cfg.get("drive", {}) or {}
    patient_files: dict[str, str] = drive_cfg.get("patient_files", {}) or {}
    gdown_fuzzy = bool(drive_cfg.get("gdown_fuzzy", True))

    if not patient_files:
        logger.warning(
            "drive.patient_files is empty in %s. Add patient_id -> Drive file IDs.",
            args.config,
        )

    if args.list:
        _list_status(patient_files, raw_dir)
        return

    logger.info(
        "If downloads fail with network errors, gdown may be blocked; "
        "see module docstring (VPN / manual download)."
    )

    targets = (
        {args.patient: patient_files[args.patient]}
        if args.patient
        else patient_files
    )
    if args.patient and args.patient not in patient_files:
        logger.error("Patient %s not found in drive.patient_files.", args.patient)
        sys.exit(1)

    failed = []
    for pid, fid in targets.items():
        try:
            ok = _download_patient(pid, fid, raw_dir, gdown_fuzzy=gdown_fuzzy)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected download error for %s: %s", pid, exc)
            record_failure("download", pid, exc, logs_dir=logs_dir)
            failed.append(pid)
            continue
        if not ok:
            failed.append(pid)
    if failed:
        logger.error("Some downloads failed: %s", failed)
        sys.exit(2)


if __name__ == "__main__":
    main()
