"""Phase 3: PyRadiomics -> VaRFS -> classical baseline.

Usage
-----
    python scripts/run_radiomics.py --config configs/default.yaml --features-config configs/features.yaml
    python scripts/run_radiomics.py --config configs/default.yaml --log-level DEBUG
    python scripts/run_radiomics.py --config configs/default.yaml --force-radiomics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.baseline_classifier import RadiomicsBaseline, save_metrics  # noqa: E402
from src.features.radiomics_extractor import RadiomicsExtractor  # noqa: E402
from src.features.varfs_selection import VaRFSSelector  # noqa: E402
from src.utils.config import (  # noqa: E402
    ensure_dirs,
    load_config,
    load_features_config,
    set_seed,
)
from src.utils.logger import (  # noqa: E402
    get_logger,
    log_dict,
    log_section,
    setup_logger,
)
from src.utils.run_metadata import record_failure, record_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run radiomics + VaRFS + baseline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--features-config", type=str, default="configs/features.yaml")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--force-radiomics",
        action="store_true",
        help="Delete raw radiomics CSV and re-extract all patients (ignore resume).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    features_cfg = load_features_config(args.features_config)
    ensure_dirs(cfg)
    set_seed(cfg)

    logs_dir = Path(cfg["paths"]["logs_dir"])
    setup_logger(
        "hcc",
        log_file=logs_dir / "radiomics.log",
        console_level=args.log_level,
    )
    logger = get_logger("hcc.radiomics")
    record_run("radiomics", cfg, vars(args))
    log_section(logger, "Radiomics + VaRFS + baseline")
    radiomics_cfg = features_cfg.get("radiomics")
    if not isinstance(radiomics_cfg, dict):
        raise KeyError("Missing required key features_config['radiomics'].")
    log_dict(logger, "config.radiomics", radiomics_cfg)
    varfs_cfg = features_cfg.get("varfs")
    if not isinstance(varfs_cfg, dict):
        raise KeyError("Missing required key features_config['varfs'].")
    log_dict(logger, "config.varfs", varfs_cfg)

    baseline_cfg = features_cfg.get("baseline")
    if not isinstance(baseline_cfg, dict):
        raise KeyError("Missing required key features_config['baseline'].")

    processed_dir = Path(cfg["paths"]["processed_dir"])
    labels_csv = Path(cfg["paths"]["labels_csv"])
    feature_paths = cfg["paths"]["features"]
    raw_features_csv = Path(feature_paths["raw_radiomics_csv"])
    filtered_features_csv = Path(feature_paths["varfs_filtered_csv"])
    selected_features_json = Path(feature_paths["varfs_selected_json"])
    model_dir = Path(cfg["paths"]["model_save_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])

    logger.info("Phase 3.1: PyRadiomics extraction")
    extractor = RadiomicsExtractor(radiomics_cfg)
    raw_features = extractor.extract_all(
        processed_dir,
        labels_csv,
        raw_features_csv,
        force=bool(args.force_radiomics),
    )
    if raw_features.empty:
        logger.error("No radiomics features extracted; aborting.")
        sys.exit(2)

    logger.info("Phase 3.2: VaRFS feature selection")
    selector = VaRFSSelector(
        n_bootstrap=int(varfs_cfg["n_bootstrap"]),
        stability_threshold=float(varfs_cfg["stability_threshold"]),
        correlation_threshold=float(varfs_cfg["correlation_threshold"]),
        top_k_per_iter=int(varfs_cfg["top_k_per_iter"]),
        use_robust_scaler=bool(varfs_cfg["use_robust_scaler"]),
        random_state=int(cfg["seed"]),
    )
    labels = raw_features["label"]
    feature_cols = [c for c in raw_features.columns if c not in ("patient_id", "label")]
    selector.fit(raw_features[["patient_id", "label", *feature_cols]], labels)
    filtered = selector.transform(raw_features)
    filtered.to_csv(filtered_features_csv, index=False)
    logger.info("Saved filtered features -> %s", filtered_features_csv)
    selector.save_selection(selected_features_json)

    logger.info("Phase 3.3: Classical baseline (RandomForest)")
    try:
        baseline = RadiomicsBaseline(baseline_cfg)
        metrics = baseline.train(filtered)
        baseline.save(model_dir / "radiomics_baseline.pkl")
        save_metrics(metrics, results_dir / "baseline_metrics.json")
    except Exception as exc:  # noqa: BLE001
        logger.error("Baseline training failed: %s", exc)
        record_failure("radiomics", "baseline_cv", exc, logs_dir=logs_dir)
        raise

    # Persist OOF predictions (one prediction per patient produced by the
    # fold in which they were the validation set). These are leakage-free
    # and what `run_evaluation.py` should use for ROC plotting.
    if baseline.oof_scores_ is not None:
        oof_df = pd.DataFrame(
            {
                "patient_id": [str(pid) for pid in baseline.oof_pids_],
                "label": list(baseline.oof_labels_),
                "score": baseline.oof_scores_.tolist(),
            }
        )
        oof_df.to_csv(results_dir / "baseline_scores.csv", index=False)
        logger.info(
            "Saved baseline OOF scores -> %s", results_dir / "baseline_scores.csv"
        )
    else:
        logger.warning("Baseline produced no OOF scores; skipping CSV.")


if __name__ == "__main__":
    main()
