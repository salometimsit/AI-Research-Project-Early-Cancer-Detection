"""Phase 6: Evaluate baseline / SwinViT / fused; export plots and heatmaps.

This script is **leakage-free**: it reads the out-of-fold (OOF) score
CSVs produced during k-fold training and joins them strictly on
``patient_id``, so the ROC / metric numbers reflect true held-out
performance.

Heatmaps are exported per patient from the canonical
``best_swinvit_model.pth`` checkpoint (the highest-AUC fold). They are
purely qualitative artifacts; metrics never reuse them.

Usage
-----
    python scripts/run_evaluation.py --config configs/default.yaml
    python scripts/run_evaluation.py --config configs/default.yaml --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import HCCDataset  # noqa: E402
from src.models.swin_vit import SwinViT3D  # noqa: E402
from src.utils.config import ensure_dirs, load_config, load_data_config, set_seed  # noqa: E402
from src.utils.logger import (  # noqa: E402
    get_logger,
    log_dict,
    log_section,
    setup_logger,
)
from src.utils.run_metadata import record_failure, record_run  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    plot_ablation_comparison,
    plot_attention_heatmap,
    plot_roc_curve,
    plot_training_curves,
)


def _safe_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute AUC, sensitivity, specificity, PR-AUC, precision, and F1.

    Args:
        y_true: 1D ground-truth label vector.
        scores: 1D predicted-probability vector aligned with ``y_true``.
        threshold: Decision threshold from ``cfg.evaluation.threshold``.

    Returns:
        Metric dict. ``auc_roc`` and ``precision_recall_auc`` are ``nan``
        when a score is undefined (e.g. a single class in ``y_true``).
        ``sensitivity`` / ``specificity`` are ``nan`` when the corresponding
        denominator is zero (no positives or no negatives in ground truth
        at the chosen threshold layout). ``f1_score`` uses the same
        threshold-derived precision and recall (sensitivity); it is ``nan``
        when either input to the harmonic mean is ``nan``.
    """
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    prec = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    if np.isnan(prec) or np.isnan(sens):
        f1 = float("nan")
    elif prec + sens == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * prec * sens / (prec + sens))
    try:
        auc_score = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc_score = float("nan")
    if np.unique(y_true).size < 2:
        pr_auc = float("nan")
    else:
        try:
            pr_auc = float(average_precision_score(y_true, scores))
        except ValueError:
            pr_auc = float("nan")
    return {
        "auc_roc": auc_score,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "precision_recall_auc": pr_auc,
        "threshold": float(threshold),
    }


def _filter_metrics(
    metrics: dict[str, float],
    keep: list[str] | None,
) -> dict[str, float]:
    """Whitelist filter over ``_safe_metrics`` output.

    ``threshold`` is always kept because it's metadata (decision threshold
    used to compute the binary metrics), not a metric itself. Pass
    ``None`` or an empty list to disable filtering.
    """
    if not keep:
        return metrics
    allowed = set(keep) | {"threshold"}
    return {k: v for k, v in metrics.items() if k in allowed}


def _load_oof(path: Path, name: str) -> pd.DataFrame | None:
    """Load an OOF score CSV; return ``None`` (with a warning) if missing."""
    logger = get_logger("hcc.evaluation")
    if not path.exists():
        logger.warning("Missing %s OOF scores: %s", name, path)
        return None
    df = pd.read_csv(path).dropna(subset=["score"])
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.drop_duplicates(subset=["patient_id"])
    df = df.set_index("patient_id")
    return df[["label", "score"]]


def _align_scores_by_pid(
    oofs: dict[str, pd.DataFrame],
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Inner-join all OOF tables on ``patient_id``.

    Args:
        oofs: Mapping ``{model_name: DataFrame[label, score]}``.

    Returns:
        Tuple ``(patient_ids, labels, scores_by_model)`` where
        ``patient_ids`` is the list of pids present in *every* OOF
        table, ``labels`` is the shared label vector, and
        ``scores_by_model`` maps each model name to its score array
        in the same order as ``patient_ids``.
    """
    if not oofs:
        return [], np.array([], dtype=int), {}
    pids = None
    for df in oofs.values():
        pids = set(df.index) if pids is None else (pids & set(df.index))
    pids_sorted = sorted(pids or [])
    if not pids_sorted:
        return [], np.array([], dtype=int), {}
    labels: np.ndarray | None = None
    scores: dict[str, np.ndarray] = {}
    for name, df in oofs.items():
        df = df.loc[pids_sorted]
        if labels is None:
            labels = df["label"].astype(int).to_numpy()
        else:
            np.testing.assert_array_equal(
                df["label"].astype(int).to_numpy(),
                labels,
                err_msg=f"Label mismatch for {name} after pid alignment.",
            )
        scores[name] = df["score"].astype(np.float64).to_numpy()
    return pids_sorted, labels, scores


def _build_swin(cfg: dict[str, Any]) -> SwinViT3D:
    swin_cfg = cfg["swin_vit"]
    return SwinViT3D(
        img_size=tuple(swin_cfg["img_size"]),
        patch_size=tuple(swin_cfg["patch_size"]),
        in_channels=int(swin_cfg["in_channels"]),
        embed_dim=int(swin_cfg["embed_dim"]),
        depths=tuple(swin_cfg["depths"]),
        num_heads=tuple(swin_cfg["num_heads"]),
        window_size=tuple(swin_cfg["window_size"]),
        mlp_ratio=float(swin_cfg["mlp_ratio"]),
        drop_path_rate=float(swin_cfg["drop_path_rate"]),
        dropout=float(swin_cfg["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
        use_checkpoint=bool(swin_cfg["use_checkpoint"]),
    )


@torch.no_grad()
def _export_attention_heatmaps(cfg: dict[str, Any], data_cfg: dict[str, Any]) -> None:
    """Save per-patient SwinViT self-attention heatmaps as NIfTI.

    Uses the canonical ``best_swinvit_model.pth`` checkpoint. This is
    qualitative output only; the OOF metrics computed elsewhere don't
    depend on it.
    """
    logger = get_logger("hcc.evaluation")
    ckpt = Path(cfg["paths"]["model_save_dir"]) / "best_swinvit_model.pth"
    if not ckpt.exists():
        logger.warning("No best_swinvit_model.pth; skipping heatmap export.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_swin(cfg).to(device)
    bundle = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(bundle["model_state"])
    model.eval()

    target_size = tuple(cfg["swin_vit"]["img_size"])
    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg, dict) else {}
    # Heatmaps must match on-disk CT geometry; never apply training augmentations.
    dataset = HCCDataset(
        processed_dir=cfg["paths"]["processed_dir"],
        labels_csv=cfg["paths"]["labels_csv"],
        target_size=target_size,
        transform=None,
        allow_missing=bool(dataset_cfg.get("allow_missing", True)),
        augmentation_seed_base=(
            int(dataset_cfg["augmentation_seed_base"])
            if dataset_cfg.get("augmentation_seed_base") is not None
            else None
        ),
    )
    attention_dir = Path(cfg["paths"]["attention_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    for i in range(len(dataset)):
        sample = dataset[i]
        pid = sample["patient_id"]
        image = sample["image"].unsqueeze(0).to(device).float()
        try:
            attn = model.get_attention_maps(image).cpu().numpy()[0]
            meta_path = processed_dir / pid / "crop_metadata.json"
            volume_path = processed_dir / pid / "before.nii.gz"
            if not meta_path.exists() or not volume_path.exists():
                logger.warning(
                    "Missing crop metadata or volume for %s; skipping heatmap.",
                    pid,
                )
                continue
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            volume = nib.load(str(volume_path))
            plot_attention_heatmap(
                volume.get_fdata(),
                attn,
                meta,
                attention_dir / f"{pid}_attention.nii.gz",
                affine=volume.affine,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Heatmap save failed for %s: %s", pid, exc)
            record_failure("evaluation", pid, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation + interpretability")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip per-patient attention heatmap export (faster sanity runs).",
    )
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
        log_file=logs_dir / "evaluation.log",
        console_level=args.log_level,
    )
    logger = get_logger("hcc.evaluation")
    record_run("evaluation", cfg, vars(args))
    log_section(logger, "Evaluation")
    log_dict(logger, "config.evaluation", cfg.get("evaluation", {}))

    results_dir = Path(cfg["paths"]["results_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])
    feature_paths = cfg["paths"]["features"]
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    logger.info("Using decision threshold = %.3f", threshold)

    # --- Load OOF score tables (leakage-free) and align by patient_id
    oof_sources = {
        "radiomics": results_dir / "baseline_scores.csv",
        "swinvit": Path(feature_paths["swinvit_oof_csv"]),
        "fused": Path(feature_paths["fusion_oof_csv"]),
    }
    oofs = {name: df for name, df in (
        (n, _load_oof(p, n)) for n, p in oof_sources.items()
    ) if df is not None}
    if not oofs:
        raise RuntimeError(
            "No OOF score CSVs found. Did you run run_radiomics.py and "
            "run_training.py?"
        )

    pids, labels, score_arrays = _align_scores_by_pid(oofs)
    if not pids:
        sys.exit("No overlapping PIDs found across models. Exiting.")
    logger.info(
        "Aligned %d patients across models: %s",
        len(pids),
        list(score_arrays.keys()),
    )

    metrics_keep = cfg.get("evaluation", {}).get("metrics")
    metrics: dict[str, dict[str, float]] = {
        name: _filter_metrics(_safe_metrics(labels, scores, threshold), metrics_keep)
        for name, scores in score_arrays.items()
    }
    with open(results_dir / "evaluation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics -> %s", results_dir / "evaluation_metrics.json")

    # --- ROC plot (uses the same pid-aligned score arrays)
    plot_roc_curve(
        labels.tolist(),
        {k: v.tolist() for k, v in score_arrays.items()},
        plots_dir / "roc_comparison.png",
    )

    # --- Ablation bar chart
    plot_ablation_comparison(metrics, plots_dir / "ablation_bar_chart.png")

    # --- Training curves: pick fold-1 histories when available
    history_paths: dict[str, Path] = {}
    for name in ("swinvit", "fusion"):
        candidates = sorted(results_dir.glob(f"training_history_{name}_fold*.json"))
        if candidates:
            history_paths[name] = candidates[0]
    if history_paths:
        plot_training_curves(history_paths, plots_dir / "training_curves.png")
    else:
        logger.warning("No per-fold training histories found; skipping curves plot.")

    # --- Heatmap export (qualitative only)
    # Config gives the default; --skip-heatmaps on the CLI can override to skip.
    export_heatmaps_default = bool(
        cfg.get("evaluation", {}).get("export_attention_heatmaps", True)
    )
    if export_heatmaps_default and not args.skip_heatmaps:
        _export_attention_heatmaps(cfg, data_cfg)
    elif not export_heatmaps_default:
        logger.info("Skipping heatmap export (evaluation.export_attention_heatmaps=false).")


if __name__ == "__main__":
    main()
