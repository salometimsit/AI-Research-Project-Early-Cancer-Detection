"""Concept-based interpretability via TCAV (per-feature + per-family).

Pipeline stage: post-Phase-6 interpretability of the SwinViT track.

Usage
-----
    python scripts/run_tcav.py --config configs/default.yaml --mode both
    python scripts/run_tcav.py --config configs/default.yaml --mode per_feature
    python scripts/run_tcav.py --config configs/default.yaml --mode per_family \\
        --checkpoint models/saved/best_swinvit_model.pth --log-level DEBUG

Outputs
-------
* ``tcav/cavs.npz`` -- raw CAV vectors keyed by concept name.
* ``tcav/train_accuracies.json`` -- linear-classifier accuracy per CAV.
* ``results/tcav_per_feature_scores.json`` and
  ``results/tcav_per_family_scores.json``.
* ``Plots/tcav_per_feature_bar.png`` and ``Plots/tcav_per_family_bar.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import HCCDataset  # noqa: E402
from src.models.swin_vit import SwinViT3D  # noqa: E402
from src.utils.config import ensure_dirs, load_config, load_data_config, set_seed  # noqa: E402
from src.utils.logger import get_logger, log_dict, log_section, setup_logger  # noqa: E402
from src.utils.run_metadata import record_failure, record_run  # noqa: E402
from src.utils.tcav import (  # noqa: E402
    RadiomicsConceptBuilder,
    TCAV3D,
    save_cav_accuracies,
    save_cavs_npz,
)
from src.utils.visualization import plot_tcav_scores  # noqa: E402


def _build_swin(cfg: dict, checkpoint: Path, device: torch.device) -> SwinViT3D:
    swin_cfg = cfg["swin_vit"]
    model = SwinViT3D(
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
    ).to(device)
    bundle = torch.load(checkpoint, map_location=device)
    state = bundle.get("model_state", bundle)
    model.load_state_dict(state)
    model.eval()
    return model


def _scores_with_significance(
    tcav: TCAV3D,
    dataset,
    cavs_real: dict,
    cavs_random: dict,
    max_samples: int | None,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    logger = get_logger("hcc.tcav")
    random_scores = []
    for name, rec in cavs_random.items():
        score = float(tcav.compute_tcav_score(dataset, rec.vector, max_samples=max_samples))
        logger.debug("Random concept %s -> score=%.3f", name, score)
        random_scores.append(score)

    n_tests = len(cavs_real)
    alpha_corrected = float(alpha / n_tests) if n_tests > 0 else float(alpha)

    out: dict[str, dict[str, Any]] = {}
    for name, rec in cavs_real.items():
        score = float(tcav.compute_tcav_score(dataset, rec.vector, max_samples=max_samples))
        p_value = float(TCAV3D.compute_significance(score, random_scores))
        is_significant = bool(p_value < alpha_corrected)
        out[name] = {
            "score": score,
            "p_value": p_value,
            "alpha_corrected": alpha_corrected,
            "is_significant": is_significant,
            "train_accuracy": float(rec.train_accuracy),
            "n_positive": int(rec.n_positive),
            "n_negative": int(rec.n_negative),
        }
        logger.info(
            "%s | score=%.3f | p=%.3f | sig(Bonferroni)=%s | train_acc=%.3f",
            name,
            score,
            p_value,
            is_significant,
            rec.train_accuracy,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="TCAV concept-based interpretability")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SwinViT checkpoint (default: models/saved/best_swinvit_model.pth).",
    )
    parser.add_argument(
        "--mode",
        choices=["per_feature", "per_family", "both"],
        default="both",
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
        log_file=logs_dir / "tcav.log",
        console_level=args.log_level,
    )
    logger = get_logger("hcc.tcav")
    record_run("tcav", cfg, vars(args))

    log_section(logger, f"TCAV | mode={args.mode}")
    log_dict(logger, "config.tcav", cfg.get("tcav", {}))

    target_size = tuple(cfg["swin_vit"]["img_size"])
    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg, dict) else {}
    pre_block = data_cfg.get("preprocessing") or {}
    if not isinstance(pre_block, dict):
        pre_block = {}
    z_cfg_raw = pre_block.get("zscore")
    z_cfg: dict[str, Any] = dict(z_cfg_raw) if isinstance(z_cfg_raw, dict) else {}
    pipeline_zscore_on = bool(z_cfg.get("enabled", True))
    zscore_runtime = not pipeline_zscore_on
    logger.info(
        "TCAV HCCDataset: preprocessing.zscore.enabled=%s -> runtime liver Z-score=%s",
        pipeline_zscore_on,
        zscore_runtime,
    )
    # No training augmentations; Z-score at load only when disk crops are HU-windowed only.
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
        zscore_enabled=zscore_runtime,
        zscore_scope=str(z_cfg.get("scope", "liver_only")),
        zscore_cfg=z_cfg if z_cfg else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint or Path(cfg["paths"]["model_save_dir"]) / "best_swinvit_model.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"SwinViT checkpoint not found: {ckpt}")
    model = _build_swin(cfg, ckpt, device)

    tcav_cfg = cfg["tcav"]
    alpha = float(tcav_cfg.get("significance_threshold", 0.05))
    tcav = TCAV3D(
        model=model,
        target_layer_name=str(tcav_cfg["target_layer"]),
        target_class=int(tcav_cfg["target_class"]),
        device=device,
        cav_c=float(tcav_cfg.get("cav_C", 0.1)),
    )
    logger.info("Hooked SwinViT layer: %s", tcav.target_layer_name)

    log_section(logger, "Extract activations")
    extract_bs = int(
        tcav_cfg.get("extract_batch_size", tcav_cfg.get("batch_size", 1))
    )
    if extract_bs > 2:
        logger.warning(
            "tcav.extract_batch_size=%d is large for 3D activations; using 1 to reduce OOM risk.",
            extract_bs,
        )
        extract_bs = 1
    extract_nw = int(
        tcav_cfg.get("extract_num_workers", tcav_cfg.get("num_workers", 0))
    )
    activations, pids = tcav.extract_activations(
        dataset,
        batch_size=extract_bs,
        num_workers=extract_nw,
    )
    pid_to_index = {pid: i for i, pid in enumerate(pids)}
    logger.info("Activations shape=%s, %d patients", activations.shape, len(pids))

    builder = RadiomicsConceptBuilder(quartile=float(tcav_cfg["quartile_split"]))
    feature_paths = cfg["paths"]["features"]
    radiomics_csv = Path(feature_paths["raw_radiomics_csv"])
    selected_json = Path(feature_paths["varfs_selected_json"])
    labels_csv = Path(cfg["paths"]["labels_csv"])

    plots_dir = Path(cfg["paths"]["plots_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    tcav_dir = Path(cfg["paths"]["tcav_dir"])

    log_section(logger, "Build random concepts (null distribution)")
    random_concepts = builder.build_random_concepts(
        all_pids=pids,
        n_concepts=int(tcav_cfg["n_random_concepts"]),
        size=int(tcav_cfg["random_concept_size"]),
        seed=int(cfg.get("seed", 42)),
    )
    cavs_random = tcav.build_cavs(activations, pid_to_index, random_concepts)

    all_real_cavs: dict = {}
    max_samples = tcav_cfg.get("max_samples")
    max_samples = None if max_samples in (None, "null") else int(max_samples)

    scores_feat: dict[str, dict[str, Any]] = {}
    scores_fam: dict[str, dict[str, Any]] = {}

    if args.mode in ("per_feature", "both"):
        log_section(logger, "Per-feature concepts")
        try:
            per_feature = builder.build_per_feature_concepts(
                radiomics_csv=radiomics_csv,
                selected_features_json=selected_json,
                top_n=int(tcav_cfg["top_n_features"]),
            )
            cavs_feat = tcav.build_cavs(activations, pid_to_index, per_feature)
            all_real_cavs.update(cavs_feat)
            scores_feat = _scores_with_significance(
                tcav, dataset, cavs_feat, cavs_random, max_samples, alpha=alpha
            )
            with open(results_dir / "tcav_per_feature_scores.json", "w", encoding="utf-8") as fh:
                json.dump(scores_feat, fh, indent=2)
            logger.info(
                "Saved per-feature scores -> %s",
                results_dir / "tcav_per_feature_scores.json",
            )
            plot_tcav_scores(
                {k: v["score"] for k, v in scores_feat.items()},
                {k: v["p_value"] for k, v in scores_feat.items()},
                plots_dir / "tcav_per_feature_bar.png",
                title="TCAV (per-feature concepts)",
                significance_threshold=alpha,
                significant={k: bool(v["is_significant"]) for k, v in scores_feat.items()},
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Per-feature TCAV failed: %s", exc)
            record_failure("tcav", "per_feature", exc)

    if args.mode in ("per_family", "both"):
        log_section(logger, "Per-family concepts")
        try:
            per_family, signs = builder.build_per_family_concepts(
                radiomics_csv=radiomics_csv,
                labels_csv=labels_csv if labels_csv.exists() else None,
                families=list(tcav_cfg["feature_families"]),
            )
            log_dict(
                logger,
                "per_family_signs",
                {s.family: s.sign for s in signs},
            )
            cavs_fam = tcav.build_cavs(activations, pid_to_index, per_family)
            all_real_cavs.update(cavs_fam)
            scores_fam = _scores_with_significance(
                tcav, dataset, cavs_fam, cavs_random, max_samples, alpha=alpha
            )
            with open(results_dir / "tcav_per_family_scores.json", "w", encoding="utf-8") as fh:
                json.dump(scores_fam, fh, indent=2)
            logger.info(
                "Saved per-family scores -> %s",
                results_dir / "tcav_per_family_scores.json",
            )
            plot_tcav_scores(
                {k: v["score"] for k, v in scores_fam.items()},
                {k: v["p_value"] for k, v in scores_fam.items()},
                plots_dir / "tcav_per_family_bar.png",
                title="TCAV (per-family concepts)",
                significance_threshold=alpha,
                significant={k: bool(v["is_significant"]) for k, v in scores_fam.items()},
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Per-family TCAV failed: %s", exc)
            record_failure("tcav", "per_family", exc)

    if all_real_cavs:
        all_p_values: dict[str, float] = {}
        all_sig: dict[str, bool] = {}
        if args.mode in ("per_feature", "both"):
            all_p_values.update(
                {k: v["p_value"] for k, v in scores_feat.items()}  # type: ignore[name-defined]
            )
            all_sig.update(
                {k: bool(v["is_significant"]) for k, v in scores_feat.items()}  # type: ignore[name-defined]
            )
        if args.mode in ("per_family", "both"):
            all_p_values.update(
                {k: v["p_value"] for k, v in scores_fam.items()}  # type: ignore[name-defined]
            )
            all_sig.update(
                {k: bool(v["is_significant"]) for k, v in scores_fam.items()}  # type: ignore[name-defined]
            )
        save_cavs_npz(all_real_cavs, tcav_dir / "cavs.npz")
        tcav_params = {
            "cav_C": float(tcav_cfg.get("cav_C", 0.1)),
            "permutation_sets": int(tcav_cfg.get("permutation_sets", 30)),
            "significance_threshold": float(
                tcav_cfg.get("significance_threshold", 0.05)
            ),
            "multiple_testing": "bonferroni",
        }
        save_cav_accuracies(
            all_real_cavs,
            tcav_dir / "train_accuracies.json",
            params=tcav_params,
            p_vals=all_p_values,
            is_significant=all_sig,
        )

    log_section(logger, "TCAV done")


if __name__ == "__main__":
    main()
