"""Patient-level k-fold CV training for the SwinViT (Phase 4) track.

Pipeline stage: J (SwinViT k-fold CV; OOF scores and deep features for L).

The single public entry point is :func:`run_swinvit_cv`. The fusion
counterpart lives in :mod:`src.models.fusion_training` (it depends on
the OOF deep features produced here).

This module also exposes the small helpers
(:func:`build_swin`, :func:`focal_alpha_from_labels`,
:func:`make_focal_loss`, :func:`make_kfold`) that fusion training
re-imports.

Inputs
------
* Full merged config from :func:`src.utils.config.load_config`.
* ``model_cv.cuda_cleanup_between_folds`` — optional GPU/RAM cleanup after each fold.

Outputs
-------
* Per-fold checkpoints and OOF CSVs under ``cfg["paths"]``.

Side effects
------------
* Calls :func:`src.utils.config.set_seed` at the start of :func:`run_swinvit_cv`.
* Per-fold training failures append to ``logs/swinvit_cv_failures.jsonl`` via
  :func:`src.utils.run_metadata.record_failure`.

Failure modes
-------------
* All folds fail during training -> no deep features -> early return without
  writing feature CSVs.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.data.dataset import HCCDataset, get_3d_augmentation, get_weighted_sampler
from src.models.classifier import FocalLoss
from src.models.pretrained_swin import load_swin_encoder_pretrained
from src.models.swin_vit import SwinViT3D
from src.models.trainer import Trainer
from src.utils.config import set_seed
from src.utils.logger import get_logger, log_section
from src.utils.run_metadata import record_failure

_PHASE: str = "swinvit_cv"


def _maybe_load_swin_pretrained(model: SwinViT3D, cfg: dict[str, Any]) -> None:
    """Warm-start ``model.encoder`` from ``swin_vit.pretrained_weights`` when set.

    If ``pretrained_weights`` is null/empty or the path is missing, logs and
    keeps random initialization (does not abort CV).
    """
    logger = get_logger("hcc.training")
    swin_cfg = cfg["swin_vit"]
    raw = swin_cfg["pretrained_weights"]
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        logger.info(
            "SwinViT encoder: training from scratch (pretrained_weights null or empty)."
        )
        return
    path = Path(str(raw).strip()).expanduser()
    path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if not path.is_file():
        logger.warning(
            "swin_vit.pretrained_weights points to missing file (%s); "
            "training encoder from scratch.",
            path,
        )
        return
    strict = bool(swin_cfg["pretrained_strict"])
    unwrap_keys = tuple(str(k) for k in swin_cfg["pretrained_unwrap_keys"])
    summary = load_swin_encoder_pretrained(
        model=model,
        ckpt_path=path,
        strict=strict,
        map_location=None,
        pretrained_load_target=str(swin_cfg["pretrained_load_target"]),
        unwrap_keys=unwrap_keys,
        unwrap_max_depth=int(swin_cfg["pretrained_unwrap_max_depth"]),
        unknown_prefix_samples_max=int(swin_cfg["pretrained_unknown_prefix_samples"]),
        shape_mismatch_preview=int(swin_cfg["pretrained_shape_mismatch_preview"]),
    )
    logger.info(
        "Pretrained encoder checkpoint applied | loaded_tensors=%d/%d",
        summary["n_loaded"],
        summary["total_encoder_keys"],
    )


def build_swin(cfg: dict[str, Any]) -> SwinViT3D:
    """Construct a :class:`SwinViT3D` from the ``swin_vit`` config block.

    Optional pretrained weights are applied afterward by
    :func:`_maybe_load_swin_pretrained` inside :func:`run_swinvit_cv`, not here.
    """
    swin_cfg = cfg["swin_vit"]
    model_cfg = cfg["model"]
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
        num_classes=int(model_cfg["num_classes"]),
        use_checkpoint=bool(swin_cfg["use_checkpoint"]),
    )


def focal_alpha_from_labels(labels: np.ndarray, fallback_alpha: float) -> float:
    """Class-1 weight for FocalLoss = ``N_neg / (N_neg + N_pos)``.

    Falls back to ``fallback_alpha`` if either class is missing in the
    training fold.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = n_pos + n_neg
    if total == 0 or n_pos == 0 or n_neg == 0:
        return float(fallback_alpha)
    return float(n_neg / total)


def make_focal_loss(cfg: dict[str, Any], train_labels: np.ndarray) -> FocalLoss:
    """Build a :class:`FocalLoss` with ``alpha`` derived from training data."""
    fl_cfg = cfg["model"]["focal_loss"]
    fallback_alpha = float(fl_cfg["fallback_alpha"])
    alpha = focal_alpha_from_labels(train_labels, fallback_alpha=fallback_alpha)
    gamma = float(fl_cfg["gamma"])
    reduction = str(fl_cfg["reduction"])
    logger = get_logger("hcc.training")
    logger.info(
        "FocalLoss alpha=%.4f (n_neg=%d, n_pos=%d), gamma=%.2f",
        alpha,
        int((train_labels == 0).sum()),
        int((train_labels == 1).sum()),
        gamma,
    )
    return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)


def make_kfold(cfg: dict[str, Any], labels: np.ndarray) -> Any:
    """Build a CV splitter honouring ``evaluation.cross_validation.strategy``.

    Returns:
        * :class:`StratifiedKFold` when ``strategy == "kfold"`` (default),
          using ``model_cv.n_folds`` capped at the smallest-class count.
        * :class:`LeaveOneOut` when ``strategy == "leave_one_patient_out"``;
          each validation fold contains one patient, so per-fold AUC is
          undefined — best-fold selection falls back to ``val_loss``.

    Both returned objects expose ``.split(X, y)`` and ``.get_n_splits(X, y)``.
    """
    strategy = str(
        cfg.get("evaluation", {})
        .get("cross_validation", {})
        .get("strategy", "kfold")
    ).lower()
    if strategy == "leave_one_patient_out":
        return LeaveOneOut()
    if strategy != "kfold":
        get_logger("hcc.training").warning(
            "Unknown evaluation.cross_validation.strategy=%r; falling back to kfold.",
            strategy,
        )
    cv_cfg = cfg["model_cv"]
    n_folds = int(cv_cfg["n_folds"])
    smallest_class = int(np.bincount(labels).min()) or n_folds
    n_folds = max(2, min(n_folds, smallest_class))
    return StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=int(cfg["seed"]),
    )


def _make_loaders(
    cfg: dict[str, Any],
    dataset: HCCDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    """Build (train, val) :class:`DataLoader`s with optional weighted sampler."""
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    sampler = None
    cv_cfg = cfg["model_cv"]
    if bool(cv_cfg["weighted_sampler"]):
        # Inject a fold-restricted view into get_weighted_sampler without
        # constructing a full new HCCDataset (which would re-read disk).
        sub = HCCDataset.__new__(HCCDataset)
        sub.samples = [dataset.samples[i] for i in train_idx]
        sub.labels = dataset.labels[train_idx]
        sampler = get_weighted_sampler(sub)
    train_loader = DataLoader(
        train_subset,
        batch_size=int(cfg["training"]["batch_size"]),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(cv_cfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cv_cfg["num_workers"]),
    )
    return train_loader, val_loader


@torch.no_grad()
def _swin_predict(
    model: SwinViT3D,
    dataset: HCCDataset,
    indices: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs[:, 1], deep_features)`` for the given dataset indices."""
    model.eval()
    probs: list[float] = []
    deeps: list[np.ndarray] = []
    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device).float()
        logits, deep = model(image)
        probs.append(float(torch.softmax(logits, dim=-1)[0, 1].item()))
        deeps.append(deep.squeeze(0).cpu().numpy())
    return np.asarray(probs, dtype=np.float64), np.stack(deeps).astype(np.float32)


def _maybe_cuda_reclaim(enabled: bool) -> None:
    """Run GC and optionally flush the CUDA allocator cache."""
    if not enabled:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_swinvit_cv(cfg: dict[str, Any]) -> None:
    """Run patient-level k-fold CV for the SwinViT track.

    Persists per-fold checkpoints, promotes the best fold to
    ``best_swinvit_model.pth``, and writes both OOF score and OOF deep
    feature CSVs (paths from ``cfg["paths"]["features"]``).

    Before each fold's training loop, the encoder may be warm-started from
    ``cfg["swin_vit"]["pretrained_weights"]`` (see :mod:`src.models.pretrained_swin`).
    """
    set_seed(cfg)
    logger = get_logger("hcc.training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_cfg = cfg["model_cv"]
    logs_dir = Path(cfg["paths"]["logs_dir"])
    cuda_cleanup_between_folds = bool(cv_cfg["cuda_cleanup_between_folds"])

    target_size = tuple(cfg["swin_vit"]["img_size"])
    transform = get_3d_augmentation(cfg)
    dataset = HCCDataset(
        processed_dir=cfg["paths"]["processed_dir"],
        labels_csv=cfg["paths"]["labels_csv"],
        transform=transform,
        target_size=target_size,
    )
    eval_dataset = HCCDataset(  # no augmentation, used for OOF + deep export
        processed_dir=cfg["paths"]["processed_dir"],
        labels_csv=cfg["paths"]["labels_csv"],
        transform=None,
        target_size=target_size,
    )
    pids = [s[0] for s in dataset.samples]
    labels = dataset.labels

    model_dir = Path(cfg["paths"]["model_save_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = cfg["paths"]["features"]
    oof_csv_path = Path(feature_paths["swinvit_oof_csv"])
    deep_csv_path = Path(feature_paths["deep_features_csv"])
    oof_csv_path.parent.mkdir(parents=True, exist_ok=True)
    deep_csv_path.parent.mkdir(parents=True, exist_ok=True)
    skf = make_kfold(cfg, labels)
    total_folds = int(skf.get_n_splits(np.zeros(len(labels)), labels))
    cv_strategy = (
        cfg.get("evaluation", {}).get("cross_validation", {}).get("strategy", "kfold")
    )
    logger.info(
        "Phase 4: SwinViT CV (strategy=%s, n_folds=%d) on %s",
        cv_strategy, total_folds, device,
    )

    oof_scores = np.full(len(labels), np.nan)
    oof_deep: dict[int, np.ndarray] = {}
    fold_aucs: list[float] = []
    best_fold_idx = -1
    best_fold_auc = -float("inf")
    best_fold_loss = float("inf")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        log_section(logger, f"SwinViT fold {fold}/{total_folds}", char="-")
        train_loader, val_loader = _make_loaders(cfg, dataset, train_idx, val_idx)
        model = build_swin(cfg)
        _maybe_load_swin_pretrained(model, cfg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        loss_fn = make_focal_loss(cfg, labels[train_idx])
        trainer = Trainer(model, optimizer, loss_fn, device, cfg)

        fold_ckpt = model_dir / f"swinvit_fold{fold}.pth"
        history_path = results_dir / f"training_history_swinvit_fold{fold}.json"
        try:
            summary = trainer.fit(train_loader, val_loader, fold_ckpt, history_path)
        except Exception as exc:  # noqa: BLE001 -- per-fold isolation is intentional
            logger.error("!!! Fold %d failed with error: %s !!!", fold, exc)
            record_failure(_PHASE, f"fold_{fold}", exc, logs_dir=logs_dir)
            del model, optimizer, trainer, train_loader, val_loader
            _maybe_cuda_reclaim(cuda_cleanup_between_folds)
            continue

        logger.info("Fold %d summary: %s", fold, summary)
        val_auc = float(summary["best_val_auc"])
        val_loss = float(summary["best_val_loss"])
        fold_aucs.append(val_auc)

        # Mirror fusion_training: when AUC is finite, pick best by AUC.
        # When every epoch of a fold had a single-class val set (AUC=nan ->
        # trainer leaves best_val_auc at -inf), fall back to val_loss so the
        # canonical checkpoint still gets promoted.
        if np.isfinite(val_auc) and val_auc > best_fold_auc:
            best_fold_auc = val_auc
            best_fold_loss = val_loss
            best_fold_idx = fold
        elif not np.isfinite(val_auc) and val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_idx = fold

        bundle = torch.load(fold_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(bundle["model_state"])
        model.eval()
        scores, deeps = _swin_predict(model, eval_dataset, val_idx.tolist(), device)
        oof_scores[val_idx] = scores
        for i, vi in enumerate(val_idx):
            oof_deep[int(vi)] = deeps[i]

        del model, optimizer, trainer, train_loader, val_loader
        _maybe_cuda_reclaim(cuda_cleanup_between_folds)

    if not oof_deep:
        logger.error(
            "No deep features were extracted. Check whether any CV fold completed "
            "successfully. Aborting OOF / deep feature export."
        )
        return

    if best_fold_idx > 0:
        best_path = model_dir / f"swinvit_fold{best_fold_idx}.pth"
        canonical = model_dir / "best_swinvit_model.pth"
        map_location = str(cfg["model_cv"]["checkpoint_map_location"])
        torch.save(
            torch.load(best_path, map_location=map_location, weights_only=True),
            canonical,
        )
        logger.info(
            "Promoted fold %d (AUC=%.3f) -> %s",
            best_fold_idx,
            best_fold_auc,
            canonical,
        )
        torch.save(
            torch.load(best_path, map_location=map_location, weights_only=True),
            model_dir / "swinvit_deep_extractor.pth",
        )

    pid_strs = [str(p) for p in pids]
    pd.DataFrame(
        {"patient_id": pid_strs, "label": labels, "score": oof_scores}
    ).to_csv(oof_csv_path, index=False)
    logger.info("Saved SwinViT OOF scores -> %s", oof_csv_path.resolve())

    deep_dim = next(iter(oof_deep.values())).shape[0]
    deep_rows: list[dict[str, Any]] = []
    for i, pid in enumerate(pids):
        vec = oof_deep.get(i, np.full(deep_dim, np.nan, dtype=np.float32))
        row: dict[str, Any] = {"patient_id": str(pid), "label": int(labels[i])}
        row.update({f"deep_{k}": float(v) for k, v in enumerate(vec)})
        deep_rows.append(row)
    pd.DataFrame(deep_rows).to_csv(deep_csv_path, index=False)
    logger.info("Saved OOF deep features -> %s", deep_csv_path.resolve())
    logger.info(
        "SwinViT CV fold AUCs: %s (mean=%.3f)",
        fold_aucs,
        float(np.nanmean(fold_aucs)) if fold_aucs else float("nan"),
    )

