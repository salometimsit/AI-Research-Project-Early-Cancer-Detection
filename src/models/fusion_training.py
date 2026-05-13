"""Patient-level k-fold CV training for the cross-attention fusion track.

Pipeline stage: L (Phase 5 of the V4 design).

Inputs
------
* OOF deep features (``cfg["paths"]["features"]["deep_features_csv"]``).
* VaRFS-filtered radiomics (``cfg["paths"]["features"]["varfs_filtered_csv"]``).
* ``cfg["fusion_training"]`` — CSV column names, in-memory batch keys,
  ``merge_coverage_warn_fraction``, ``positive_class_index``.
* ``cfg["model"]["num_classes"]`` — must match the classifier output width.

Outputs
-------
* ``models/saved/fused_cross_attention_fold{i}.pth`` per fold (schema:
  ``{"fusion_state", "classifier_state", "radio_dim", "deep_dim", "cfg"}``,
  consumed by :mod:`scripts.run_inference`).
* ``models/saved/fused_cross_attention.pth`` -- copy of the best fold.
* ``data/fusion_oof_scores.csv`` for :mod:`scripts.run_evaluation`.
* ``results/training_history_fusion_fold{i}.json`` per fold.

Side effects
------------
* Calls :func:`src.utils.config.set_seed`.
* Records per-fold failures via :func:`src.utils.run_metadata.record_failure`.

Failure modes
-------------
* Empty radiomics/deep merge -> ``RuntimeError``.
* All-NaN deep features -> ``RuntimeError``.
* CUDA OOM in a fold -> logged + recorded; the loop continues.
"""

from __future__ import annotations

import functools
import gc
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.models.classifier import HCCClassifier
from src.models.cv_training import make_focal_loss, make_kfold
from src.models.fusion import CrossAttentionFusion
from src.models.trainer import Trainer
from src.utils.config import set_seed
from src.utils.logger import get_logger, log_section
from src.utils.run_metadata import record_failure

logger = get_logger(__name__)

_PHASE: str = "fusion_training"


def _maybe_cuda_reclaim(enabled: bool) -> None:
    """Run GC and optionally flush the CUDA allocator cache between folds."""
    if not enabled:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _FusionPipeline(nn.Module):
    """Bundle ``CrossAttentionFusion`` + ``HCCClassifier`` into one module.

    Wrapping the two sub-modules lets us reuse the project ``Trainer``
    without duplicating its training loop.
    """

    def __init__(self, fusion: CrossAttentionFusion, classifier: HCCClassifier) -> None:
        super().__init__()
        self.fusion = fusion
        self.classifier = classifier

    def forward(self, radio: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fusion(radio, deep))


class _FeatureDictDataset(Dataset):
    """In-memory dict-style dataset over (radiomics, deep, label) triples.

    Memory note:
        The full radiomics + deep feature matrices live on the **CPU** as
        torch tensors; the ``DataLoader`` ships mini-batches to the GPU.
        This relies on the assumption that the merged feature matrix fits
        comfortably in RAM (<100 MB for a few hundred patients). Do NOT
        copy this in-memory pattern to volumetric CT tensors -- those must
        stream from disk via :class:`src.data.dataset.HCCDataset`.
    """

    def __init__(
        self,
        x_radio: np.ndarray,
        x_deep: np.ndarray,
        y: np.ndarray,
        radio_key: str,
        deep_key: str,
        label_key: str,
    ) -> None:
        self.x_radio = torch.from_numpy(x_radio).float()
        self.x_deep = torch.from_numpy(x_deep).float()
        self.y = torch.from_numpy(y).long()
        self._radio_key = radio_key
        self._deep_key = deep_key
        self._label_key = label_key

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            self._radio_key: self.x_radio[idx],
            self._deep_key: self.x_deep[idx],
            self._label_key: self.y[idx],
        }


def _fusion_forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    radio_key: str,
    deep_key: str,
    label_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Trainer`` forward adapter for the fusion pipeline."""
    logits = model(batch[radio_key].float(), batch[deep_key].float())
    return logits, batch[label_key].long()


def _load_and_align_features(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load radiomics + deep CSVs and align them on (patient_id, label)."""
    logger.debug("Entering _load_and_align_features")
    ft = cfg["fusion_training"]
    pid_col = str(ft["csv_patient_id_column"])
    label_col = str(ft["csv_label_column"])
    warn_frac = float(ft["merge_coverage_warn_fraction"])

    fp = cfg["paths"]["features"]
    radio_df = pd.read_csv(fp["varfs_filtered_csv"])
    deep_df = pd.read_csv(fp["deep_features_csv"])
    radio_df[pid_col] = radio_df[pid_col].astype(str)
    deep_df[pid_col] = deep_df[pid_col].astype(str)

    merged = radio_df.merge(deep_df, on=[pid_col, label_col], how="inner")
    if merged.empty:
        raise RuntimeError("Cannot align radiomics + deep features (empty merge).")

    deep_cols = [c for c in deep_df.columns if c not in (pid_col, label_col)]
    n_pre = len(merged)
    merged = merged.dropna(subset=deep_cols)
    if merged.empty:
        logger.error(
            "All merged rows have NaN deep features after inner merge. "
            "Check SwinViT Phase 4 OOF / deep_features_csv output."
        )
        raise RuntimeError("Empty deep feature merge (all NaN in deep columns).")
    if len(merged) < len(radio_df) * warn_frac:
        logger.warning(
            "Fewer than %.0f%% of radiomics patients remain after merge/dropna "
            "(n_merged=%d vs n_radio=%d). Check patient_id alignment and SwinViT export.",
            warn_frac * 100,
            len(merged),
            len(radio_df),
        )
    if n_pre - len(merged):
        logger.warning("Dropped %d rows with NaN deep features.", n_pre - len(merged))

    radio_cols = [c for c in radio_df.columns if c not in (pid_col, label_col)]
    n_radio_nan = int(merged[radio_cols].isna().sum().sum())
    if n_radio_nan:
        logger.warning(
            "Filling %d NaN radiomics values with 0.0 (check radiomics_extractor).",
            n_radio_nan,
        )

    x_radio = merged[radio_cols].fillna(0.0).to_numpy(dtype=np.float32)
    x_deep = merged[deep_cols].to_numpy(dtype=np.float32)
    y = merged[label_col].astype(int).to_numpy()
    pids = merged[pid_col].astype(str).tolist()
    logger.debug(
        "Exiting _load_and_align_features (n=%d, radio_dim=%d, deep_dim=%d)",
        len(y), x_radio.shape[1], x_deep.shape[1],
    )
    return x_radio, x_deep, y, pids


def _build_models(
    cfg: dict[str, Any], radio_dim: int, deep_dim: int, device: torch.device,
) -> tuple[CrossAttentionFusion, HCCClassifier, _FusionPipeline]:
    """Construct fusion + classifier and wire them through ``_FusionPipeline``."""
    logger.debug("Entering _build_models(radio=%d, deep=%d)", radio_dim, deep_dim)
    ca_cfg = cfg["cross_attention"]
    fusion = CrossAttentionFusion(
        radiomics_dim=radio_dim,
        deep_dim=deep_dim,
        num_heads=int(ca_cfg["num_heads"]),
        hidden_dim=int(ca_cfg["hidden_dim"]),
        dropout=float(ca_cfg["dropout"]),
    )
    model_cfg = cfg["model"]
    classifier = HCCClassifier(
        fused_dim=fusion.output_dim,
        dropout=float(model_cfg["dropout"]),
        num_classes=int(model_cfg["num_classes"]),
    )
    pipeline = _FusionPipeline(fusion, classifier).to(device)
    logger.debug("Exiting _build_models")
    return fusion, classifier, pipeline


def _make_loaders(
    cfg: dict[str, Any], dataset: Dataset, train_idx: np.ndarray, val_idx: np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val :class:`DataLoader` for one fold (val is unshuffled)."""
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["model_cv"]["num_workers"])
    train = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=bs,
                       shuffle=True, num_workers=nw)
    val = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=bs,
                     shuffle=False, num_workers=nw)
    return train, val


def _save_fold_checkpoint(
    trainer_ckpt: Path, radio_dim: int, deep_dim: int,
    cfg: dict[str, Any], fold_path: Path,
) -> None:
    """Re-save the trainer checkpoint in the schema ``run_inference`` expects."""
    map_location = str(cfg["model_cv"]["checkpoint_map_location"])
    bundle = torch.load(trainer_ckpt, map_location=map_location, weights_only=True)
    state = bundle["model_state"]
    payload = {
        "fusion_state": {
            k.removeprefix("fusion."): v
            for k, v in state.items() if k.startswith("fusion.")
        },
        "classifier_state": {
            k.removeprefix("classifier."): v
            for k, v in state.items() if k.startswith("classifier.")
        },
        "radio_dim": int(radio_dim),
        "deep_dim": int(deep_dim),
        "cfg": cfg,
    }
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, fold_path)
    logger.info("Saved fusion fold checkpoint -> %s", fold_path)


@torch.no_grad()
def _predict_val_scores(
    pipeline: _FusionPipeline,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    positive_class_index: int,
    radio_key: str,
    deep_key: str,
) -> np.ndarray:
    """Return positive-class probabilities for ``val_loader`` in load order."""
    pipeline.eval()
    chunks: list[np.ndarray] = []
    for batch in val_loader:
        radio = batch[radio_key].float().to(device)
        deep = batch[deep_key].float().to(device)
        logits = pipeline(radio, deep)
        assert logits.shape[-1] == num_classes, (
            "Classifier output width must match cfg model.num_classes"
        )
        probs = torch.softmax(logits, dim=-1)[:, positive_class_index]
        chunks.append(probs.cpu().numpy().astype(np.float64))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)


def _train_one_fold(
    cfg: dict[str, Any],
    fold: int,
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_labels: np.ndarray,
    radio_dim: int,
    deep_dim: int,
    model_dir: Path,
    results_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Train one fusion fold via the project ``Trainer``.

    Returns:
        ``(val_scores, best_val_loss)`` -- positive-class probabilities for
        the validation indices (in ``val_idx`` order) and the best
        validation loss reached during the fold.
    """
    logger.debug("Entering _train_one_fold(fold=%d)", fold)
    ft = cfg["fusion_training"]
    radio_key = str(ft["batch_radio_key"])
    deep_key = str(ft["batch_deep_key"])
    label_key = str(ft["batch_label_key"])
    num_classes = int(cfg["model"]["num_classes"])
    pos_idx = int(ft["positive_class_index"])
    if not (0 <= pos_idx < num_classes):
        raise ValueError(
            f"fusion_training.positive_class_index={pos_idx} invalid for "
            f"model.num_classes={num_classes}"
        )

    train_loader, val_loader = _make_loaders(cfg, dataset, train_idx, val_idx)
    fusion, classifier, pipeline = _build_models(cfg, radio_dim, deep_dim, device)

    optimizer = torch.optim.Adam(
        pipeline.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    loss_fn = make_focal_loss(cfg, train_labels)
    forward_fn = functools.partial(
        _fusion_forward,
        radio_key=radio_key,
        deep_key=deep_key,
        label_key=label_key,
    )
    trainer = Trainer(pipeline, optimizer, loss_fn, device, cfg, forward_fn=forward_fn)

    trainer_ckpt = model_dir / f"_fusion_trainer_fold{fold}.pth"
    history_path = results_dir / f"training_history_fusion_fold{fold}.json"
    summary = trainer.fit(train_loader, val_loader, trainer_ckpt, history_path)
    logger.info("Saved fusion fold %d history -> %s", fold, history_path)

    fold_path = model_dir / f"fused_cross_attention_fold{fold}.pth"
    _save_fold_checkpoint(trainer_ckpt, radio_dim, deep_dim, cfg, fold_path)
    trainer_ckpt.unlink(missing_ok=True)

    bundle = torch.load(fold_path, map_location=device, weights_only=True)
    fusion.load_state_dict(bundle["fusion_state"])
    classifier.load_state_dict(bundle["classifier_state"])
    val_scores = _predict_val_scores(
        pipeline,
        val_loader,
        device,
        num_classes,
        pos_idx,
        radio_key,
        deep_key,
    )

    best_val_loss = float(summary["best_val_loss"])
    logger.debug(
        "Exiting _train_one_fold(fold=%d) | best_val_loss=%.4f", fold, best_val_loss
    )
    return val_scores, best_val_loss


def _promote_best_fold(
    model_dir: Path,
    best_fold: int,
    best_loss: float,
    best_auc: float,
) -> None:
    """Copy the best-fold checkpoint to the canonical name."""
    logger.debug("Entering _promote_best_fold(best=%d)", best_fold)
    if best_fold <= 0:
        logger.warning("No best fold found; skipping promotion.")
        return
    src = model_dir / f"fused_cross_attention_fold{best_fold}.pth"
    dst = model_dir / "fused_cross_attention.pth"
    shutil.copyfile(src, dst)
    auc_display = f"{best_auc:.4f}" if np.isfinite(best_auc) else "nan"
    logger.info(
        "Promoted fusion fold %d (val_auc=%s, val_loss=%.4f) -> %s",
        best_fold,
        auc_display,
        best_loss,
        dst,
    )
    logger.debug("Exiting _promote_best_fold")


def _save_oof_csv(
    pids: list[str],
    y: np.ndarray,
    oof: np.ndarray,
    fold_aucs: list[float],
    output_path: Path,
    patient_id_column: str,
    label_column: str,
    score_column: str,
) -> None:
    """Write the OOF score CSV consumed by ``scripts/run_evaluation.py``."""
    logger.debug("Entering _save_oof_csv(n=%d)", len(pids))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {patient_id_column: pids, label_column: y, score_column: oof}
    ).to_csv(
        output_path, index=False
    )
    mean_auc = float(np.nanmean(fold_aucs)) if fold_aucs else float("nan")
    logger.info(
        "Saved Fusion OOF scores -> %s | fold AUCs=%s | mean=%.3f",
        output_path.resolve(),
        fold_aucs,
        mean_auc,
    )
    logger.debug("Exiting _save_oof_csv")


def run_fusion_cv(cfg: dict[str, Any]) -> None:
    """Run patient-level k-fold CV for the cross-attention fusion track.

    Args:
        cfg: Full pipeline config dict (see ``configs/default.yaml``).

    Side effects:
        * ``models/saved/fused_cross_attention_fold{i}.pth`` per fold.
        * ``models/saved/fused_cross_attention.pth`` (best fold copy).
        * ``data/fusion_oof_scores.csv`` (path from
          ``cfg["paths"]["features"]["fusion_oof_csv"]``).
        * ``results/training_history_fusion_fold{i}.json`` per fold.
        * ``logs/fusion_training_failures.jsonl`` for any failed fold.
        * Creates output directories under ``cfg["paths"]`` when missing.
        * Optional CUDA cache reclaim between folds when
          ``model_cv.cuda_cleanup_between_folds`` is true (same flag as SwinViT CV).
    """
    logger.debug("Entering run_fusion_cv")
    set_seed(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 5: cross-attention fusion CV on %s", device)

    x_radio, x_deep, y, pids = _load_and_align_features(cfg)
    ft = cfg["fusion_training"]
    radio_key = str(ft["batch_radio_key"])
    deep_key = str(ft["batch_deep_key"])
    label_key = str(ft["batch_label_key"])
    pid_col = str(ft["csv_patient_id_column"])
    label_col = str(ft["csv_label_column"])
    score_col = str(ft["csv_score_column"])
    feature_dataset = _FeatureDictDataset(
        x_radio, x_deep, y, radio_key, deep_key, label_key
    )

    skf = make_kfold(cfg, y)
    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []
    model_dir = Path(cfg["paths"]["model_save_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    fusion_oof_path = Path(cfg["paths"]["features"]["fusion_oof_csv"])
    fusion_oof_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg["paths"]["logs_dir"]
    cuda_cleanup_between_folds = bool(cfg["model_cv"]["cuda_cleanup_between_folds"])
    best_fold_idx = -1
    best_fold_loss = float("inf")
    best_auc = -float("inf")

    splits = list(skf.split(x_radio, y))
    for fold, (train_idx, val_idx) in enumerate(
        tqdm(splits, desc="Fusion folds", unit="fold"), start=1
    ):
        log_section(logger, f"Fusion fold {fold}/{skf.n_splits}", char="-")
        try:
            scores, val_loss = _train_one_fold(
                cfg, fold, feature_dataset, train_idx, val_idx,
                y[train_idx], x_radio.shape[1], x_deep.shape[1],
                model_dir, results_dir, device,
            )
        except Exception as exc:  # noqa: BLE001 -- per-fold isolation is intentional
            logger.error("Fusion fold %d failed: %s", fold, exc)
            record_failure(_PHASE, f"fold_{fold}", exc, logs_dir=logs_dir)
            _maybe_cuda_reclaim(cuda_cleanup_between_folds)
            continue

        oof[val_idx] = scores
        if len(np.unique(y[val_idx])) > 1:
            try:
                val_auc = float(roc_auc_score(y[val_idx], scores))
            except (ValueError, RuntimeError) as exc:
                logger.warning("AUC calculation failed for fold %d: %s", fold, exc)
                val_auc = float("nan")
        else:
            logger.warning(
                "Fold %d validation has only one class; val_auc set to nan.",
                fold,
            )
            val_auc = float("nan")
        fold_aucs.append(val_auc)

        improved = False
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_fold_loss = val_loss
            improved = True
        elif np.isnan(val_auc) and val_loss < best_fold_loss:
            best_fold_loss = val_loss
            improved = True

        if improved:
            best_fold_idx = fold
            auc_display = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
            logger.info(
                "Fusion fold %d is now best (val_auc=%s, val_loss=%.4f)",
                fold,
                auc_display,
                val_loss,
            )

        _maybe_cuda_reclaim(cuda_cleanup_between_folds)

    _promote_best_fold(model_dir, best_fold_idx, best_fold_loss, best_auc)
    _save_oof_csv(
        pids,
        y,
        oof,
        fold_aucs,
        fusion_oof_path,
        pid_col,
        label_col,
        score_col,
    )
    logger.debug("Exiting run_fusion_cv")
