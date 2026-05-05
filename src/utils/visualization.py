"""Visualization helpers for SwinViT attention, ROC, ablation, and TCAV.

Pipeline stage: K, L, M outputs in the A->M flow.

Inputs:
    - SwinViT activation tensors (for ``plot_attention_heatmap``).
    - Per-model score arrays (for ``plot_roc_curve``).
    - Per-model metric dicts (for ``plot_ablation_comparison``).
    - Training history JSON files (for ``plot_training_curves``).
    - TCAV score / p-value dicts (for ``plot_tcav_scores``).

Outputs:
    - PNG plots saved under ``Plots/``.
    - NIfTI heatmaps saved under ``attention/`` (via
      :func:`plot_attention_heatmap`).

Side effects:
    - Creates parent directories on demand.
    - Writes one log line per saved artefact.

Failure modes:
    - Missing ``crop_metadata.json`` -> ``KeyError`` from
      :func:`plot_attention_heatmap`. Re-run preprocessing first.
    - Mismatched lengths in ``y_scores_by_model`` -> shapely error from
      ``sklearn.metrics``. All score arrays must align with ``y_true``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.metrics import auc, roc_curve

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_attention_heatmap(
    volume: np.ndarray,
    attention_map: np.ndarray,
    crop_metadata: dict[str, Any],
    output_path: Path | str,
    affine: np.ndarray | None = None,
    global_max: float | None = None,
) -> Path:
    """Reverse-map a SwinViT attention map back to the original CT space.

    The attention map is trilinear-upsampled to the cropped-volume shape,
    normalized and placed back into a zero-volume of the original shape using
    the half-open bbox stored in ``crop_metadata``. The result is saved as a
    NIfTI heatmap that overlays cleanly on the source CT (same affine).

    Normalization behaviour:
        When ``global_max`` is provided the heatmap is divided by that value
        so all patients in a cohort share the same scale — a healthy liver
        will genuinely appear "cooler" than one with a clear tumour.  When
        ``global_max`` is ``None`` the map is normalized per-sample (classic
        min-max to ``[0, 1]``), which is acceptable for qualitative review
        but should not be used for cross-patient comparisons.

    Axis-order convention:
        ``volume`` and the bbox in ``crop_metadata`` MUST share the
        same axis order (the order returned by
        ``nibabel.load(...).get_fdata()``). The 3D attention map is
        assumed to already be 3D and is squeezed/resized accordingly.

    Args:
        volume: Original (un-cropped) CT volume; only its ``shape`` is
            used here for sanity-checking against ``crop_metadata``.
        attention_map: 3D attention activations from the deepest Swin
            stage (any spatial resolution).
        crop_metadata: Dict with ``bbox.{start,stop}``,
            ``original_shape``, and ``cropped_shape`` produced by
            :func:`src.data.cropping.crop_to_bbox`.
        output_path: Destination ``.nii.gz`` path.
        affine: NIfTI affine to attach to the saved heatmap; defaults
            to the 4x4 identity.
        global_max: Optional cohort-level maximum attention value used as
            the normalization denominator. Pass the maximum value observed
            across all patients to enable cross-patient comparison.
            When ``None`` per-sample min-max normalization is applied.

    Returns:
        The resolved ``output_path`` after the heatmap is saved.

    Raises:
        AssertionError: If the volume / metadata shapes are
            inconsistent or the bbox lies outside the original volume.
    """
    bbox = crop_metadata["bbox"]
    original_shape = tuple(int(v) for v in crop_metadata["original_shape"])
    start = bbox["start"]
    stop = bbox["stop"]
    cropped_shape = tuple(b - a for a, b in zip(start, stop))

    # The metadata's original_shape MUST match the volume we were handed,
    # otherwise we'd index into the wrong axis order (e.g. (D, H, W) vs
    # (W, H, D)) and produce garbage heatmaps.
    assert volume.ndim == 3, f"volume must be 3D, got {volume.ndim}D"
    assert tuple(volume.shape) == original_shape, (
        f"volume.shape={volume.shape} does not match crop_metadata "
        f"original_shape={original_shape}; axis-order mismatch likely."
    )
    assert len(start) == 3 and len(stop) == 3, "bbox must be 3D"
    for axis, (a, b, n) in enumerate(zip(start, stop, original_shape)):
        assert 0 <= a < b <= n, (
            f"bbox out of range on axis {axis}: start={a}, stop={b}, "
            f"original={n}"
        )

    attn = _resize_to(attention_map, cropped_shape)
    denom = float(global_max) if global_max is not None else (float(np.ptp(attn)) + 1e-8)
    attn = attn / denom
    attn = np.clip(attn, 0.0, 1.0).astype(np.float32)

    full = np.zeros(original_shape, dtype=np.float32)
    full[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = attn

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(full, affine=affine if affine is not None else np.eye(4))
    nib.save(nii, str(output_path))
    logger.info("Saved attention heatmap -> %s", output_path)
    return output_path


def _vget(viz_cfg: dict | None, key: str, default: Any) -> Any:
    """Retrieve a nested key from viz_cfg using dot notation with a fallback.

    Example: ``_vget(cfg, "figsize.roc", (6, 6))`` reads
    ``cfg["figsize"]["roc"]`` and falls back to ``(6, 6)`` when absent.
    """
    if viz_cfg is None:
        return default
    node: Any = viz_cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _resize_to(volume: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(volume.astype(np.float32))
    while tensor.dim() < 5:
        tensor = tensor.unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=size, mode="trilinear", align_corners=False
    )
    return resized.squeeze().numpy().astype(np.float32)


def plot_roc_curve(
    y_true: Sequence[int],
    y_scores_by_model: dict[str, Sequence[float]],
    output_path: Path | str,
    title: str = "ROC Comparison",
    viz_cfg: dict | None = None,
) -> Path:
    """Plot ROC curves for one or more models on the same axes."""
    figsize = _vget(viz_cfg, "figsize.roc", (6, 6))
    dpi = _vget(viz_cfg, "dpi", 150)
    fig, ax = plt.subplots(figsize=figsize)
    for label, scores in y_scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved ROC plot -> %s", output_path)
    return output_path


def plot_ablation_comparison(
    metrics_dict: dict[str, dict[str, float]],
    output_path: Path | str,
    metric: str = "auc_roc",
    title: str = "3-Model Ablation",
    viz_cfg: dict | None = None,
) -> Path:
    """Bar chart comparing radiomics-only, SwinViT-only, and fused models."""
    figsize = _vget(viz_cfg, "figsize.ablation", (6, 4))
    dpi = _vget(viz_cfg, "dpi", 150)
    models = list(metrics_dict.keys())
    values = [float(metrics_dict[m].get(metric, 0.0)) for m in models]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(models, values, color=["#4C72B0", "#DD8452", "#55A868"][: len(models)])
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved ablation chart -> %s", output_path)
    return output_path


def plot_training_curves(
    history_files: dict[str, Path | str],
    output_path: Path | str,
    viz_cfg: dict | None = None,
) -> Path:
    """Plot per-epoch training/validation loss for each phase."""
    figsize = _vget(viz_cfg, "figsize.training_curves", (8, 5))
    dpi = _vget(viz_cfg, "dpi", 150)
    fig, ax = plt.subplots(figsize=figsize)
    for label, path in history_files.items():
        path = Path(path)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            history = json.load(fh)
        epochs = [entry["epoch"] for entry in history]
        train_loss = [entry.get("train_loss", float("nan")) for entry in history]
        val_loss = [entry.get("val_loss", float("nan")) for entry in history]
        ax.plot(epochs, train_loss, label=f"{label} train")
        ax.plot(epochs, val_loss, linestyle="--", label=f"{label} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved training curves -> %s", output_path)
    return output_path


def plot_tcav_scores(
    scores: dict[str, float],
    p_values: dict[str, float],
    output_path: Path | str,
    title: str = "TCAV scores",
    significance_threshold: float = 0.05,
    viz_cfg: dict | None = None,
) -> Path:
    """Horizontal bar chart of TCAV scores with significance markers.

    Concepts with ``p_values[concept] < significance_threshold`` are
    annotated with a trailing ``"*"``. Bars are sorted descending by
    score so the strongest concepts appear at the top.
    """
    if not scores:
        logger.warning("plot_tcav_scores called with empty scores; skipping.")
        return Path(output_path)

    bar_w = _vget(viz_cfg, "figsize.tcav_bar_width", 8)
    bar_min_h = _vget(viz_cfg, "figsize.tcav_bar_min_height", 3)
    bar_hpl = _vget(viz_cfg, "figsize.tcav_bar_height_per_label", 0.35)
    dpi = _vget(viz_cfg, "dpi", 150)

    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    labels = []
    values = []
    for concept, score in items:
        p = p_values.get(concept, float("nan"))
        marker = "*" if not np.isnan(p) and p < significance_threshold else ""
        labels.append(f"{concept}{marker}")
        values.append(score)

    fig, ax = plt.subplots(figsize=(bar_w, max(bar_min_h, bar_hpl * len(labels))))
    y_positions = np.arange(len(labels))
    ax.barh(y_positions, values, color="#4C72B0")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("TCAV score (fraction positive directional derivatives)")
    ax.set_xlim(0, 1)
    ax.set_title(f"{title} ({significance_threshold * 100:.0f}% sig. marked '*')")
    for i, v in enumerate(values):
        ax.text(min(v + 0.01, 0.98), i, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved TCAV bar chart -> %s", output_path)
    return output_path


def plot_concept_importance_heatmap(
    per_feature_scores: dict[str, float],
    per_family_scores: dict[str, float],
    output_path: Path | str,
    title: str = "TCAV concept importance",
    viz_cfg: dict | None = None,
) -> Path:
    """Combined heatmap of per-feature and per-family TCAV scores."""
    hm_w = _vget(viz_cfg, "figsize.concept_heatmap_width", 4)
    hm_min_h = _vget(viz_cfg, "figsize.concept_heatmap_min_height", 3)
    hm_hpl = _vget(viz_cfg, "figsize.concept_heatmap_height_per_label", 0.30)
    dpi = _vget(viz_cfg, "dpi", 150)

    rows: list[tuple[str, str, float]] = []
    for k, v in per_family_scores.items():
        rows.append(("family", k, v))
    for k, v in per_feature_scores.items():
        rows.append(("feature", k, v))
    if not rows:
        logger.warning(
            "plot_concept_importance_heatmap called with empty inputs; skipping."
        )
        return Path(output_path)

    rows.sort(key=lambda r: (r[0], -r[2]))
    labels = [f"[{kind}] {name}" for kind, name, _ in rows]
    values = np.asarray([[r[2]] for r in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(hm_w, max(hm_min_h, hm_hpl * len(labels))))
    im = ax.imshow(values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([])
    ax.set_title(title)
    for i, v in enumerate(values.ravel()):
        ax.text(0, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved TCAV heatmap -> %s", output_path)
    return output_path


def plot_slice_overlay(
    volume: np.ndarray,
    mask: np.ndarray,
    slice_idx: int,
    output_path: Path | str | None = None,
    window_center: int = 40,
    window_width: int = 400,
    viz_cfg: dict | None = None,
) -> None:
    """Display (and optionally save) a single 2D slice with mask overlay.

    Uses a clinical liver HU window by default so the 20–40 HU contrast
    difference between healthy parenchyma and an HCC lesion is visible.
    Without windowing both tissues map to the same shade of grey and the
    overlay is diagnostically useless.

    Args:
        volume: 3D CT volume ``(H, W, D)`` or a single 2D slice ``(H, W)``.
        mask: Binary liver/tumour mask, same shape as ``volume``.
        slice_idx: Index along the last axis to display (ignored for 2D input).
        output_path: When provided the figure is saved here; otherwise it is
            displayed interactively.
        window_center: HU window centre (default 40 = liver soft tissue).
        window_width: HU window width (default 400 = standard liver window).
    """
    figsize = _vget(viz_cfg, "figsize.slice_overlay", (5, 5))
    dpi = _vget(viz_cfg, "dpi", 150)
    slc = volume[..., slice_idx] if volume.ndim == 3 else volume
    msk = mask[..., slice_idx] if mask.ndim == 3 else mask
    vmin = window_center - window_width // 2
    vmax = window_center + window_width // 2
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(slc.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_where(msk.T == 0, msk.T), alpha=0.4, cmap="autumn", origin="lower")
    ax.set_title(f"Slice {slice_idx} (Liver Window: C={window_center} W={window_width})")
    ax.axis("off")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
