"""End-to-end inference for a new patient (design-doc 'New Patient' use case).

Steps: phase filter -> NIfTI -> HU/Z-score -> liver mask -> crop ->
PyRadiomics -> VaRFS-stable subset -> SwinViT deep vector -> cross-attention
fusion -> probability + (optional) attention heatmap.

The script never writes into ``data/raw/``: DICOMs are staged inside a
:class:`tempfile.TemporaryDirectory` (preferring symbolic links to the
user-supplied folder, with a copy fallback when symlinks are disallowed).
Only the canonical ``processed/`` artefacts (NIfTI, mask, crop) survive.

Usage
-----
    python scripts/run_inference.py --config configs/default.yaml \\
        --data-config configs/data.yaml --features-config configs/features.yaml \\
        --dicom_dir /path/to/new_patient/before/ --save_heatmap \\
        --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cropping import crop_patient  # noqa: E402
from src.data.dicom_loader import DICOMLoader  # noqa: E402
from src.data.liver_segmentation import LiverSegmentor  # noqa: E402
from src.data.preprocessing import preprocess_volume  # noqa: E402
from src.features.radiomics_extractor import RadiomicsExtractor  # noqa: E402
from src.models.classifier import HCCClassifier  # noqa: E402
from src.models.fusion import CrossAttentionFusion  # noqa: E402
from src.models.swin_vit import SwinViT3D  # noqa: E402
from src.utils.config import (  # noqa: E402
    ensure_dirs,
    load_config,
    load_data_config,
    load_features_config,
    set_seed,
)
from src.utils.logger import (  # noqa: E402
    get_logger,
    log_dict,
    log_section,
    setup_logger,
)
from src.utils.run_metadata import record_run  # noqa: E402
from src.utils.visualization import plot_attention_heatmap  # noqa: E402


def _stage_dicoms(src_dir: Path, staging_root: Path, pid: str) -> Path:
    """Stage DICOMs under ``staging_root/<pid>/before/`` for :class:`DICOMLoader`.

    Symbolic links are created first (fast, little disk use). If the OS
    rejects symlinks (e.g. some Windows setups without Developer Mode),
    falls back to ``shutil.copytree`` / ``shutil.copy2`` for that entry.

    Args:
        src_dir: User-supplied DICOM directory (read-only).
        staging_root: Temporary base directory (e.g. from ``tempfile``).
        pid: Patient identifier.

    Returns:
        Path of the populated ``staging_root`` (suitable for use as
        ``DICOMLoader``'s ``raw_dir``).
    """
    target = staging_root / pid / "before"
    target.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        dest = target / path.name
        if dest.exists():
            continue
        link_target = path.resolve()
        try:
            if path.is_dir():
                dest.symlink_to(link_target, target_is_directory=True)
            else:
                dest.symlink_to(link_target, target_is_directory=False)
        except OSError:
            if path.is_dir():
                shutil.copytree(path, dest)
            else:
                shutil.copy2(path, dest)
    return staging_root


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


def main() -> None:
    parser = argparse.ArgumentParser(description="HCC inference for a new patient")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--features-config", type=str, default="configs/features.yaml")
    parser.add_argument("--dicom_dir", type=str, required=True, help="Path to raw DICOM directory")
    parser.add_argument("--patient_id", type=str, default=None)
    parser.add_argument("--save_heatmap", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_data_config(args.data_config)
    features_cfg = load_features_config(args.features_config)
    ensure_dirs(cfg)
    set_seed(cfg)

    logs_dir = Path(cfg["paths"]["logs_dir"])
    setup_logger(
        "hcc",
        log_file=logs_dir / "inference.log",
        console_level=args.log_level,
    )
    logger = get_logger("hcc.inference")
    record_run("inference", cfg, vars(args))
    log_section(logger, "Inference (new patient)")
    log_dict(logger, "cli_args", vars(args))

    processed_dir = Path(cfg["paths"]["processed_dir"])
    feature_paths = cfg["paths"]["features"]
    dicom_loader_cfg = data_cfg.get("dicom_loader")
    if not isinstance(dicom_loader_cfg, dict):
        raise KeyError("Missing required key data_config['dicom_loader'].")
    liver_seg_cfg = data_cfg.get("liver_segmentation")
    if not isinstance(liver_seg_cfg, dict):
        raise KeyError("Missing required key data_config['liver_segmentation'].")
    pre_data_cfg = data_cfg.get("preprocessing")
    if not isinstance(pre_data_cfg, dict):
        raise KeyError("Missing required key data_config['preprocessing'].")
    pid = args.patient_id or f"INF_{uuid.uuid4().hex[:8]}"
    logger.info("Running inference for patient_id=%s", pid)

    src_dir = Path(args.dicom_dir).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"--dicom_dir does not exist: {src_dir}")

    # Stage DICOMs into a tempdir (symlink-first; never touch data/raw/).
    with tempfile.TemporaryDirectory(prefix=f"hcc_inf_{pid}_") as tmp:
        staging_root = Path(tmp)
        logger.info("Staging DICOMs in temp dir: %s", staging_root)
        _stage_dicoms(src_dir, staging_root, pid)

        loader = DICOMLoader(
            staging_root,
            cfg["preprocessing"]["target_phase"],
            processed_dir,
            loader_cfg=dicom_loader_cfg,
        )
        volume_path = loader.convert_to_nifti(pid)
        segmentor = LiverSegmentor(
            gpu=bool(cfg["preprocessing"]["liver_segmentation"].get("gpu", True)),
            roi_subset=tuple(liver_seg_cfg.get("roi_subset", ["liver"])),
            ml=bool(liver_seg_cfg.get("ml", False)),
            fast=bool(liver_seg_cfg.get("fast", False)),
            require_non_empty_mask=bool(liver_seg_cfg.get("require_non_empty_mask", True)),
        )
        mask_path = segmentor.segment(volume_path)
        _, stats, normalized_path = preprocess_volume(volume_path, mask_path, pre_data_cfg)
        cropping_cfg = data_cfg.get("cropping", {})
        if "padding" not in cropping_cfg:
            raise KeyError("Missing required key data_config['cropping']['padding'].")
        padding = int(cropping_cfg["padding"])
        cropped_path, meta_path, cropped_mask_path = crop_patient(
            volume_path.parent,
            zscore_stats=stats,
            padding=padding,
            volume_filename=normalized_path.name,
        )

    # --- Radiomics + VaRFS-stable subset ---------------------------------
    radiomics_cfg = features_cfg.get("radiomics")
    if not isinstance(radiomics_cfg, dict):
        raise KeyError("Missing required key features_config['radiomics'].")
    extractor = RadiomicsExtractor(radiomics_cfg)
    feats = extractor.extract_patient(cropped_path, cropped_mask_path)
    feats_df = pd.DataFrame([{**feats, "patient_id": pid, "label": -1}])

    selected_path = Path(feature_paths["varfs_selected_json"])
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Run scripts/run_radiomics.py first to produce {selected_path}."
        )
    with open(selected_path, "r", encoding="utf-8") as fh:
        selected = json.load(fh)["selected_features"]

    missing = [c for c in selected if c not in feats_df.columns]
    if missing:
        raise RuntimeError(
            "Radiomics extractor produced no value for the following selected "
            f"features: {missing}. Re-run extraction or rebuild VaRFS selection."
        )
    radio_vec = feats_df[selected].fillna(0.0).to_numpy(dtype=np.float32)

    # --- SwinViT deep vector --------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin = _build_swin(cfg).to(device)
    swin_ckpt_path = Path(cfg["paths"]["model_save_dir"]) / "best_swinvit_model.pth"
    if not swin_ckpt_path.exists():
        raise FileNotFoundError(f"Missing SwinViT checkpoint: {swin_ckpt_path}")
    swin_ckpt = torch.load(swin_ckpt_path, map_location=device, weights_only=True)
    swin.load_state_dict(swin_ckpt["model_state"])
    swin.eval()

    cropped = nib.load(str(cropped_path)).get_fdata().astype(np.float32)
    target = tuple(cfg["swin_vit"]["img_size"])
    tensor = torch.from_numpy(cropped)[None, None].float()
    tensor = torch.nn.functional.interpolate(
        tensor, size=target, mode="trilinear", align_corners=False
    )
    tensor = tensor.to(device)
    with torch.no_grad():
        _, deep = swin(tensor)
        if args.save_heatmap:
            attn = swin.get_attention_maps(tensor).cpu().numpy()[0]
            volume_full = nib.load(str(volume_path))
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            plot_attention_heatmap(
                volume_full.get_fdata(),
                attn,
                meta,
                Path(cfg["paths"]["attention_dir"]) / f"{pid}_attention.nii.gz",
                affine=volume_full.affine,
            )

    # --- Fusion + classifier --------------------------------------------
    fusion_ckpt_path = Path(cfg["paths"]["model_save_dir"]) / "fused_cross_attention.pth"
    if not fusion_ckpt_path.exists():
        raise FileNotFoundError(f"Missing fusion checkpoint: {fusion_ckpt_path}")
    fusion_ckpt = torch.load(fusion_ckpt_path, map_location=device, weights_only=True)

    expected_radio_dim = int(fusion_ckpt["radio_dim"])
    if radio_vec.shape[1] != expected_radio_dim:
        # Strict: silently padding/truncating would mask a versioning bug
        # between the trained fusion model and the current VaRFS subset.
        raise RuntimeError(
            "Radiomics feature dimension mismatch: got "
            f"{radio_vec.shape[1]} features but fusion checkpoint expects "
            f"{expected_radio_dim}. Re-run scripts/run_radiomics.py + "
            "scripts/run_training.py to retrain fusion against the "
            "current VaRFS selection."
        )

    fusion = CrossAttentionFusion(
        radiomics_dim=expected_radio_dim,
        deep_dim=int(fusion_ckpt["deep_dim"]),
        num_heads=int(cfg["cross_attention"]["num_heads"]),
        hidden_dim=int(cfg["cross_attention"]["hidden_dim"]),
        dropout=float(cfg["cross_attention"]["dropout"]),
    ).to(device)
    classifier = HCCClassifier(
        fused_dim=fusion.output_dim,
        dropout=float(cfg["model"]["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)
    fusion.load_state_dict(fusion_ckpt["fusion_state"])
    classifier.load_state_dict(fusion_ckpt["classifier_state"])
    fusion.eval(); classifier.eval()

    with torch.no_grad():
        radio_t = torch.from_numpy(radio_vec).to(device)
        fused = fusion(radio_t, deep)
        logits = classifier(fused)
        probability = float(torch.softmax(logits, dim=-1)[0, 1].item())

    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    decision = int(probability >= threshold)
    result = {
        "patient_id": pid,
        "hcc_probability": probability,
        "decision_threshold": threshold,
        "predicted_label": decision,
    }
    out_path = Path(cfg["paths"]["results_dir"]) / f"inference_{pid}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Inference complete: %s", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
