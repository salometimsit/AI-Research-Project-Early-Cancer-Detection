# HCC Early Prediction from CT Scans

Predict future Hepatocellular Carcinoma (HCC) development from pre-diagnosis CT
scans of cirrhotic livers. The pipeline combines **global radiomics** with a
**3D Vision Transformer (SwinViT)** and fuses them through a **Cross-Attention
Joint Embedding** layer.

## Overview

Dual-track architecture that produces a single risk probability per patient:

1. **Radiomics track** — PyRadiomics features over the whole liver volume,
   filtered through VaRFS bootstrap stability selection to keep only stable
   biological markers.
2. **Deep learning track** — 3D SwinViT consumes the cropped liver bounding
   box and uses native self-attention to focus on suspicious pre-cancerous
   regions (avoiding manual patch sampling).
3. **Fusion** — A cross-attention layer learns the correlation between
   radiomics texture and deep visual features, then a classification head
   outputs the HCC risk probability.

## Pipeline (steps A–M)

```
A1: CT Scans (Cirrhotic) ─┐
A2: Labels (0/1)          ├─► B: Phase Filter + DICOM→NIfTI
                          │
                          ▼
            C: HU Windowing + Z-Score (within liver mask)
                          │
                          ▼
                  D: Liver Segmentation
                          │
                          ▼
              E: Liver Bounding Box Cropping
                  ┌───────┴────────┐
                  ▼                ▼
        F: PyRadiomics      G: 3D SwinViT Input
                  │                ▼
                  ▼          J: Self-Attention
        H: VaRFS Stability         │
                  │                ▼
                  ▼          K: Deep Visual Vector
        I: Stable Features         │
                  └───────┬────────┘
                          ▼
              L: Cross-Attention Fusion
                          │
                          ▼
              M: HCC Risk Probability (0–1)
```

## Project structure

```
Project_root/
├── .gitignore
├── .cursor/rules/                  # Cursor rules for AI guidance
├── requirements.txt
├── README.md
├── configs/
│   ├── default.yaml                # Required: all scientific/clinical hyperparameters
│   └── visualization.yaml          # Optional: DPI and figure sizes for plots
├── data/
│   ├── raw/                        # Original DICOM (data/raw/<PID>/before/)
│   ├── processed/                  # NIfTI volumes, masks, crop_metadata
│   │   └── <PID>/
│   │       ├── before.nii.gz
│   │       ├── before_liver.nii.gz
│   │       ├── before_cropped.nii.gz
│   │       └── crop_metadata.json
│   ├── raw_radiomics_features.csv
│   ├── varfs_filtered_features.csv
│   ├── varfs_selected_features.json
│   ├── deep_features.csv
│   └── labels.csv
├── notebooks/
├── scripts/
│   ├── download_data.py
│   ├── run_preprocessing.py
│   ├── run_radiomics.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_inference.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── results/
├── attention/                     # SwinViT self-attention heatmaps (NIfTI)
├── tcav/                          # TCAV CAV vectors + per-concept artefacts
├── Plots/
├── logs/
├── models/
│   ├── pretrained/                  # optional: MONAI / SSL Swin .pth checkpoints
│   └── saved/
```

Note: the tree above shows `models/pretrained/` as the conventional location for
`swin_vit.pretrained_weights` files; create it locally after downloading weights.

## Raw DICOM layout

`data/raw/<PID>/before/` is allowed to contain a **flat dump of all DICOM
files** exported from the hospital disk — typically several hundred to a
few thousand individual `.dcm` slices, often interleaved with multiple
contrast phases (arterial, portal-venous, delayed, etc.) and unrelated
clutter such as `.dll` viewer artefacts, `AUTORUN.INF`, or vendor PDFs.

You do **not** need to pre-sort or group these files manually.
[`src/data/dicom_loader.py`](src/data/dicom_loader.py) uses SimpleITK +
GDCM (`ImageSeriesReader.GetGDCMSeriesIDs`) to:

1. Group every `.dcm` file by its `SeriesInstanceUID` automatically.
2. Inspect each series' `SeriesDescription` / `ProtocolName` and keep
   only the one matching `preprocessing.target_phase` from
   [`configs/default.yaml`](configs/default.yaml) (e.g. `portal_venous`).
3. Write a single ordered NIfTI to
   `data/processed/<PID>/before.nii.gz`.

Non-DICOM files (Windows viewer `.dll`s, READMEs, autorun scripts) are
silently skipped — they have no DICOM header for GDCM to parse. Every
ACCEPTED / REJECTED series decision is logged so you can audit which
phase was picked per patient.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TotalSegmentator downloads its weights on first run (~1.5 GB).

## Data download (Google Drive)

Per-patient DICOM zips are downloaded from Google Drive via `gdown` (no auth
required for shared links). Configure `drive.patient_files` in
`configs/default.yaml` mapping each `patient_id` to its Drive file ID:

```yaml
drive:
  patient_files:
    P001: "1aBcDeFgHiJkLmNoPqRsTuVwXyZ"
    P002: "1xYzAbCdEfGhIjKlMnOpQrStUv"
```

```bash
python scripts/download_data.py --config configs/default.yaml
python scripts/download_data.py --config configs/default.yaml --patient P001
python scripts/download_data.py --config configs/default.yaml --list
```

## Usage

```bash
# Phase 2: DICOM → NIfTI → HU/Z-score → segment → crop
python scripts/run_preprocessing.py --config configs/default.yaml

# Phase 3: PyRadiomics → VaRFS → classical baseline
python scripts/run_radiomics.py --config configs/default.yaml

# Phase 4 + 5: SwinViT, then cross-attention fusion
python scripts/run_training.py --config configs/default.yaml --phase swinvit
python scripts/run_training.py --config configs/default.yaml --phase fusion

# Phase 6: cross-validation, SwinViT attention maps, ablation plots
python scripts/run_evaluation.py --config configs/default.yaml

# Concept-based interpretability (TCAV per-feature + per-family)
python scripts/run_tcav.py --config configs/default.yaml --mode both

# Inference on a new patient
python scripts/run_inference.py --config configs/default.yaml \
    --dicom_dir /path/to/new_patient/before/
```

## Configuration

The project uses two YAML configuration files:

### `configs/default.yaml` (required)

All scientific, clinical, and training hyperparameters. The pipeline **fails immediately** when this file is missing or lacks a mandatory section (`paths`, `seed`, `preprocessing`, `swin_vit`). Never hardcode values — always read via `load_config()`.

Key highlights:

| Section | Notable parameters |
| ------- | ------------------ |
| `preprocessing` | `phase_keywords`, `hu_window`, `normalization_scope`, `voxel_spacing` |
| `swin_vit` | `img_size`, `window_size` (must evenly divide `img_size`), `embed_dim`, `depths` |
| `cross_attention` | `num_heads`, `hidden_dim` |
| `radiomics` / `varfs` | feature classes, bin_width, stability threshold |
| `training` | optimizer, focal loss, `deterministic`, `cudnn_benchmark` |
| `tcav` | `cav_C` (LogisticRegression C), `significance_threshold`, `extract_batch_size` |
| `visualization` | `window_center`, `window_width` — clinical HU window for CT display |
| `seed` | Global random seed (passed to `set_seed(cfg)`) |

### `configs/visualization.yaml` (optional)

Presentation-only parameters (DPI, per-chart figure sizes). The pipeline runs without it; all `plot_*` functions use built-in defaults when it is absent. Load with:

```python
from src.utils.config import load_viz_config
viz_cfg = load_viz_config()   # returns {} when file is absent
```

Pass `viz_cfg=viz_cfg` to any `plot_*` call to override DPI and figure sizes.

## Pretrained SwinViT (transfer learning)

Training a 3D Swin Transformer **from scratch** on a small patient cohort is difficult and
often unstable. The pipeline supports **warm-starting only the MONAI Swin backbone**
(`SwinViT3D.encoder`) from an external `.pth` checkpoint while keeping the classification
`head` randomly initialized for your binary HCC task.

| Config key | Meaning |
| ---------- | ------- |
| `swin_vit.pretrained_weights` | `null` (default) → random encoder weights. Set to a path such as `models/pretrained/ssl_pretrained_weights.pth` (relative paths resolve from the **current working directory**, usually the repo root when using `scripts/run_training.py`). |
| `swin_vit.pretrained_strict` | `false` (recommended): allow partial load when UNETR checkpoints omit layers or shapes differ slightly. |

If `pretrained_weights` is set but the file is **missing**, training logs a **WARNING** and
continues **from scratch** (fold CV is not aborted).

Place downloaded checkpoints under [`paths.pretrained_dir`](configs/default.yaml)
(`models/pretrained/` by convention). You **must align** `embed_dim`, `depths`,
`num_heads`, `window_size`, and `patch_size` with the recipe used to produce the checkpoint;
otherwise many tensors are skipped (shape mismatch) and you effectively train mostly random
layers — see logs from [`src/models/pretrained_swin.py`](src/models/pretrained_swin.py).

Run Phase 4 after pointing the config at a weight file:

```bash
python scripts/run_training.py --config configs/default.yaml --phase swinvit
```

### Where to get weights (examples)

| Source | Notes | Link |
| ------ | ----- | ---- |
| MONAI discussion | SSL / backbone `.pth` links posted by maintainers | [MONAI Discussion #7208](https://github.com/Project-MONAI/MONAI/discussions/7208) |
| MONAI research-contributions | SwinUNETR pretraining scripts and weights | [SwinUNETR/Pretrain](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain) |
| MONAI tutorial | BTCV 3D segmentation with Swin-UNETR | [swin_unetr_btcv_segmentation_3d.ipynb](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb) |
| NVIDIA NGC | Search for MONAI Swin-UNETR bundles | [catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/) |

Papers to cite: Tang et al., CVPR 2022 (self-supervised 3D Swin); Hatamizadeh et al., Swin UNETR.

## Data model

| Stage          | Artifact                                      |
| -------------- | --------------------------------------------- |
| DICOM ingest   | `data/raw/<PID>/before/`                      |
| NIfTI volume   | `data/processed/<PID>/before.nii.gz`          |
| Liver mask     | `data/processed/<PID>/before_liver.nii.gz`    |
| Cropped liver  | `data/processed/<PID>/before_cropped.nii.gz`  |
| Crop metadata  | `data/processed/<PID>/crop_metadata.json`     |
| Radiomics      | `data/raw_radiomics_features.csv`             |
| VaRFS-stable   | `data/varfs_filtered_features.csv`            |
| Deep features  | `data/deep_features.csv`                      |
| Labels         | `data/labels.csv` (`patient_id,label`)        |

`label`: `0` = stable cirrhosis (low risk), `1` = future HCC (high risk).
