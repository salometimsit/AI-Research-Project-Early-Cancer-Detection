# Result Saving Rules

Output and artifact format rules. Always apply.

- Every function that produces an artifact must save it under the appropriate
  directory and log the full output path after saving.
- File-format conventions:
  - `.json` for metrics and metadata (human-readable, git-friendly).
  - `.csv` for tabular feature data (pandas-compatible).
  - `.nii.gz` for all 3D volumes and masks.
  - `.pkl` for scikit-learn models (via `joblib`).
  - `.pth` for PyTorch model weights.
- Save `crop_metadata.json` alongside every cropped volume; it must contain
  bounding-box coordinates and Z-score statistics for reverse-mapping
  attention heatmaps to the original CT.
- Per-epoch training history must always be saved to `results/` as JSON.
- Phase outputs must land in their canonical directories:
  - Per-patient artifacts → `data/processed/<PID>/`
  - Models → `models/saved/`
  - Metrics → `results/`
  - SwinViT self-attention heatmaps → `attention/<PID>_attention.nii.gz`
  - TCAV CAVs and per-concept artefacts → `tcav/`
  - Plots → `Plots/`
  - Logs → `logs/`
