# Project rules for Claude Code

These rules mirror `.cursor/rules/*.mdc` and must always be part of Claude's
thinking and view on this repository. They were copied from Cursor into
`.claude/rules/` so any tool reading CLAUDE.md picks them up automatically.

## Always-on rules (imported)

@.claude/rules/project-conventions.md
@.claude/rules/pipeline-structure.md
@.claude/rules/medical-imaging.md
@.claude/rules/result-saving.md

## Architecture quick reference

- Dual-track architecture: radiomics + 3D SwinViT, fused via cross-attention.
- Entry points live in `scripts/`, reusable logic in `src/`.
- Configuration is layered YAML, all loaded via `src.utils.config`:
  - `configs/default.yaml` — `paths`, `preprocessing`, `training`,
    `evaluation`, `tcav`, `visualization`, `seed`, `drive`.
  - `configs/models.yaml` — `swin_vit`, `cross_attention`, `fusion_training`,
    `model`, `model_cv`. Deep-merged into the base config by
    `load_config` (models has precedence on conflicts).
  - `configs/data.yaml` — cropping, dataset, augmentation, DICOM loader,
    liver segmentation, preprocessing details (loaded via `load_data_config`).
  - `configs/features.yaml` — radiomics, VaRFS, baseline classifier
    (loaded via `load_features_config`).
  - `configs/visualization.yaml` — optional, loaded via `load_viz_config`.
- `evaluation.*` (in `default.yaml`) controls test-time behavior:
  - `cross_validation.strategy`: `kfold` (StratifiedKFold using
    `model_cv.n_folds`) or `leave_one_patient_out` (LeaveOneOut; per-fold AUC
    becomes NaN — best-fold selection in `cv_training.run_swinvit_cv` and
    `fusion_training.run_fusion_cv` falls back to `val_loss`).
  - `metrics`: whitelist applied by `_filter_metrics` in `run_evaluation.py`
    to the per-model dict written to `results/evaluation_metrics.json`
    (`threshold` is metadata and always kept).
  - `export_attention_heatmaps`: default for `run_evaluation.py`; the
    `--skip-heatmaps` CLI flag overrides to skip.
  - `threshold`: decision threshold used by `_safe_metrics` and by
    `run_inference.py` to derive the binary label.
- Never reintroduce removed phases (registration, manual patch extraction).
