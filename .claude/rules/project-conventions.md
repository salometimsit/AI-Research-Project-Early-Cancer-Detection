# Project Conventions

Project-wide code conventions for the HCC pipeline. Always apply.

- Add type hints to every function signature.
- Use Google-style docstrings on all public functions and classes.
- Always use `pathlib.Path` for filesystem paths; never use string concatenation.
- Never hardcode hyperparameters. Read them via `src.utils.config`:
  - `load_config()` — `configs/default.yaml` deep-merged with `configs/models.yaml`
    (paths, training, evaluation, swin_vit, model, model_cv, cross_attention,
    fusion_training, tcav, preprocessing, seed). `models.yaml` overrides `default.yaml`
    on conflicts.
  - `load_data_config()` — `configs/data.yaml` (DICOM loader, liver segmentation,
    cropping, dataset, augmentation, runtime preprocessing details).
  - `load_features_config()` — `configs/features.yaml` (radiomics, VaRFS,
    baseline classifier).
  - `load_viz_config()` — optional `configs/visualization.yaml`.
- Always call `set_seed(cfg)` before any random operation; the helper reads
  `cfg["seed"]`, `cfg["training"]["deterministic"]`, and
  `cfg["training"]["cudnn_benchmark"]`.
- Use `tqdm` for any loop that iterates over patients or training epochs.
- Every module must have a corresponding `__init__.py` that exports its public
  classes/functions.
- Log entry/exit of important functions at `DEBUG` level using
  `src.utils.logger.get_logger(__name__)`.

## Module documentation template

Every new module under `src/` must start with a docstring that includes:

- One-line summary.
- "Pipeline stage: <letter> in A->M".
- "Inputs", "Outputs", "Side effects" sections.
- "Failure modes" listing typical errors and what they mean.

## Logging and run metadata

- Each `scripts/*.py` entry point must:
  - Accept `--log-level {DEBUG,INFO,WARNING,ERROR}` and forward to `setup_logger`.
  - Call `record_run(<phase>, cfg, vars(args))` at the start of `main()`.
  - Use `log_section` to mark major phase boundaries and `log_dict` to dump
    the relevant config sub-dicts at the start.
  - Wrap every per-patient loop in a try/except that calls
    `record_failure(<phase>, patient_id, exc)` so failed runs leave a
    structured JSONL trail under `logs/<phase>_failures.jsonl`.
