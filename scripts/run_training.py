"""Phase 4 + 5 entry point: train SwinViT, then cross-attention fusion.

Both tracks run patient-level k-fold cross-validation as configured in
``evaluation.cross_validation``. The heavy lifting lives in
:mod:`src.models.cv_training`; this script is a thin orchestrator.

Usage
-----
    python scripts/run_training.py --config configs/default.yaml --phase swinvit
    python scripts/run_training.py --config configs/default.yaml --phase fusion
    python scripts/run_training.py --config configs/default.yaml --phase all \\
        --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cv_training import run_swinvit_cv  # noqa: E402
from src.models.fusion_training import run_fusion_cv  # noqa: E402
from src.utils.config import ensure_dirs, load_config, load_data_config, set_seed  # noqa: E402
from src.utils.logger import (  # noqa: E402
    get_logger,
    log_dict,
    log_section,
    setup_logger,
)
from src.utils.run_metadata import record_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training (SwinViT and/or fusion)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument(
        "--phase",
        choices=["all", "swinvit", "fusion"],
        default="all",
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
        log_file=logs_dir / "training.log",
        console_level=args.log_level,
    )
    logger = get_logger("hcc.training")

    for key in ("augmentation", "dataset", "preprocessing"):
        block = data_cfg.get(key)
        if isinstance(block, dict):
            cfg[key] = block
        else:
            logger.warning(
                "Section %r missing or invalid in %s; training may use defaults.",
                key,
                args.data_config,
            )

    record_run("training", cfg, vars(args))
    log_section(logger, f"Training | phase={args.phase}")
    log_dict(logger, "config.training", cfg["training"])
    log_dict(logger, "config.swin_vit", cfg["swin_vit"])
    log_dict(
        logger,
        "config.evaluation.cv",
        cfg.get("evaluation", {}).get("cross_validation", {}),
    )

    if args.phase in ("all", "swinvit"):
        run_swinvit_cv(cfg)
    if args.phase in ("all", "fusion"):
        ckpt_path = Path(cfg["paths"]["model_save_dir"]) / "best_swinvit_model.pth"
        if not ckpt_path.is_file():
            logger.error(
                "Cannot run fusion: SwinViT checkpoint missing at %s. "
                "Run with --phase swinvit or --phase all first.",
                ckpt_path,
            )
            sys.exit(1)
        run_fusion_cv(cfg)


if __name__ == "__main__":
    main()
