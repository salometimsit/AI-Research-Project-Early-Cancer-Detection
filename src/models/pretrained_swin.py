"""Load pretrained weights into the MONAI Swin backbone of :class:`~src.models.swin_vit.SwinViT3D`.

Pipeline stage: J (Phase 4 warm-start); invoked from :mod:`src.models.cv_training`
before :class:`~src.models.trainer.Trainer.fit`.

Inputs
------
* A :class:`~src.models.swin_vit.SwinViT3D` instance (encoder + head already constructed).
* Path to a ``.pth`` / ``.pt`` checkpoint (often a Swin-UNETR or SSL backbone).

Outputs
-------
* In-place update of ``model.encoder`` weights for keys that match in name and shape.
* Return dict with counts and lists for logging (missing / shape-skipped / loaded).

Side effects
------------
* Mutates ``model.encoder`` parameters; does **not** load into ``model.head`` or ``model.norm``
  unless keys accidentally match encoder submodule names (they should not).

Failure modes
-------------
* Corrupt or non-dict checkpoint -> ``RuntimeError`` / ``TypeError`` from PyTorch.
* Tensor shape mismatch for a given key -> key skipped with WARNING (partial load).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.swin_vit import SwinViT3D
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _unwrap_checkpoint(
    blob: Any,
    unwrap_keys: tuple[str, ...],
    unwrap_max_depth: int,
    depth: int = 0,
) -> dict[str, Any]:
    """Drill into common checkpoint wrappers until a flat-ish dict is reached."""
    if depth > unwrap_max_depth:
        raise ValueError("Checkpoint nesting too deep; unknown format.")
    if not isinstance(blob, dict):
        raise TypeError(f"Expected mapping at checkpoint root, got {type(blob).__name__}.")
    for key in unwrap_keys:
        inner = blob.get(key)
        if isinstance(inner, dict) and inner:
            return _unwrap_checkpoint(
                inner,
                unwrap_keys=unwrap_keys,
                unwrap_max_depth=unwrap_max_depth,
                depth=depth + 1,
            )
    return blob


def _normalize_encoder_key(key: str) -> str:
    """Strip DDP / SwinUNETR prefixes so keys match ``model.encoder`` state dict."""
    k = key
    while True:
        if k.startswith("module."):
            k = k[len("module.") :]
        elif k.startswith("swinViT."):
            k = k[len("swinViT.") :]
        elif k.startswith("swin_vit."):
            k = k[len("swin_vit.") :]
        elif k.startswith("encoder."):
            k = k[len("encoder.") :]
        elif k.startswith("backbone."):
            k = k[len("backbone.") :]
        elif k.startswith("model."):
            k = k[len("model.") :]
        else:
            break
    return k


def _resolve_map_location(
    model: SwinViT3D,
    map_location: str | torch.device | None,
    pretrained_load_target: str,
) -> str | torch.device:
    """Resolve checkpoint load device from explicit arg or config mode."""
    if map_location is not None:
        return map_location
    if pretrained_load_target == "cpu":
        return "cpu"
    if pretrained_load_target == "model_device":
        return next(model.parameters()).device
    raise ValueError(
        "Unsupported pretrained_load_target value "
        f"'{pretrained_load_target}'. Expected 'cpu' or 'model_device'."
    )


def load_swin_encoder_pretrained(
    model: SwinViT3D,
    ckpt_path: Path | str,
    *,
    strict: bool = False,
    map_location: str | torch.device | None = None,
    pretrained_load_target: str,
    unwrap_keys: tuple[str, ...],
    unwrap_max_depth: int,
    unknown_prefix_samples_max: int,
    shape_mismatch_preview: int,
) -> dict[str, Any]:
    """Load overlapping backbone weights into ``model.encoder``.

    Typical checkpoints (MONAI Swin-UNETR, SSL pretraining) store keys under
    ``swinViT.*`` or ``module.swinViT.*``. Those are renamed to match the
    parameter names inside :attr:`SwinViT3D.encoder`.

    Args:
        model: Built :class:`~src.models.swin_vit.SwinViT3D` instance.
        ckpt_path: Filesystem path to the checkpoint (``.pth`` / ``.pt``).
        strict: Forwarded to ``load_state_dict``. Use ``False`` for partial transfer.
        map_location: Optional explicit device for ``torch.load``.
        pretrained_load_target: Default load policy when ``map_location`` is
            not provided. Supported: ``cpu`` or ``model_device``.
        unwrap_keys: Wrapper keys checked recursively when unwrapping.
        unwrap_max_depth: Maximum recursive unwrap depth.
        unknown_prefix_samples_max: Number of unmapped-key samples to log.
        shape_mismatch_preview: Number of shape mismatch keys to preview in log.

    Returns:
        Dict with keys ``loaded_keys``, ``missing_keys``, ``unexpected_keys``,
        ``shape_mismatch_keys``, ``total_encoder_keys``, ``n_loaded``
        for structured logging.

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist (callers may avoid this
            by checking the path first; this function validates existence).
        TypeError: If the checkpoint root is not a mapping.
        ValueError: If checkpoint nesting is malformed.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path.resolve()}")

    loc = _resolve_map_location(model, map_location, pretrained_load_target)
    raw = torch.load(path, map_location=loc, weights_only=True)
    flat = _unwrap_checkpoint(
        raw,
        unwrap_keys=unwrap_keys,
        unwrap_max_depth=unwrap_max_depth,
    )

    encoder_ref = model.encoder.state_dict()
    renamed: dict[str, torch.Tensor] = {}
    unknown_prefix_samples: list[str] = []

    for key, value in flat.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        nk = _normalize_encoder_key(key)
        if nk not in encoder_ref:
            if len(unknown_prefix_samples) < unknown_prefix_samples_max and "." in key:
                unknown_prefix_samples.append(key)
            continue
        renamed[nk] = value

    filtered: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []
    for k, v in renamed.items():
        ref = encoder_ref[k]
        if ref.shape != v.shape:
            shape_mismatch.append(f"{k}: ckpt={tuple(v.shape)} model={tuple(ref.shape)}")
            continue
        filtered[k] = v

    incompatible = model.encoder.load_state_dict(filtered, strict=strict)

    n_loaded = len(filtered)
    if n_loaded == 0:
        raise RuntimeError(
            f"Zero weights loaded from {path.name}. This usually means checkpoint "
            "prefixes or architecture do not match encoder structure. "
            "Check normalization rules and model config."
        )
    total = len(encoder_ref)
    logger.info(
        "Loaded pretrained encoder weights from %s | matched=%d/%d tensors | "
        "strict=%s",
        path.resolve(),
        n_loaded,
        total,
        strict,
    )
    if shape_mismatch:
        logger.warning(
            "Skipped %d tensors due to shape mismatch (align swin_vit.* YAML "
            "with the checkpoint recipe): %s",
            len(shape_mismatch),
            "; ".join(shape_mismatch[:shape_mismatch_preview])
            + ("; ..." if len(shape_mismatch) > shape_mismatch_preview else ""),
        )
    if unknown_prefix_samples:
        logger.debug(
            "Sample checkpoint keys not mapped to encoder (first %d): %s",
            unknown_prefix_samples_max,
            unknown_prefix_samples,
        )
    if incompatible.missing_keys:
        logger.info(
            "Encoder missing_keys after load (%d): head layers still random — expected.",
            len(incompatible.missing_keys),
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "unexpected_keys from load_state_dict: %s",
            incompatible.unexpected_keys[:10],
        )

    return {
        "loaded_keys": sorted(filtered.keys()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "shape_mismatch_keys": shape_mismatch,
        "total_encoder_keys": total,
        "n_loaded": n_loaded,
        "checkpoint_path": str(path.resolve()),
    }
