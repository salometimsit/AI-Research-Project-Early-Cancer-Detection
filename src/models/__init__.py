"""SwinViT, fusion, classifier head and trainer."""

from typing import Any

from src.models.classifier import FocalLoss, FusedHCCModel, HCCClassifier
from src.models.fusion import CrossAttentionFusion
from src.models.trainer import EarlyStopping, Trainer
from src.utils.logger import get_logger

_logger = get_logger(__name__)

try:
    from src.models.swin_vit import SwinViT3D
except ImportError as exc:
    _logger.warning(
        "Could not import SwinViT3D dependencies (likely `monai` or a "
        "torch extension); the SwinViT track will be unavailable. Install "
        "`monai` to enable SwinViT3D (original error: %s).",
        exc,
    )

    class SwinViT3D:  # type: ignore[no-redef]
        """Stub used when MONAI (or a required torch extension) is missing.

        The real implementation lives in :mod:`src.models.swin_vit`. When
        that module fails to import, this stub is bound in its place so
        the rest of ``src.models`` (fusion, classifier, trainer) can still
        load. Any attempt to instantiate it raises ``ImportError`` with a
        clear remediation message.
        """

        _import_error: ImportError = exc

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "SwinViT3D is unavailable because MONAI (or one of its "
                "torch-extension dependencies) failed to import. Install "
                "`monai` to enable the SwinViT track. "
                f"Original error: {self._import_error}"
            )


__all__ = [
    "CrossAttentionFusion",
    "EarlyStopping",
    "FocalLoss",
    "FusedHCCModel",
    "HCCClassifier",
    "SwinViT3D",
    "Trainer",
]
