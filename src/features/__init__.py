"""Radiomics extraction, VaRFS selection and classical baseline."""

from typing import Any

from src.utils.logger import get_logger

_logger = get_logger(__name__)

try:
    from src.features.baseline_classifier import RadiomicsBaseline, save_metrics
    from src.features.varfs_selection import VaRFSSelector
except ImportError as exc:
    _logger.warning(
        "Missing ML dependencies (scikit-learn/pandas). "
        "VaRFS and Baseline will be unavailable: %s",
        exc,
    )

    class RadiomicsBaseline:  # type: ignore[no-redef]
        """Stub used when scikit-learn or pandas is not installed."""

        _import_error: ImportError = exc

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "RadiomicsBaseline is unavailable because scikit-learn or "
                "pandas failed to import. "
                f"Original error: {self._import_error}"
            )

    class VaRFSSelector:  # type: ignore[no-redef]
        """Stub used when scikit-learn or pandas is not installed."""

        _import_error: ImportError = exc

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "VaRFSSelector is unavailable because scikit-learn or "
                "pandas failed to import. "
                f"Original error: {self._import_error}"
            )

    def save_metrics(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        raise ImportError(
            "save_metrics is unavailable because scikit-learn or pandas "
            f"failed to import. Original error: {exc}"
        )


try:
    from src.features.radiomics_extractor import RadiomicsExtractor
except ImportError as exc:
    _logger.warning(
        "Could not import pyradiomics dependencies; feature extraction will "
        "be unavailable. Install `pyradiomics` to enable RadiomicsExtractor "
        "(original error: %s).",
        exc,
    )

    class RadiomicsExtractor:  # type: ignore[no-redef]
        """Stub used when ``pyradiomics`` (or one of its deps) is not installed.

        The real implementation lives in
        :mod:`src.features.radiomics_extractor`. When that module fails to
        import, this stub is bound in its place so the rest of
        ``src.features`` can still load. Any attempt to instantiate it
        raises ``ImportError`` with a clear remediation message.
        """

        _import_error: ImportError = exc

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "RadiomicsExtractor is unavailable because pyradiomics (or "
                "one of its dependencies) failed to import. Install "
                "pyradiomics to enable radiomics feature extraction. "
                f"Original error: {self._import_error}"
            )


__all__ = [
    "RadiomicsBaseline",
    "RadiomicsExtractor",
    "VaRFSSelector",
    "save_metrics",
]
