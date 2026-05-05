"""TCAV (Testing with Concept Activation Vectors) for the HCC dual-track model.

The TCAV implementation is split across four focused modules:

* :mod:`src.utils.tcav.types`     -- shared type aliases + dataclasses
  (``ConceptSets``, ``CAVRecord``, ``FamilySign``).
* :mod:`src.utils.tcav.concepts`  -- :class:`RadiomicsConceptBuilder` (per-feature,
  per-family via PCA-1, and random concept sets for null distribution).
* :mod:`src.utils.tcav.core`      -- :class:`TCAV3D`: forward-hook activation
  extraction, linear-CAV training, ``torch.autograd.grad`` directional
  derivatives, score, and permutation significance.
* :mod:`src.utils.tcav.io`        -- ``save_cavs_npz`` and
  ``save_cav_accuracies`` writers.

This ``__init__`` re-exports the public API so existing callers
(``scripts/run_tcav.py``, ``notebooks/11_tcav.ipynb``) work unchanged
with ``from src.utils.tcav import RadiomicsConceptBuilder, TCAV3D``.
"""

from __future__ import annotations

from src.utils.logger import get_logger

_logger = get_logger(__name__)

try:
    from src.utils.tcav.concepts import RadiomicsConceptBuilder
    from src.utils.tcav.core import TCAV3D
    from src.utils.tcav.io import save_cav_accuracies, save_cavs_npz
    from src.utils.tcav.types import CAVRecord, ConceptSets, FamilySign
except ImportError as exc:
    _logger.warning(
        "TCAV dependencies (torch, numpy, scikit-learn) are not installed. "
        "TCAV will be unavailable: %s",
        exc,
    )

__all__ = [
    "CAVRecord",
    "ConceptSets",
    "FamilySign",
    "RadiomicsConceptBuilder",
    "TCAV3D",
    "save_cav_accuracies",
    "save_cavs_npz",
]
