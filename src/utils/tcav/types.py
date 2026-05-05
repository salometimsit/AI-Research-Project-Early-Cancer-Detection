"""Shared types for the TCAV package.

Keeping these in a tiny standalone module avoids circular imports between
:mod:`concepts` and :mod:`core` and makes the public dataclasses easy to
spot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ConceptSets = dict[str, tuple[list[str], list[str]]]
"""Mapping ``concept_name -> (positive_pids, negative_pids)``."""


@dataclass(frozen=True)
class FamilySign:
    """Sign-orientation decision for a per-family PCA-1 concept.

    Attributes
    ----------
    family:
        Radiomics family name (e.g. ``"glcm"``).
    sign:
        ``+1`` or ``-1``; multiplied into PCA-1 so that "positive" always
        means "more HCC-like" (i.e. positively correlated with ``label==1``).
    """

    family: str
    sign: int


@dataclass
class CAVRecord:
    """A trained Concept Activation Vector plus metadata.

    Attributes
    ----------
    concept:
        Concept name as produced by :class:`RadiomicsConceptBuilder`
        (e.g. ``"feat::original_glcm_Idmn"`` or ``"fam::shape"``).
    vector:
        Unit-norm CAV in activation space (``(D,)``).
    train_accuracy:
        Linear classifier accuracy on the training set used to fit the CAV.
    n_positive, n_negative:
        Number of patient activations in each arm.
    """

    concept: str
    vector: np.ndarray
    train_accuracy: float
    n_positive: int
    n_negative: int
