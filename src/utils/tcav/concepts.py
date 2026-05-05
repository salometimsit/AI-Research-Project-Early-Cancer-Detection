"""Concept-set construction from radiomics tables.

Pipeline stage: post-K interpretability of the SwinViT track.

Inputs
------
* Radiomics CSV (``data/radiomics_features.csv``) -- one row per patient.
* VaRFS-stable feature selection JSON (``data/varfs_selected.json``) for
  per-feature concepts.
* Optional labels CSV for per-family PCA-1 sign orientation.

Outputs (in-memory only)
------------------------
* :data:`ConceptSets` -- ``{concept_name: (positive_pids, negative_pids)}``.
* List of :class:`FamilySign` decisions (per-family mode).

Failure modes
-------------
* ``radiomics_csv`` missing the ``patient_id`` column -> ``ValueError``.
* Quartile threshold yielding empty positive / negative arms -> the
  concept is logged and skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "TCAV concepts require numpy, pandas and scikit-learn. "
        "Install them to enable RadiomicsConceptBuilder."
    ) from exc

from src.utils.logger import get_logger
from src.utils.tcav.types import ConceptSets, FamilySign

logger = get_logger(__name__)


class RadiomicsConceptBuilder:
    """Build TCAV concept sets from radiomics tables.

    Parameters
    ----------
    quartile:
        Fraction of patients per tail (default ``0.25``);
        ``positives = top quartile``, ``negatives = bottom quartile``.
    family_prefix_map:
        Optional override mapping family -> substring used to identify
        feature columns. Defaults to PyRadiomics ``original_<family>_*``
        naming.
    """

    DEFAULT_FAMILY_PREFIXES = {
        "firstorder": "_firstorder_",
        "glcm": "_glcm_",
        "glrlm": "_glrlm_",
        "glszm": "_glszm_",
        "shape": "_shape_",
    }

    def __init__(
        self,
        quartile: float = 0.25,
        family_prefix_map: dict[str, str] | None = None,
    ) -> None:
        if not 0.0 < quartile < 0.5:
            raise ValueError("quartile must be in (0, 0.5).")
        self.quartile = float(quartile)
        self.family_prefix_map = dict(
            family_prefix_map or self.DEFAULT_FAMILY_PREFIXES
        )

    @staticmethod
    def _split_top_bottom(
        scores: pd.Series, quartile: float
    ) -> tuple[list[str], list[str]]:
        """Return ``(top_pids, bottom_pids)`` for a series indexed by patient_id."""
        if scores.empty:
            return [], []
        scores = scores.dropna()
        if scores.empty:
            return [], []
        upper_q = scores.quantile(1.0 - quartile)
        lower_q = scores.quantile(quartile)
        positives = scores[scores >= upper_q].index.astype(str).tolist()
        negatives = scores[scores <= lower_q].index.astype(str).tolist()
        return positives, negatives

    def build_per_feature_concepts(
        self,
        radiomics_csv: Path | str,
        selected_features_json: Path | str,
        top_n: int,
    ) -> ConceptSets:
        """One concept per top-N VaRFS-stable feature.

        ``selected_features_json`` is the artefact written by
        :class:`src.features.varfs_selection.VaRFSSelector.save_selection`;
        the ``stability_scores`` field is used to rank features.
        """
        radio_df = pd.read_csv(radiomics_csv)
        if "patient_id" not in radio_df.columns:
            raise ValueError(
                "radiomics_csv must contain a 'patient_id' column."
            )
        radio_df = radio_df.set_index("patient_id")

        with open(selected_features_json, "r", encoding="utf-8") as fh:
            selection = json.load(fh)
        stability = selection.get("stability_scores", {})
        ranked = sorted(stability.items(), key=lambda kv: kv[1], reverse=True)

        concepts: ConceptSets = {}
        for feature, _stab in ranked[:top_n]:
            if feature not in radio_df.columns:
                logger.warning(
                    "Skipping concept '%s' (column missing in radiomics CSV).",
                    feature,
                )
                continue
            pos, neg = self._split_top_bottom(radio_df[feature], self.quartile)
            if not pos or not neg:
                logger.warning(
                    "Skipping concept '%s' (empty pos/neg set).", feature
                )
                continue
            concepts[f"feat::{feature}"] = (pos, neg)
        logger.info(
            "Built %d per-feature concepts (quartile=%.2f).",
            len(concepts),
            self.quartile,
        )
        return concepts

    def build_per_family_concepts(
        self,
        radiomics_csv: Path | str,
        labels_csv: Path | str | None,
        families: Iterable[str],
    ) -> tuple[ConceptSets, list[FamilySign]]:
        """One concept per radiomics family via PCA-1.

        When ``labels_csv`` is provided we flip PCA-1's sign so that the
        positive direction always correlates with ``label==1`` (more
        HCC-like). Without labels the sign defaults to ``+1``.
        """
        radio_df = pd.read_csv(radiomics_csv)
        if "patient_id" not in radio_df.columns:
            raise ValueError("radiomics_csv must contain a 'patient_id' column.")
        radio_df = radio_df.set_index("patient_id")

        labels: pd.Series | None = None
        if labels_csv is not None and Path(labels_csv).exists():
            label_df = pd.read_csv(labels_csv).set_index("patient_id")
            labels = label_df["label"].astype(int)

        concepts: ConceptSets = {}
        signs: list[FamilySign] = []
        for family in families:
            family_concept = self._build_one_family(family, radio_df, labels)
            if family_concept is None:
                continue
            concept_name, (pos, neg), sign = family_concept
            signs.append(FamilySign(family=family, sign=sign))
            concepts[concept_name] = (pos, neg)
        logger.info(
            "Built %d per-family concepts (quartile=%.2f).",
            len(concepts),
            self.quartile,
        )
        return concepts, signs

    def _build_one_family(
        self,
        family: str,
        radio_df: pd.DataFrame,
        labels: pd.Series | None,
    ) -> tuple[str, tuple[list[str], list[str]], int] | None:
        """Return ``(concept_name, (pos, neg), sign)`` or ``None`` on skip."""
        prefix = self.family_prefix_map.get(family, f"_{family}_")
        cols = [c for c in radio_df.columns if prefix in c]
        if len(cols) < 2:
            logger.warning(
                "Family '%s' has %d columns; skipping.", family, len(cols)
            )
            return None
        sub = radio_df[cols].replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna(axis=1, how="all")
        if sub.shape[1] < 2:
            logger.warning(
                "Family '%s' lost too many columns after NaN drop; skipping.",
                family,
            )
            return None
        sub = sub.fillna(sub.median(numeric_only=True))
        scaled = StandardScaler().fit_transform(sub.values)
        pca = PCA(n_components=1, random_state=0)
        pc1 = pca.fit_transform(scaled).ravel()
        score_series = pd.Series(pc1, index=sub.index)

        sign = 1
        if labels is not None:
            aligned = labels.reindex(score_series.index).dropna()
            if not aligned.empty:
                common = score_series.loc[aligned.index]
                if common.std() > 0:
                    corr = float(np.corrcoef(common.values, aligned.values)[0, 1])
                    if not np.isnan(corr) and corr < 0:
                        sign = -1
                score_series = score_series * sign

        pos, neg = self._split_top_bottom(score_series, self.quartile)
        if not pos or not neg:
            logger.warning(
                "Family concept '%s' produced empty split; skipping.", family
            )
            return None
        return f"fam::{family}", (pos, neg), sign

    def build_random_concepts(
        self,
        all_pids: list[str],
        n_concepts: int,
        size: int,
        seed: int = 0,
    ) -> ConceptSets:
        """Random patient splits used as a null distribution."""
        rng = np.random.default_rng(seed)
        n = len(all_pids)
        if n < 2 * size:
            size = max(2, n // 2)
            logger.warning(
                "Not enough patients for requested random concept size; "
                "using size=%d instead.",
                size,
            )
        concepts: ConceptSets = {}
        for i in range(n_concepts):
            shuffled = rng.permutation(np.asarray(all_pids))
            pos = shuffled[:size].tolist()
            neg = shuffled[size : 2 * size].tolist()
            concepts[f"random::{i:03d}"] = (pos, neg)
        logger.info(
            "Built %d random concepts (size=%d).", len(concepts), size
        )
        return concepts
