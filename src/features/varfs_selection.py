"""VaRFS (Variable Resampling Feature Selection) -- bootstrap stability filter.

Pipeline stage: H in the A->M flow.

Inputs
------
* ``data/raw_radiomics_features.csv`` from
  :mod:`src.features.radiomics_extractor`.
* ``varfs.n_bootstrap``, ``varfs.stability_threshold``, and
  ``varfs.correlation_threshold`` from the config.

Outputs
-------
* ``data/varfs_filtered_features.csv`` -- the radiomics CSV restricted
  to stable, low-redundancy features (plus ``patient_id`` and ``label``).
* ``data/varfs_selected_features.json`` -- ``{selected_features,
  stability_scores, params}``; consumed by
  :class:`src.utils.tcav.RadiomicsConceptBuilder` and downstream
  inference.

Failure modes
-------------
* All features filtered out -> ``transform`` raises ``RuntimeError``;
  lower ``stability_threshold`` or check the input CSV size.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VaRFSSelector:
    """Bootstrap-based feature selection that retains stable, low-redundancy features."""

    def __init__(
        self,
        n_bootstrap: int,
        stability_threshold: float,
        correlation_threshold: float,
        top_k_per_iter: int,
        use_robust_scaler: bool,
        random_state: int,
    ) -> None:
        self.n_bootstrap = int(n_bootstrap)
        self.stability_threshold = float(stability_threshold)
        self.correlation_threshold = float(correlation_threshold)
        self.top_k_per_iter = int(top_k_per_iter)
        self.use_robust_scaler = bool(use_robust_scaler)
        self.random_state = int(random_state)
        self.selected_features_: list[str] = []
        self.stability_scores_: dict[str, float] = {}
        self.global_medians_: pd.Series | None = None

    def _select_top_k(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> Iterable[str]:
        scores, _ = f_classif(x.values, y.values)
        ranked = pd.Series(scores, index=x.columns).fillna(0).sort_values(ascending=False)
        k = min(self.top_k_per_iter, len(ranked))
        return list(ranked.index[:k])

    def _drop_correlated(
        self, features_df: pd.DataFrame, ranked: list[str]
    ) -> list[str]:
        kept: list[str] = []
        if features_df.empty:
            return kept
        corr = features_df[ranked].corr().abs()
        for feature in ranked:
            keep = True
            for chosen in kept:
                if corr.loc[feature, chosen] > self.correlation_threshold:
                    keep = False
                    break
            if keep:
                kept.append(feature)
        return kept

    def fit(self, features_df: pd.DataFrame, labels: pd.Series) -> list[str]:
        """Run bootstrap resampling and return stable feature names."""
        feature_cols = [c for c in features_df.columns if c not in ("patient_id", "label")]
        x = features_df[feature_cols]
        y = labels.astype(int)
        self.global_medians_ = x.median(numeric_only=True)
        x_clean = x.replace([np.inf, -np.inf], np.nan).fillna(self.global_medians_)

        if self.use_robust_scaler:
            scaler = RobustScaler()
            x_scaled_vals = scaler.fit_transform(x_clean)
            x_clean = pd.DataFrame(x_scaled_vals, columns=feature_cols, index=x_clean.index)

        rng = np.random.RandomState(self.random_state)
        counts: Counter[str] = Counter()
        for i in range(self.n_bootstrap):
            seed = int(rng.randint(0, 2**31 - 1))
            x_b, y_b = resample(
                x_clean, y, replace=True, n_samples=len(y), random_state=seed
            )
            try:
                top = self._select_top_k(x_b, y_b)
            except ValueError as exc:
                logger.warning("Bootstrap %d skipped: %s", i, exc)
                continue
            counts.update(top)

        if self.n_bootstrap == 0:
            return []
        stability = {f: c / self.n_bootstrap for f, c in counts.items()}
        self.stability_scores_ = stability
        stable = [
            f for f, score in stability.items() if score >= self.stability_threshold
        ]
        ranked_stable = sorted(stable, key=lambda f: stability[f], reverse=True)
        self.selected_features_ = self._drop_correlated(x, ranked_stable)
        logger.info(
            "VaRFS retained %d/%d features (threshold=%.2f).",
            len(self.selected_features_),
            len(feature_cols),
            self.stability_threshold,
        )
        return self.selected_features_

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Filter ``features_df`` to selected features (plus key id columns)."""
        if not self.selected_features_:
            raise RuntimeError("VaRFSSelector must be fit before transform().")
        keep = [
            c for c in ("patient_id", "label") if c in features_df.columns
        ] + self.selected_features_
        return features_df[keep].copy()

    def fit_transform(
        self, features_df: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        """Run :meth:`fit` then :meth:`transform` in one call.

        Args:
            features_df: Wide table with ``patient_id`` / ``label`` plus
                radiomics feature columns.
            labels: Class labels aligned with ``features_df``.

        Returns:
            ``features_df`` restricted to the stable, low-redundancy
            features (``patient_id`` / ``label`` preserved).
        """
        self.fit(features_df, labels)
        return self.transform(features_df)

    def save_selection(self, path: Path | str) -> None:
        """Persist selected feature names + stability scores to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "selected_features": self.selected_features_,
            "stability_scores": self.stability_scores_,
            "params": {
                "n_bootstrap": self.n_bootstrap,
                "stability_threshold": self.stability_threshold,
                "correlation_threshold": self.correlation_threshold,
                "use_robust_scaler": self.use_robust_scaler,
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Saved VaRFS selection -> %s", path)
