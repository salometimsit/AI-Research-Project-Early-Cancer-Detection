"""Classical ML baseline trained on VaRFS-stable radiomics features.

Pipeline stage: I in the A->M flow.

Inputs
------
* ``data/varfs_filtered_features.csv`` from
  :mod:`src.features.varfs_selection`.
* ``baseline.*`` from the config (model type, hyperparameters,
  ``class_weight``).

Outputs
-------
* ``models/saved/radiomics_baseline.pkl`` -- serialized classifier.
* ``results/baseline_metrics.json`` -- AUC-ROC, sensitivity, specificity,
  PR-AUC for each CV fold plus aggregate mean/std.
* ``results/baseline_scores.csv`` -- per-patient ``predict_proba``
  scores for downstream ROC plotting.

Failure modes
-------------
* Empty filtered features -> ``RuntimeError`` from ``train``; revisit
  VaRFS thresholds.
* Single-class fold (label imbalance) -> AUC-ROC reported as ``nan``;
  metrics aggregator skips ``nan`` cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RadiomicsBaseline:
    """RandomForest (default) classifier over radiomics features."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Build a baseline classifier from the ``baseline`` config block.

        Args:
            cfg: Dict accepting ``model`` (only ``"random_forest"`` is
                supported), ``n_estimators``, ``max_depth``,
                ``class_weight`` (default ``"balanced"``), and
                ``random_state``.

        Raises:
            ValueError: If ``cfg["model"]`` is not ``"random_forest"``.
        """
        self.cfg = cfg
        model_name = str(cfg.get("model", "random_forest")).lower()
        if model_name != "random_forest":
            raise ValueError(f"Unsupported baseline model: {model_name}")
        self.model = self._build_model()
        imputation_cfg = cfg.get("imputation", {})
        scaling_cfg = cfg.get("scaling", {})
        safety_cfg = cfg.get("safety", {})
        self.imputation_strategy = str(imputation_cfg.get("strategy", "median"))
        self.scaling_enabled = bool(scaling_cfg.get("enabled", True))
        self.replace_inf_with_nan = bool(safety_cfg.get("replace_inf_with_nan", True))
        self.fail_fast_on_preprocess_error = bool(
            safety_cfg.get("fail_fast_on_preprocess_error", True)
        )
        self.feature_columns_: list[str] = []
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.oof_scores_: np.ndarray | None = None
        self.oof_pids_: list[str] = []

    def _build_model(self) -> RandomForestClassifier:
        """Build a fresh baseline model instance from config."""
        return RandomForestClassifier(
            n_estimators=int(self.cfg.get("n_estimators", 200)),
            max_depth=self.cfg.get("max_depth"),
            class_weight=self.cfg.get("class_weight", "balanced"),
            random_state=int(self.cfg.get("random_state", 42)),
            n_jobs=-1,
        )

    def _sanitize_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Replace non-finite values to keep preprocessing numerically stable."""
        if not self.replace_inf_with_nan:
            return x
        return x.replace([np.inf, -np.inf], np.nan)

    def _fit_preprocessor(self, x: pd.DataFrame) -> tuple[SimpleImputer, StandardScaler | None]:
        """Fit imputer/scaler on train data only."""
        imputer = SimpleImputer(strategy=self.imputation_strategy)
        x_imputed = imputer.fit_transform(x)
        scaler: StandardScaler | None = None
        if self.scaling_enabled:
            scaler = StandardScaler()
            scaler.fit(x_imputed)
        return imputer, scaler

    @staticmethod
    def _transform_features(
        x: pd.DataFrame,
        imputer: SimpleImputer,
        scaler: StandardScaler | None,
    ) -> np.ndarray:
        """Transform features with an already-fitted preprocessor."""
        x_imputed = imputer.transform(x)
        if scaler is not None:
            return scaler.transform(x_imputed)
        return x_imputed

    def _split_xy(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split a feature dataframe into ``(X, y)`` and remember the columns.

        Args:
            features_df: Wide table with ``patient_id`` / ``label``.

        Returns:
            ``(x, y)`` where ``x`` has only feature columns and ``y`` is the
            integer label series.
        """
        feature_cols = [
            c for c in features_df.columns if c not in ("patient_id", "label")
        ]
        self.feature_columns_ = feature_cols
        x = self._sanitize_features(features_df[feature_cols].copy())
        y = features_df["label"].astype(int)
        return x, y

    def train(self, features_df: pd.DataFrame) -> dict[str, float]:
        """Cross-validate, then refit on the full data.

        Stratified k-fold CV is run with ``n_folds = min(5, smallest
        class count)`` (and at least 2 folds). The OOF predictions are
        cached on ``self.oof_scores_`` / ``self.oof_pids_`` so callers
        can persist them without re-running the model.

        Args:
            features_df: Wide table with ``patient_id`` / ``label``.

        Returns:
            Dict with ``auc_roc_mean``, ``auc_roc_std``, ``pr_auc_mean``,
            ``accuracy_mean`` and ``n_folds``.
        """
        x, y = self._split_xy(features_df)
        min_class_count = int(np.bincount(y).min())
        if min_class_count < 2:
            raise RuntimeError(
                "Baseline CV requires at least 2 samples in each class."
            )
        cv_cfg = self.cfg.get("cv", {})
        max_folds = int(cv_cfg.get("max_folds", 5))
        n_folds = max(2, min(max_folds, min_class_count))
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=bool(cv_cfg.get("shuffle", True)),
            random_state=int(cv_cfg.get("random_state", 42)),
        )
        aucs: list[float] = []
        prs: list[float] = []
        accs: list[float] = []
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
            try:
                fold_imputer, fold_scaler = self._fit_preprocessor(x.iloc[train_idx])
                x_train = self._transform_features(
                    x.iloc[train_idx], fold_imputer, fold_scaler
                )
                x_val = self._transform_features(
                    x.iloc[val_idx], fold_imputer, fold_scaler
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Baseline preprocessing failed at CV fold %d: %s", fold, exc)
                if self.fail_fast_on_preprocess_error:
                    raise RuntimeError(
                        f"Baseline preprocessing failed at CV fold {fold}: {exc}"
                    ) from exc
                continue
            fold_model = self._build_model()
            fold_model.fit(x_train, y.iloc[train_idx])
            probs = fold_model.predict_proba(x_val)[:, 1]
            oof[val_idx] = probs
            preds = (probs >= 0.5).astype(int)
            try:
                aucs.append(roc_auc_score(y.iloc[val_idx], probs))
            except ValueError:
                pass
            prs.append(average_precision_score(y.iloc[val_idx], probs))
            accs.append(accuracy_score(y.iloc[val_idx], preds))
            logger.debug("Baseline fold %d AUC=%.3f", fold, aucs[-1] if aucs else float("nan"))

        self.oof_scores_ = oof
        self.oof_pids_ = features_df["patient_id"].astype(str).tolist()

        self.imputer, self.scaler = self._fit_preprocessor(x)
        x_full = self._transform_features(x, self.imputer, self.scaler)
        self.model = self._build_model()
        self.model.fit(x_full, y)
        metrics = {
            "auc_roc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_roc_std": float(np.std(aucs)) if aucs else float("nan"),
            "pr_auc_mean": float(np.mean(prs)),
            "accuracy_mean": float(np.mean(accs)),
            "n_folds": n_folds,
        }
        logger.info("Radiomics baseline metrics: %s", metrics)
        return metrics

    def evaluate(self, features_df: pd.DataFrame) -> dict[str, float]:
        """Score the *trained* model on ``features_df``.

        Args:
            features_df: Wide table with ``patient_id`` / ``label``.

        Returns:
            Dict with ``auc_roc``, ``pr_auc``, ``accuracy``.
        """
        x, y = self._split_xy(features_df)
        if self.imputer is None:
            raise RuntimeError("Model preprocessor not fitted. Call train() or load() first.")
        x_proc = self._transform_features(x, self.imputer, self.scaler)
        probs = self.model.predict_proba(x_proc)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "auc_roc": float(roc_auc_score(y, probs)),
            "pr_auc": float(average_precision_score(y, probs)),
            "accuracy": float(accuracy_score(y, preds)),
        }

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return ``P(label == 1)`` for every row of ``features_df``.

        Args:
            features_df: Wide table containing the feature columns
                stored at fit time.

        Returns:
            1D numpy array of probabilities aligned with
            ``features_df.index``.
        """
        if self.imputer is None:
            raise RuntimeError("Model preprocessor not fitted. Call train() or load() first.")
        x = self._sanitize_features(features_df[self.feature_columns_].copy())
        x_proc = self._transform_features(x, self.imputer, self.scaler)
        return self.model.predict_proba(x_proc)[:, 1]

    def save(self, path: Path | str) -> None:
        """Persist the trained sklearn model + feature columns + config.

        Args:
            path: Output ``.pkl`` path; parents are created on demand.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns_,
                "cfg": self.cfg,
                "imputer": self.imputer,
                "scaler": self.scaler,
            },
            path,
        )
        logger.info("Saved radiomics baseline -> %s", path)

    def load(self, path: Path | str) -> None:
        """Restore a previously :meth:`save`-d bundle.

        Args:
            path: Path to a ``.pkl`` produced by :meth:`save`.
        """
        path = Path(path)
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.feature_columns_ = bundle["feature_columns"]
        self.cfg = bundle.get("cfg", self.cfg)
        self.imputer = bundle.get("imputer")
        self.scaler = bundle.get("scaler")


def save_metrics(metrics: dict[str, float], path: Path | str) -> None:
    """Write metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved baseline metrics -> %s", path)
