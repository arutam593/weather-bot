"""
Anomaly detection.

Two complementary detectors:

  1) IsolationForest over the current feature vector
     → catches unusual conjunctions of inputs (e.g. a feature vector this
       location has never shown).
  2) CUSUM over the residual stream (prediction - actual)
     → catches gradual drift or sudden regime change in model skill.

When any detector fires, the prediction is still issued but flagged, and
the alert engine is consulted for possible user-facing warnings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Deque
from collections import deque

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

log = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    is_anomaly: bool
    feature_score: float           # IsolationForest score (lower = more anomalous)
    residual_cusum: float          # current CUSUM statistic
    reasons: list[str]


class AnomalyDetector:

    def __init__(self, contamination: float = 0.02,
                 cusum_threshold: float = 4.0,
                 cusum_window: int = 48):
        if IsolationForest is None:
            raise ImportError("scikit-learn is required for AnomalyDetector")
        self.iforest = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=200)
        self.cusum_threshold = cusum_threshold
        self._iforest_fitted = False
        self._feature_cols: list[str] | None = None     # set at fit time
        self._pos: dict[str, Deque[float]] = {}
        self._neg: dict[str, Deque[float]] = {}
        self._window = cusum_window

    def fit_feature_detector(self, X: pd.DataFrame) -> None:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        self._feature_cols = list(X.columns)
        self.iforest.fit(X.values)
        self._iforest_fitted = True
        log.info("anomaly: iforest fitted on %d samples × %d features",
                 len(X), X.shape[1])

    def check(self, X_current: pd.DataFrame) -> AnomalyReport:
        reasons: list[str] = []

        # Feature-space anomaly — align columns to training schema
        feature_score = 0.0
        is_anom_feat = False
        if self._iforest_fitted and len(X_current) > 0:
            cols = self._feature_cols or list(X_current.columns)
            X = X_current.reindex(columns=cols, fill_value=0.0)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            scores = self.iforest.score_samples(X.values)
            feature_score = float(scores[-1])
            # Compare against the trained model's natural threshold
            # (offset_ is the contamination-derived cutoff used by `predict`).
            is_anom_feat = feature_score < self.iforest.offset_
            if is_anom_feat:
                reasons.append("feature-vector outlier vs. training distribution")

        # CUSUM summary across variables (max of their stats)
        max_cusum = max(
            (max(self._cusum_stat(var, "pos"), self._cusum_stat(var, "neg"))
             for var in self._pos.keys()),
            default=0.0,
        )
        is_anom_cusum = max_cusum > self.cusum_threshold
        if is_anom_cusum:
            reasons.append(
                f"residual drift (CUSUM={max_cusum:.2f} > {self.cusum_threshold})")

        return AnomalyReport(
            is_anomaly=is_anom_feat or is_anom_cusum,
            feature_score=feature_score,
            residual_cusum=max_cusum,
            reasons=reasons,
        )

    # ------------------------------------------------------------------

    def observe_residual(self, variable: str, residual: float,
                         target_std: float) -> None:
        """Update CUSUM with a new (prediction - actual) residual."""
        # Normalize residual by typical spread so thresholds are unit-free.
        z = residual / max(target_std, 1e-6)
        pos = self._pos.setdefault(variable, deque(maxlen=self._window))
        neg = self._neg.setdefault(variable, deque(maxlen=self._window))
        prev_pos = pos[-1] if pos else 0.0
        prev_neg = neg[-1] if neg else 0.0
        pos.append(max(0.0, prev_pos + z - 0.5))   # slack = 0.5σ
        neg.append(max(0.0, prev_neg - z - 0.5))

    def _cusum_stat(self, variable: str, side: str) -> float:
        d = self._pos if side == "pos" else self._neg
        return d.get(variable, deque([0.0]))[-1] if d.get(variable) else 0.0
