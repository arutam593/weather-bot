"""
MOS (Model Output Statistics) bias-correction.

Per-(variable, location) Ridge regressor that learns the systematic
residual of the NWP forecast and corrects for it.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
except ImportError:
    Ridge = None

log = logging.getLogger(__name__)


@dataclass
class MOSReport:
    variable: str
    n_train: int
    raw_mae: float
    mos_mae: float
    improvement_pct: float


class MOSCorrector:
    FEATURE_COLS = [
        "season_sin", "season_cos", "diurnal_sin", "diurnal_cos",
        "geo__elevation_m", "geo__coast_distance_km", "geo__urban_heat_index",
        "consensus__pressure_hpa", "consensus__cloud_pct",
        "consensus__humidity_pct", "pressure_tendency_3h",
    ]

    def __init__(self, alpha: float = 1.0):
        if Ridge is None:
            raise ImportError("scikit-learn required")
        self.alpha = alpha
        self.models: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, location_key: str):
        reports = []
        for var in y.columns:
            f_col = f"consensus__{var}"
            if f_col not in X.columns:
                continue
            forecast = X[f_col]
            actual = y[var]
            residual = actual - forecast
            feats = X[[c for c in self.FEATURE_COLS if c in X.columns]]
            mask = feats.notna().all(axis=1) & residual.notna()
            feats, residual = feats[mask], residual[mask]
            if len(feats) < 50:
                continue
            scaler = StandardScaler().fit(feats.values)
            ridge = Ridge(alpha=self.alpha).fit(
                scaler.transform(feats.values), residual.values)
            self.models[(var, location_key)] = (scaler, ridge)
            corr = forecast[mask] + ridge.predict(scaler.transform(feats.values))
            raw_mae = mean_absolute_error(actual[mask], forecast[mask])
            mos_mae = mean_absolute_error(actual[mask], corr)
            reports.append(MOSReport(
                variable=var, n_train=len(feats),
                raw_mae=float(raw_mae), mos_mae=float(mos_mae),
                improvement_pct=float(100 * (raw_mae - mos_mae) / max(raw_mae, 1e-9)),
            ))
        return reports

    def correct(self, X: pd.DataFrame, variable: str, location_key: str) -> pd.Series:
        f_col = f"consensus__{variable}"
        if f_col not in X.columns:
            return pd.Series(np.nan, index=X.index)
        raw = X[f_col]
        key = (variable, location_key)
        if key not in self.models:
            return raw
        scaler, ridge = self.models[key]
        feats = X[[c for c in self.FEATURE_COLS if c in X.columns]]
        mask = feats.notna().all(axis=1)
        if mask.sum() == 0:
            return raw
        delta = pd.Series(ridge.predict(scaler.transform(feats[mask].values)),
                          index=feats[mask].index)
        out = raw.copy()
        out.loc[delta.index] = raw.loc[delta.index] + delta
        return out

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"alpha": self.alpha, "models": self.models}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        c = cls(alpha=blob["alpha"])
        c.models = blob["models"]
        return c
