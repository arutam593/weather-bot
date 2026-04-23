"""
Short-term prediction (0–72h) using LightGBM quantile regression.

Why quantile regression: we need calibrated prediction intervals, not just
a point forecast. We fit three models (q=0.1, 0.5, 0.9) per target variable.
Interval width [q0.9 - q0.1] directly feeds the confidence score.

Model is trained per target (temp, precip, wind) — simple and effective.
For 10+ targets, switch to a multi-output GBM or a neural joint model.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # allows import without lightgbm for structure tests

log = logging.getLogger(__name__)


@dataclass
class QuantilePrediction:
    variable: str
    valid_time: pd.Timestamp
    lead_hours: float
    q10: float
    q50: float       # median point forecast
    q90: float

    @property
    def interval_width(self) -> float:
        return self.q90 - self.q10


class ShortTermModel:
    """Per-target quantile LightGBM models (3 per target)."""

    DEFAULT_PARAMS = dict(
        objective="quantile", metric="quantile",
        learning_rate=0.05, num_leaves=31, max_depth=-1,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
        verbosity=-1, n_estimators=120,
    )

    def __init__(self, quantiles=(0.1, 0.5, 0.9),
                 horizon_hours: int = 72):
        if lgb is None:
            raise ImportError("lightgbm is required for ShortTermModel")
        self.quantiles = quantiles
        self.horizon_hours = horizon_hours
        # models[var][q] = fitted LGBMRegressor
        self.models: dict[str, dict[float, lgb.LGBMRegressor]] = {}
        self._feature_cols: list[str] | None = None

    # ---------------------------------------------------------- fit/predict

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "ShortTermModel":
        """y has one column per target variable."""
        X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        self._feature_cols = list(X.columns)
        for var in y.columns:
            self.models[var] = {}
            for q in self.quantiles:
                params = dict(self.DEFAULT_PARAMS, alpha=q)
                m = lgb.LGBMRegressor(**params)
                m.fit(X.values, y[var].values)
                self.models[var][q] = m
            log.info("short_term: fitted %s with %d features, %d samples",
                     var, X.shape[1], len(y))
        return self

    def predict(self, X: pd.DataFrame,
                now: pd.Timestamp | None = None) -> list[QuantilePrediction]:
        """Generate quantile predictions for every row in X.

        `now`, if given, is used to compute `lead_hours` on each prediction
        (useful for offline simulation). It does NOT filter rows — the caller
        decides which horizons to keep. This keeps the model decoupled from
        wall-clock time.
        """
        if not self.models:
            raise RuntimeError("model not fitted")
        # Be robust to feature-set drift between training and inference:
        # - columns the model expects but we don't have → filled with 0
        # - extra columns we have but the model never saw       → dropped
        # This matters because per-source columns (e.g. `open_meteo__temp_c`)
        # naturally vary across runs while the consensus + derived + geo
        # + nlp + seasonal features are stable.
        X_aligned = X.reindex(columns=self._feature_cols, fill_value=0.0)
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan) \
                             .ffill().fillna(0)

        ref = now if now is not None else pd.Timestamp.utcnow().tz_localize(None)
        if ref.tzinfo is None and X_aligned.index.tz is not None:
            ref = ref.tz_localize("UTC")

        preds: list[QuantilePrediction] = []
        for idx, row in X_aligned.iterrows():
            lead = (idx - ref).total_seconds() / 3600
            x = row.to_frame().T  # 1×N DataFrame so LGBM keeps column names
            for var, qmodels in self.models.items():
                q10 = float(qmodels[0.1].predict(x)[0])
                q50 = float(qmodels[0.5].predict(x)[0])
                q90 = float(qmodels[0.9].predict(x)[0])
                preds.append(QuantilePrediction(
                    variable=var, valid_time=idx, lead_hours=lead,
                    q10=q10, q50=q50, q90=q90,
                ))
        return preds

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"models": self.models,
                         "quantiles": self.quantiles,
                         "horizon_hours": self.horizon_hours,
                         "feature_cols": self._feature_cols}, f)

    @classmethod
    def load(cls, path: str | Path) -> "ShortTermModel":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        inst = cls(quantiles=blob["quantiles"],
                   horizon_hours=blob["horizon_hours"])
        inst.models = blob["models"]
        inst._feature_cols = blob["feature_cols"]
        return inst

    # ------------------------------------------------------------ SHAP hook

    def shap_values(self, X: pd.DataFrame, variable: str = "temp_c") -> np.ndarray:
        """TreeExplainer on the median model for the requested variable."""
        import shap
        X = X.reindex(columns=self._feature_cols, fill_value=0.0).fillna(0)
        model = self.models[variable][0.5]
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X)
