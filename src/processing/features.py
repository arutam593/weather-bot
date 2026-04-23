"""
Feature engineering.

Takes raw SourceReadings from multiple sources and produces a single
feature matrix `X` + target series `y` (one per predicted variable)
ready for model training or inference.

Key features built:
  • Per-source current/forecast values (temp, precip, wind, pressure, ...)
  • Lag features (1h, 3h, 6h, 12h, 24h) for each variable
  • Rolling statistics (mean, std, max) over 3h / 12h / 24h windows
  • Seasonal Fourier terms (annual + diurnal)
  • Geographic features (elevation, coast_distance, UHI, ruggedness)
  • Pressure tendency (dP/dt) — classic storm predictor
  • NLP event signals (hurricane_signal, flood_signal, ...)
  • Climate-normal deltas (temp - normal_for_doy) — anomaly-aware
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from src.ingestion.base import SourceReading
from src.ingestion.geographic import GeoFeatures
from src.ingestion.historical import ClimateNormal

log = logging.getLogger(__name__)

TARGET_VARIABLES = ["temp_c", "precip_mm", "wind_ms"]


@dataclass
class FeatureFrame:
    X: pd.DataFrame           # feature matrix, index = valid_time
    y: pd.DataFrame | None    # targets (if building training data), else None
    feature_names: list[str]


class FeatureBuilder:

    LAGS = [1, 3, 6, 12, 24]
    ROLL_WINDOWS = [3, 12, 24]

    def __init__(self, geo: GeoFeatures | None = None,
                 normals: dict[int, ClimateNormal] | None = None):
        self.geo = geo
        self.normals = normals or {}

    # ---------------------------------------------------------------- public

    def build_inference(
        self,
        readings: Iterable[SourceReading],
        nlp_signals: dict[str, float],
    ) -> FeatureFrame:
        """Build features for prediction (no targets)."""
        df = self._pivot_sources(readings)
        df = self._add_derived(df)
        df = self._add_geo(df)
        df = self._add_nlp(df, nlp_signals)
        df = self._add_climate_delta(df)
        df = self._add_seasonality(df)
        return FeatureFrame(X=df, y=None, feature_names=list(df.columns))

    def build_training(
        self,
        readings: Iterable[SourceReading],
        truth: pd.DataFrame,
        nlp_signals: dict[str, float],
    ) -> FeatureFrame:
        """
        Build features + targets for training.
        `truth` is indexed by valid_time with columns matching TARGET_VARIABLES.
        """
        frame = self.build_inference(readings, nlp_signals)
        # Force a suffix on every truth column so we can extract them
        # cleanly regardless of whether they collided with feature names.
        truth_suffixed = truth.add_suffix("__truth")
        aligned = frame.X.join(truth_suffixed, how="inner")
        y_cols = [c for c in TARGET_VARIABLES if c + "__truth" in aligned.columns]
        if not y_cols:
            raise ValueError(
                f"No target columns found in `truth`. "
                f"Expected one of {TARGET_VARIABLES}, got {list(truth.columns)}.")
        y = aligned[[c + "__truth" for c in y_cols]].rename(
            columns=lambda c: c.replace("__truth", ""))
        X = aligned.drop(columns=[c + "__truth" for c in y_cols])
        return FeatureFrame(X=X, y=y, feature_names=list(X.columns))

    # --------------------------------------------------------------- private

    def _pivot_sources(self, readings: Iterable[SourceReading]) -> pd.DataFrame:
        """Rows = valid_time, columns = <source>__<variable>."""
        rows = []
        for r in readings:
            for var, val in r.variables.items():
                rows.append({
                    "valid_time": r.valid_time,
                    "source": r.source,
                    "var": var,
                    "value": val,
                    "lead_hours": r.lead_hours,
                    "reliability": r.reliability_prior,
                })
        if not rows:
            return pd.DataFrame()

        long = pd.DataFrame(rows)
        # Weighted mean across sources per (time, var) — source reliability
        # is folded in here. The downstream model still sees per-source
        # columns too (below).
        wide_per_source = long.pivot_table(
            index="valid_time", columns=["source", "var"], values="value",
            aggfunc="mean",
        )
        wide_per_source.columns = [f"{s}__{v}"
                                   for s, v in wide_per_source.columns]

        # Source-weighted consensus per variable
        consensus = (long.assign(w_value=long["value"] * long["reliability"])
                         .groupby(["valid_time", "var"])
                         .agg(num=("w_value", "sum"), den=("reliability", "sum")))
        consensus["consensus"] = consensus["num"] / consensus["den"]
        consensus = consensus["consensus"].unstack("var")
        consensus.columns = [f"consensus__{c}" for c in consensus.columns]

        out = wide_per_source.join(consensus, how="outer").sort_index()
        return out

    def _add_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Pressure tendency — dP/dt, 3h window, a classic storm signal.
        if "consensus__pressure_hpa" in df.columns:
            df["pressure_tendency_3h"] = (
                df["consensus__pressure_hpa"].diff(3)
            )

        # Lag + rolling features for each consensus variable.
        for col in [c for c in df.columns if c.startswith("consensus__")]:
            var = col.replace("consensus__", "")
            for lag in self.LAGS:
                df[f"{var}_lag{lag}h"] = df[col].shift(lag)
            for w in self.ROLL_WINDOWS:
                df[f"{var}_roll{w}h_mean"] = df[col].rolling(w).mean()
                df[f"{var}_roll{w}h_std"]  = df[col].rolling(w).std()
                if var == "precip_mm":
                    df[f"{var}_roll{w}h_max"] = df[col].rolling(w).max()

        return df

    def _add_geo(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or self.geo is None:
            return df
        for k, v in self.geo.as_dict().items():
            if k in ("lat", "lon"):
                continue
            df[f"geo__{k}"] = v
        return df

    def _add_nlp(self, df: pd.DataFrame,
                 signals: dict[str, float]) -> pd.DataFrame:
        if df.empty:
            return df
        # NLP signals are scalar today — broadcast across all rows.
        # (A more advanced approach: time-decay the signal toward zero.)
        for k, v in signals.items():
            df[f"nlp__{k}"] = float(v)
        return df

    def _add_climate_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not self.normals:
            return df
        if "consensus__temp_c" not in df.columns:
            return df
        doy = df.index.to_series().apply(lambda t: t.timetuple().tm_yday)
        normal_mean = doy.map(lambda d: self.normals.get(d, ClimateNormal(
            d, df["consensus__temp_c"].mean(), 3.0, 0.0)).temp_mean_c)
        normal_std = doy.map(lambda d: self.normals.get(d, ClimateNormal(
            d, 0, 3.0, 0.0)).temp_std_c)
        df["climate__temp_anomaly_c"] = df["consensus__temp_c"] - normal_mean
        df["climate__temp_zscore"] = (
            (df["consensus__temp_c"] - normal_mean) / normal_std.replace(0, 1)
        )
        return df

    def _add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        t = df.index.to_series()
        # Annual cycle
        doy = t.apply(lambda x: x.timetuple().tm_yday) / 366.0
        df["season_sin"] = np.sin(2 * np.pi * doy)
        df["season_cos"] = np.cos(2 * np.pi * doy)
        # Diurnal cycle
        hod = t.apply(lambda x: x.hour + x.minute / 60.0) / 24.0
        df["diurnal_sin"] = np.sin(2 * np.pi * hod)
        df["diurnal_cos"] = np.cos(2 * np.pi * hod)
        return df
