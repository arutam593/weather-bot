"""
Plain-English explainer.

Given an EnsemblePrediction + the feature vector that produced it, emits
human-readable reasons. Two layers:

  1) Rule-based signals that are obvious and always worth surfacing:
       • pressure_tendency_3h < -3 hPa  → "sharp pressure drop"
       • nlp__hurricane_signal > 0.5    → "hurricane signal in news"
       • climate__temp_anomaly_c > 8    → "extreme warm anomaly vs. normal"
       • cloud_pct forecast trend       → "clearing / clouding up"

  2) SHAP-based top-k feature attributions translated through a friendly
     name map. This explains *why this particular number* and not
     something else — useful for trust and debugging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.ensemble import EnsemblePrediction

log = logging.getLogger(__name__)


_FRIENDLY_NAMES = {
    "pressure_tendency_3h":      "3-hour pressure change",
    "consensus__pressure_hpa":   "mean sea-level pressure",
    "consensus__temp_c":         "current temperature",
    "consensus__wind_ms":        "wind speed",
    "consensus__precip_mm":      "precipitation",
    "consensus__cloud_pct":      "cloud cover",
    "geo__elevation_m":          "elevation",
    "geo__coast_distance_km":    "distance from coast",
    "geo__urban_heat_index":     "urban heat-island intensity",
    "geo__terrain_ruggedness_m": "terrain ruggedness",
    "climate__temp_anomaly_c":   "temperature anomaly vs. climate normal",
    "climate__temp_zscore":      "temperature z-score vs. climate normal",
    "nlp__hurricane_signal":     "hurricane/cyclone news signal",
    "nlp__flood_signal":         "flood news signal",
    "nlp__heatwave_signal":      "heatwave news signal",
    "nlp__cold_front_signal":    "cold-front news signal",
    "nlp__storm_general_signal": "general storm news signal",
    "season_sin":                "seasonal cycle",
    "season_cos":                "seasonal cycle",
    "diurnal_sin":               "time of day",
    "diurnal_cos":               "time of day",
}


@dataclass
class Explanation:
    summary: str
    key_factors: list[str]
    raw_shap: list[tuple[str, float]]


class Explainer:

    def explain(
        self,
        pred: EnsemblePrediction,
        features: pd.Series,
        shap_row: np.ndarray | None,
        feature_names: list[str],
    ) -> Explanation:
        rule_factors = self._rule_based(features, pred)
        shap_factors, shap_pairs = self._shap_based(shap_row, features,
                                                    feature_names)

        # Deduplicate while preserving order; keep top ~4 reasons
        seen = set()
        ordered = []
        for f in rule_factors + shap_factors:
            if f not in seen:
                ordered.append(f)
                seen.add(f)
        top = ordered[:4]

        summary = self._compose_summary(pred, top)
        return Explanation(summary=summary, key_factors=top, raw_shap=shap_pairs)

    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based(feat: pd.Series, pred: EnsemblePrediction) -> list[str]:
        out: list[str] = []
        p_tend = feat.get("pressure_tendency_3h", np.nan)
        if pd.notna(p_tend):
            if p_tend < -3:
                out.append("sharp pressure drop (indicator of incoming storm)")
            elif p_tend > 3:
                out.append("rising pressure (stabilizing conditions)")

        for sig_name, friendly in [
            ("nlp__hurricane_signal",
             "active hurricane/cyclone coverage in recent news"),
            ("nlp__flood_signal",   "flood warnings in recent news"),
            ("nlp__heatwave_signal", "heatwave coverage in recent news"),
            ("nlp__cold_front_signal",
             "incoming cold front reported in recent news"),
        ]:
            if feat.get(sig_name, 0.0) >= 0.5:
                out.append(friendly)

        anom = feat.get("climate__temp_anomaly_c", np.nan)
        if pd.notna(anom):
            if anom > 8:
                out.append(
                    f"temperatures {anom:+.1f}°C vs. climate normal (very warm)")
            elif anom < -8:
                out.append(
                    f"temperatures {anom:+.1f}°C vs. climate normal (very cold)")

        return out

    @staticmethod
    def _shap_based(shap_row: np.ndarray | None, feat: pd.Series,
                    names: list[str]) -> tuple[list[str], list[tuple[str, float]]]:
        if shap_row is None:
            return [], []
        # Get top-k by absolute attribution
        order = np.argsort(-np.abs(shap_row))[:5]
        raw = [(names[i], float(shap_row[i])) for i in order]
        friendly = []
        for n, val in raw:
            label = _FRIENDLY_NAMES.get(n, n)
            direction = "pushed forecast up" if val > 0 else "pushed forecast down"
            friendly.append(f"{label} {direction}")
        return friendly, raw

    @staticmethod
    def _compose_summary(pred: EnsemblePrediction, factors: list[str]) -> str:
        when = pred.valid_time.strftime("%a %H:%M UTC")
        unit_map = {"temp_c": "°C", "precip_mm": "mm", "wind_ms": "m/s"}
        unit = unit_map.get(pred.variable, "")
        var_nice = {"temp_c": "temperature", "precip_mm": "precipitation",
                    "wind_ms": "wind speed"}.get(pred.variable, pred.variable)
        base = (f"At {when} (lead ~{pred.lead_hours:.0f}h), forecast "
                f"{var_nice}: {pred.point:.1f}{unit} "
                f"(interval {pred.lower:.1f}–{pred.upper:.1f}{unit}, "
                f"confidence {pred.confidence_pct:.0f}%).")
        if factors:
            base += " Key drivers: " + "; ".join(factors) + "."
        return base
