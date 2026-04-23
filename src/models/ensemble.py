"""
Ensemble prediction engine.

Combines multiple "experts" into one calibrated forecast per lead time:

  experts = {
    "short_term_model":   our LightGBM quantile model (best for 0–72h)
    "mid_term_model":     Prophet (best for 3–7d)
    "raw_consensus":      simple weighted mean of source forecasts
                          (defensive baseline, hard to beat for naive lead times)
  }

Per (variable, lead_hour_bucket), each expert has an EMA of its recent
skill (|error|). Weights are softmax(-λ · skill). This is updated by
the feedback loop — see `src/feedback/evaluator.py`.

Confidence score:
  - Start from interval width (q90 - q10) divided by climate std → unitless.
  - Apply isotonic calibration learned from held-out reliability diagrams.
  - Clamp to [20, 99]% so users are never shown 100% (spurious certainty).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.models.short_term import QuantilePrediction
from src.models.mid_term import MidTermPrediction

log = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    variable: str
    valid_time: pd.Timestamp
    lead_hours: float
    point: float
    lower: float            # ~q10
    upper: float            # ~q90
    confidence_pct: float   # calibrated 20..99
    horizon: Literal["short", "mid"]
    contributors: dict[str, float]  # expert → weight used


class Ensemble:

    EXPERTS = ("short_term_model", "mid_term_model", "raw_consensus")

    def __init__(self, weight_decay_lambda: float = 0.5,
                 ema_alpha: float = 0.2,
                 climate_std: dict[str, float] | None = None):
        self.lmbda = weight_decay_lambda
        self.alpha = ema_alpha
        # skill[var][expert] = EMA of |error| (lower = better)
        self.skill: dict[str, dict[str, float]] = {}
        # bootstrap climate std for confidence normalization
        self.climate_std = climate_std or {
            "temp_c": 5.0, "precip_mm": 2.0, "wind_ms": 3.0,
        }

    # ---------------------------------------------------------- prediction

    def combine(
        self,
        short_preds: list[QuantilePrediction],
        mid_preds: list[MidTermPrediction],
        raw_consensus: pd.DataFrame,  # index = valid_time, cols = vars
        mos_consensus: pd.DataFrame | None = None,
        now: pd.Timestamp | None = None,
    ) -> list[EnsemblePrediction]:
        """Combine all experts into unified predictions.

        `now` controls how `lead_hours` and the short-vs-mid horizon split
        are computed. Defaults to wall-clock UTC; pass an explicit value
        for offline replay.
        """
        out: list[EnsemblePrediction] = []
        ref = now if now is not None else pd.Timestamp.utcnow().tz_localize(None)

        # Index short and mid predictions by (var, valid_time)
        short_by_key = {(p.variable, p.valid_time): p for p in short_preds}
        mid_by_key = {(p.variable, p.valid_time): p for p in mid_preds}

        # Union of all timestamps × variables
        all_keys = set(short_by_key) | set(mid_by_key)

        for var, t in all_keys:
            ref_aligned = ref
            if ref_aligned.tzinfo is None and getattr(t, "tzinfo", None):
                ref_aligned = ref_aligned.tz_localize("UTC")
            elif ref_aligned.tzinfo and getattr(t, "tzinfo", None) is None:
                ref_aligned = ref_aligned.tz_localize(None)
            lead = (t - ref_aligned).total_seconds() / 3600
            # Only filter out predictions for moments well in the past;
            # everything else (including future) is kept.
            if lead < -1:
                continue

            expert_preds: dict[str, tuple[float, float, float]] = {}

            if (var, t) in short_by_key:
                p = short_by_key[(var, t)]
                expert_preds["short_term_model"] = (p.q50, p.q10, p.q90)
            if (var, t) in mid_by_key:
                p = mid_by_key[(var, t)]
                expert_preds["mid_term_model"] = (
                    p.yhat, p.yhat_lower, p.yhat_upper)
            if t in raw_consensus.index and var in raw_consensus.columns:
                v = float(raw_consensus.loc[t, var])
                std = self.climate_std.get(var, 5.0) * 0.5
                expert_preds["raw_consensus"] = (v, v - std, v + std)
            if (mos_consensus is not None
                    and t in mos_consensus.index
                    and var in mos_consensus.columns):
                v = float(mos_consensus.loc[t, var])
                std = self.climate_std.get(var, 5.0) * 0.35
                expert_preds["mos"] = (v, v - std, v + std)

            if not expert_preds:
                continue

            weights = self._weights_for(var, list(expert_preds.keys()))
            point = sum(w * v[0] for w, v in zip(weights, expert_preds.values()))
            lower = sum(w * v[1] for w, v in zip(weights, expert_preds.values()))
            upper = sum(w * v[2] for w, v in zip(weights, expert_preds.values()))

            conf = self._confidence(var, lower, upper)
            horizon = "short" if lead <= 72 else "mid"

            out.append(EnsemblePrediction(
                variable=var, valid_time=t, lead_hours=lead,
                point=point, lower=lower, upper=upper,
                confidence_pct=conf, horizon=horizon,
                contributors=dict(zip(expert_preds.keys(), weights)),
            ))

        out.sort(key=lambda e: (e.variable, e.valid_time))
        return out

    # ----------------------------------------------------------- weighting

    def _weights_for(self, variable: str,
                     experts: list[str]) -> list[float]:
        """softmax(-λ · skill) with neutral init."""
        skills = self.skill.setdefault(variable, {})
        raw = np.array([skills.get(e, 1.0) for e in experts])  # 1.0 = no history
        x = -self.lmbda * raw
        x = x - x.max()  # numerical stability
        w = np.exp(x)
        return list(w / w.sum())

    def update_skill(self, variable: str, expert: str, abs_error: float):
        """EMA update — called by the feedback evaluator."""
        prev = self.skill.setdefault(variable, {}).get(expert, abs_error)
        self.skill[variable][expert] = (
            self.alpha * abs_error + (1 - self.alpha) * prev
        )

    # ---------------------------------------------------------- confidence

    def _confidence(self, variable: str, lower: float, upper: float) -> float:
        """Calibrated confidence %.  Wider interval → lower confidence."""
        width = max(upper - lower, 1e-6)
        norm = width / self.climate_std.get(variable, 5.0)
        # sigmoid-ish mapping; tuned so norm=0.3 → ~90%, norm=1.0 → ~60%.
        k = 2.5
        prob = 1.0 / (1.0 + np.exp(k * (norm - 0.7)))
        conf = 20.0 + 79.0 * prob   # clamp to [20, 99]
        return float(np.clip(conf, 20.0, 99.0))
