"""
Confidence-score calibration via isotonic regression.

Problem: an interval-width-derived confidence score is biased — claiming
"80%" but only being right 65% of the time is over-confidence, and the
opposite is under-confidence. Both are bad: over-confidence makes alerts
fire spuriously, under-confidence loses user trust.

Solution: collect (claimed_confidence, was_truth_inside_interval) pairs
from past predictions, fit a monotonic mapping with isotonic regression,
and apply it as a post-hoc correction inside the ensemble.

Theory: this is the same machinery used to calibrate classifier
probabilities (Niculescu-Mizil & Caruana, 2005) and the standard fix
for forecast reliability diagrams.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None

from src.feedback.store import PredictionStore

log = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    n_samples: int
    pre_brier: float       # before calibration
    post_brier: float      # after
    bins: list[tuple[float, float, int]]   # (claimed, observed, n)


class ConfidenceCalibrator:
    """Per-variable isotonic recalibration of confidence percentages."""

    def __init__(self):
        if IsotonicRegression is None:
            raise ImportError("scikit-learn is required for ConfidenceCalibrator")
        # one isotonic mapping per variable
        self.maps: dict[str, IsotonicRegression] = {}

    # -------------------------------------------------- public

    def fit_from_store(self, store: PredictionStore,
                       cutoff: datetime | None = None) -> CalibrationReport:
        """Pull historical (claim, hit) pairs and fit the maps."""
        pairs_by_var: dict[str, list[tuple[float, int]]] = {}
        with store.session() as s:
            from src.feedback.store import PredictionRow
            q = s.query(PredictionRow)
            if cutoff:
                q = q.filter(PredictionRow.created_at < cutoff)
            for p in q.all():
                actual = store.observation_at(
                    p.location_key, p.variable, p.valid_time)
                if actual is None:
                    continue
                hit = int(p.lower <= actual <= p.upper)
                pairs_by_var.setdefault(p.variable, []).append(
                    (p.confidence / 100.0, hit))

        n_total = sum(len(v) for v in pairs_by_var.values())
        if n_total < 50:
            log.warning("calibrator: only %d samples — refusing to fit "
                        "(need ~50+ per variable)", n_total)
            return CalibrationReport(n_samples=n_total, pre_brier=float("nan"),
                                     post_brier=float("nan"), bins=[])

        pre, post = [], []
        for var, pairs in pairs_by_var.items():
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(x, y)
            self.maps[var] = iso
            pre.append(np.mean((x - y) ** 2))
            post.append(np.mean((iso.predict(x) - y) ** 2))
            log.info("calibrator[%s]: %d samples, pre-Brier=%.3f, post-Brier=%.3f",
                     var, len(pairs), pre[-1], post[-1])

        bins = self._reliability_bins(pairs_by_var)
        return CalibrationReport(
            n_samples=n_total,
            pre_brier=float(np.mean(pre)),
            post_brier=float(np.mean(post)),
            bins=bins,
        )

    def apply(self, variable: str, raw_confidence_pct: float) -> float:
        """Apply the per-variable map. If unfitted, return input unchanged."""
        m = self.maps.get(variable)
        if m is None:
            return raw_confidence_pct
        recal = float(m.predict([raw_confidence_pct / 100.0])[0]) * 100.0
        # Re-clamp to [20, 99] for the same reasons as the raw scorer
        return float(np.clip(recal, 20.0, 99.0))

    # ------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.maps, f)

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibrator":
        c = cls()
        with open(path, "rb") as f:
            c.maps = pickle.load(f)
        return c

    # ------------------------------------------------ diagnostics

    @staticmethod
    def _reliability_bins(pairs_by_var: dict[str, list[tuple[float, int]]]
                          ) -> list[tuple[float, float, int]]:
        """Reliability diagram bins, aggregated across variables."""
        all_pairs = [p for ps in pairs_by_var.values() for p in ps]
        if not all_pairs:
            return []
        edges = np.linspace(0, 1, 11)   # 10 bins
        x = np.array([p[0] for p in all_pairs])
        y = np.array([p[1] for p in all_pairs])
        bins: list[tuple[float, float, int]] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (x >= lo) & (x < hi)
            n = int(mask.sum())
            if n > 0:
                bins.append((float((lo + hi) / 2), float(y[mask].mean()), n))
        return bins
