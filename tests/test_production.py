"""
Production-quality tests for the ensemble, alert engine, and calibrator.
Run: pytest -q tests/test_production.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import Ensemble
from src.models.short_term import QuantilePrediction
from src.models.mid_term import MidTermPrediction
from src.alerts.alert_engine import AlertEngine
from src.models.anomaly import AnomalyReport


# ─────────────────────────────────────────────── Ensemble dynamic weighting

def _qp(var, t, q10, q50, q90):
    return QuantilePrediction(variable=var, valid_time=t, lead_hours=1.0,
                              q10=q10, q50=q50, q90=q90)


def _mp(var, t, point, lo, hi):
    return MidTermPrediction(variable=var, valid_time=t, lead_hours=24.0,
                             yhat=point, yhat_lower=lo, yhat_upper=hi)


def test_ensemble_weights_initial_uniform():
    """No history → equal weights across experts."""
    e = Ensemble()
    weights = e._weights_for("temp_c", ["short_term_model", "mid_term_model"])
    assert len(weights) == 2
    assert abs(weights[0] - weights[1]) < 1e-6
    assert abs(sum(weights) - 1.0) < 1e-6


def test_ensemble_weights_shift_toward_better_expert():
    """After the short-term model is much more accurate, it gets more weight."""
    e = Ensemble(weight_decay_lambda=1.0, ema_alpha=0.5)
    # Simulate 20 cycles where short is good (err ~0.5) and mid is bad (err ~5)
    for _ in range(20):
        e.update_skill("temp_c", "short_term_model", 0.5)
        e.update_skill("temp_c", "mid_term_model", 5.0)
    w = e._weights_for("temp_c", ["short_term_model", "mid_term_model"])
    assert w[0] > 0.9, f"short should dominate, got {w}"
    assert w[1] < 0.1


def test_ensemble_combine_uses_weighted_mean():
    """If both experts disagree on the point, the combined point sits between
    them, weighted by their skill EMAs."""
    t = pd.Timestamp("2026-04-22 12:00", tz="UTC")
    e = Ensemble(weight_decay_lambda=2.0, ema_alpha=1.0)
    # Train: short is 10× better than mid
    for _ in range(10):
        e.update_skill("temp_c", "short_term_model", 0.1)
        e.update_skill("temp_c", "mid_term_model", 1.0)

    short = [_qp("temp_c", t, 18.0, 20.0, 22.0)]
    mid   = [_mp("temp_c", t, 30.0, 27.0, 33.0)]
    consensus = pd.DataFrame()  # no raw consensus → only the two experts

    out = e.combine(short, mid, consensus, now=t - pd.Timedelta(hours=1))
    assert len(out) == 1
    # With short heavily weighted, the combined point should be close to 20
    assert out[0].point < 22.0, f"weighted mean should favor short, got {out[0].point}"


def test_ensemble_confidence_in_valid_range():
    """Confidence is always clamped to [20, 99]%."""
    t = pd.Timestamp("2026-04-22 12:00", tz="UTC")
    e = Ensemble(climate_std={"temp_c": 5.0})
    # Tiny interval → should still cap at 99
    out_tight = e.combine(
        [_qp("temp_c", t, 19.99, 20.0, 20.01)], [], pd.DataFrame(),
        now=t - pd.Timedelta(hours=1))
    # Huge interval → should still floor at 20
    out_wide = e.combine(
        [_qp("temp_c", t + pd.Timedelta(hours=1), -50.0, 20.0, 90.0)],
        [], pd.DataFrame(), now=t - pd.Timedelta(hours=1))
    assert 20.0 <= out_tight[0].confidence_pct <= 99.0
    assert 20.0 <= out_wide[0].confidence_pct <= 99.0
    assert out_tight[0].confidence_pct > out_wide[0].confidence_pct


# ──────────────────────────────────────────────────────── Alert engine

class _StubEnsemblePred:
    def __init__(self, var, point, conf=80, lower=None, upper=None):
        self.variable = var
        self.valid_time = datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc)
        self.point = point
        self.confidence_pct = conf
        self.lower = lower if lower is not None else point - 1
        self.upper = upper if upper is not None else point + 1
        self.lead_hours = 4.0


def test_alerts_fire_on_extreme_heat():
    cfg = dict(extreme_heat_c=38, extreme_cold_c=-15,
               heavy_rain_mm_per_h=15, severe_wind_ms=20)
    engine = AlertEngine(cfg)
    alerts = engine.evaluate([
        _StubEnsemblePred("temp_c", 42.0, conf=80),     # extreme heat
        _StubEnsemblePred("temp_c", 25.0, conf=90),     # normal
        _StubEnsemblePred("precip_mm", 22.0, conf=70),  # heavy rain
        _StubEnsemblePred("wind_ms", 28.0, conf=85),    # severe wind
    ], anomaly=None)
    codes = sorted(a.code for a in alerts)
    assert codes == ["EXTREME_HEAT", "HEAVY_RAIN", "SEVERE_WIND"]


def test_alerts_suppressed_when_low_confidence():
    """Extreme heat with low confidence shouldn't fire — the rule requires ≥55%."""
    engine = AlertEngine(dict(extreme_heat_c=38))
    alerts = engine.evaluate(
        [_StubEnsemblePred("temp_c", 42.0, conf=40)], None)
    assert not [a for a in alerts if a.code == "EXTREME_HEAT"]


def test_alerts_include_anomaly():
    engine = AlertEngine({})
    report = AnomalyReport(is_anomaly=True, feature_score=-0.3,
                            residual_cusum=5.0,
                            reasons=["feature outlier"])
    alerts = engine.evaluate([], anomaly=report)
    assert any(a.code == "ANOMALY" for a in alerts)


# ──────────────────────────────────────────────────── Calibrator monotonicity

def test_calibrator_is_monotonic_after_fit():
    """Isotonic regression is by construction monotonic; verify on synthetic
    miscalibrated data."""
    pytest.importorskip("sklearn")
    from src.feedback.calibrator import ConfidenceCalibrator
    from src.feedback.store import PredictionStore
    import tempfile

    cal = ConfidenceCalibrator()
    # Build synthetic (claim, hit) pairs where the model is over-confident:
    # at claimed 90%, only 70% are right; at 50%, only 40%.
    rng = np.random.default_rng(0)
    n = 500
    claims = rng.uniform(0.2, 0.99, n)
    # truth-rate = 0.7 * claim (over-confident pattern)
    p_correct = 0.7 * claims
    hits = (rng.uniform(size=n) < p_correct).astype(int)

    # Use a temp store to round-trip through the public API
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        store = PredictionStore(f"sqlite:///{tf.name}")
        pred_rows = [{
            "location_key": "0,0", "lat": 0.0, "lon": 0.0,
            "variable": "temp_c", "valid_time": datetime(2026, 1, 1, i // 24,
                                                          tzinfo=timezone.utc),
            "lead_hours": 1.0, "point": 0.0, "lower": -1.0, "upper": 1.0,
            "confidence": float(c * 100), "horizon": "short",
            "contributors": {"short_term_model": 1.0}, "features": {},
        } for i, c in enumerate(claims)]
        # Truth at 0 if hit (point=0 and 0 ∈ [-1, 1] always); shift to make miss
        obs_rows = [{
            "location_key": "0,0", "variable": "temp_c",
            "valid_time": datetime(2026, 1, 1, i // 24, tzinfo=timezone.utc),
            "value": 0.0 if h else 100.0,
        } for i, h in enumerate(hits)]
        store.save_predictions(pred_rows)
        store.save_observations(obs_rows)
        rep = cal.fit_from_store(store)

    # Brier should improve after calibration
    assert rep.post_brier <= rep.pre_brier + 1e-6, \
        f"calibrator should not increase Brier (pre={rep.pre_brier:.3f}, post={rep.post_brier:.3f})"

    # Mapping should be monotonic non-decreasing
    xs = np.linspace(20, 99, 50)
    ys = [cal.apply("temp_c", x) for x in xs]
    diffs = np.diff(ys)
    assert (diffs >= -1e-6).all(), \
        f"calibrator must be monotonic; got decreases at {np.where(diffs < -1e-6)[0]}"


def test_calibrator_apply_unfitted_is_identity():
    """Without a fitted map, apply() returns the input unchanged."""
    pytest.importorskip("sklearn")
    from src.feedback.calibrator import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    assert cal.apply("temp_c", 73.5) == 73.5
