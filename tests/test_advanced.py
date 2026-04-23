"""
Tests for the new components: backtest harness, LSTM contract, and
production middleware (auth + rate limit + metrics).
"""
from __future__ import annotations

import asyncio
import importlib.util

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────── backtest

def _toy_predictor_factory():
    """A naive 'persistence' predictor that uses the last training value
    as the median forecast for every test row."""
    def persistence(X_train, y_train, X_test):
        last = float(y_train.iloc[-1])
        std = max(float(y_train.std()), 0.5)
        idx = X_test.index
        return pd.DataFrame({
            "q10": [last - 1.28 * std] * len(idx),
            "q50": [last] * len(idx),
            "q90": [last + 1.28 * std] * len(idx),
        }, index=idx)
    return persistence


def _toy_dataset(n=240):
    rng = np.random.default_rng(0)
    t = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    y = 15 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 1, n)
    return pd.DataFrame({"x": np.arange(n), "temp_c": y}, index=t)


def test_backtest_runs_and_produces_metrics():
    from src.evaluation.backtest import backtest
    df = _toy_dataset(240)
    rep = backtest(
        df=df, target="temp_c", predictor=_toy_predictor_factory(),
        feature_cols=["x"], model_name="persistence",
        initial_train_hours=72, test_hours=24, step_hours=24,
    )
    assert rep.model_name == "persistence"
    assert rep.variable == "temp_c"
    assert len(rep.folds) >= 5
    assert rep.pooled["n_predictions"] > 0
    # All metrics present
    for m in ("mae", "rmse", "coverage", "crps"):
        assert m in rep.pooled
        assert rep.pooled[m] >= 0


def test_backtest_compare_models_returns_verdict():
    from src.evaluation.backtest import backtest, compare_models
    df = _toy_dataset(240)

    # A "good" predictor — uses the seasonal mean of training
    def smart(X_train, y_train, X_test):
        mu = y_train.mean()
        std = max(float(y_train.std()), 0.5)
        idx = X_test.index
        # Add a small diurnal swing so it actually wins
        hod = np.array([t.hour for t in idx])
        signal = mu + 4 * np.sin(2 * np.pi * (hod - 6) / 24)
        return pd.DataFrame({
            "q10": signal - 1.28 * std,
            "q50": signal,
            "q90": signal + 1.28 * std,
        }, index=idx)

    rep_a = backtest(df=df, target="temp_c", predictor=_toy_predictor_factory(),
                     feature_cols=["x"], model_name="persistence",
                     initial_train_hours=72, test_hours=24, step_hours=24)
    rep_b = backtest(df=df, target="temp_c", predictor=smart,
                     feature_cols=["x"], model_name="smart",
                     initial_train_hours=72, test_hours=24, step_hours=24)
    cmp_ = compare_models(rep_a, rep_b)
    assert cmp_["model_a"] == "persistence"
    assert cmp_["model_b"] == "smart"
    assert cmp_["verdict"] in {"A significantly better",
                                "B significantly better",
                                "no significant difference"}


def test_backtest_rejects_bad_predictor_output():
    """Predictor must return q10/q50/q90 columns."""
    from src.evaluation.backtest import backtest
    def bad(X_train, y_train, X_test):
        return pd.DataFrame({"value": [0] * len(X_test)}, index=X_test.index)
    with pytest.raises(Exception):
        backtest(df=_toy_dataset(120), target="temp_c", predictor=bad,
                 feature_cols=["x"], model_name="bad",
                 initial_train_hours=48, test_hours=24, step_hours=24)


def test_crps_lower_is_better():
    """A perfect forecast (q10=q50=q90=truth) has zero CRPS."""
    from src.evaluation.backtest import crps_quantile
    truth = np.array([10.0, 11.0, 12.0])
    perfect = {0.1: truth.copy(), 0.5: truth.copy(), 0.9: truth.copy()}
    bad     = {0.1: truth - 5, 0.5: truth - 5, 0.9: truth - 5}
    assert crps_quantile(truth, perfect) == 0.0
    assert crps_quantile(truth, bad) > crps_quantile(truth, perfect)


# ─────────────────────────────────────────────────── LSTM

@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="PyTorch not installed")
def test_lstm_fit_predict_roundtrip(tmp_path):
    """Smoke test: fit → predict → save → load → predict gives identical results."""
    from src.models.lstm import LSTMModel
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        rng.normal(size=(n, 8)),
        columns=[f"f{i}" for i in range(8)],
        index=pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"))
    y = pd.DataFrame({"temp_c": rng.normal(size=n),
                      "wind_ms": rng.normal(size=n)}, index=X.index)
    m = LSTMModel(lookback=24, epochs=2, hidden=16)
    m.fit(X, y)
    p1 = m.predict(X, now=X.index[-1])
    assert len(p1) > 0

    path = tmp_path / "lstm.pkl"
    m.save(path)
    loaded = LSTMModel.load(path)
    p2 = loaded.predict(X, now=X.index[-1])
    assert len(p1) == len(p2)
    assert all(abs(a.q50 - b.q50) < 1e-5 for a, b in zip(p1, p2))


def test_lstm_module_imports_without_torch():
    """The module must be importable even when torch is missing."""
    from src.models import lstm
    # The class itself can be referenced; only construction raises.
    assert hasattr(lstm, "LSTMModel")


# ─────────────────────────────────────────────────── middleware

def test_apikey_registry_hashes_lookup():
    from src.api.middleware import APIKeyRegistry
    reg = APIKeyRegistry.__new__(APIKeyRegistry)   # bypass __init__/file
    reg.path = None
    reg._keys = {}
    raw = "secret-abc-123"
    h = APIKeyRegistry.hash_key(raw)
    from src.api.middleware import APIKey
    reg._keys[h] = APIKey(name="test", tier="paid", hashed=h)
    assert reg.lookup(raw) is not None
    assert reg.lookup("wrong") is None
    assert reg.lookup(None) is None


def test_token_bucket_refills_over_time():
    from src.api.middleware import _Bucket
    import time
    b = _Bucket(capacity=5, refill_per_sec=10.0, tokens=5)
    for _ in range(5):
        assert b.take() is True
    assert b.take() is False     # empty
    time.sleep(0.3)               # should refill ~3 tokens
    assert b.take() is True


def test_rate_limiter_two_window_enforcement():
    from src.api.middleware import RateLimiter, APIKey
    limiter = RateLimiter()
    key = APIKey(name="test", tier="free", hashed="hhh")
    # free tier: 10/min, 500/day. Drain the minute bucket.
    for i in range(10):
        ok, _ = limiter.consume(key)
        assert ok, f"failed at request {i+1}/10"
    ok, window = limiter.consume(key)
    assert not ok
    assert window == "minute"


def test_metrics_renders_prometheus_format():
    from src.api.middleware import Metrics
    m = Metrics()
    m.record("/predict", 200, 42.5)
    m.record("/predict", 200, 18.2)
    m.record("/predict", 500, 100.0)
    m.record_rate_limit("test-key", "minute")
    out = m.render()
    assert "weather_bot_requests_total" in out
    assert 'path="/predict"' in out
    assert 'status="200"' in out
    assert 'status="500"' in out
    assert "weather_bot_rate_limited_total" in out
    # Sanity: prom format uses # HELP and # TYPE comments
    assert out.count("# HELP") >= 3
    assert out.count("# TYPE") >= 3


def test_middleware_blocks_missing_api_key():
    """Hits a fake FastAPI app with the middleware. Requests without a key
    must get 401; /health must always pass."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.middleware import (APIKeyRegistry, RateLimiter, Metrics,
                                     ProductionMiddleware, APIKey)

    app = FastAPI()

    @app.get("/health")
    def health(): return {"ok": True}

    @app.get("/secret")
    def secret(): return {"data": 42}

    reg = APIKeyRegistry.__new__(APIKeyRegistry)
    reg.path = None
    reg._keys = {APIKeyRegistry.hash_key("good-key"):
                 APIKey(name="t", tier="free", hashed="x")}

    app.add_middleware(
        ProductionMiddleware, registry=reg,
        limiter=RateLimiter(), metrics=Metrics(),
    )
    client = TestClient(app)
    # /health is always public
    assert client.get("/health").status_code == 200
    # /secret with no key → 401
    assert client.get("/secret").status_code == 401
    # /secret with bad key → 401
    assert client.get("/secret",
                      headers={"X-API-Key": "wrong"}).status_code == 401
    # /secret with good key → 200 and rate-limit headers present
    r = client.get("/secret", headers={"X-API-Key": "good-key"})
    assert r.status_code == 200
    assert "X-RateLimit-Remaining-Minute" in r.headers
    assert "X-Response-Time-ms" in r.headers
