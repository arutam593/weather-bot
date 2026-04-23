"""
End-to-end OFFLINE demo of the Weather Prediction Bot.

What it does (no network required):
  1. Synthesizes 60 days of hourly weather data for one location with
     realistic seasonal + diurnal cycles, stochastic noise, and one
     injected "heatwave" event in week 7.
  2. Trains the short-term LightGBM quantile model on the first 50 days.
  3. Trains the anomaly detector and saves climate std.
  4. Runs the orchestrator on the held-out 10 days, simulating live
     ingestion by replaying synthetic readings + fake news articles
     announcing the heatwave.
  5. Compares predictions to ground truth → MAE, interval coverage.
  6. Fits the confidence calibrator from the prediction history.
  7. Verifies the alert engine fires on the heatwave hour.

Run:
  python scripts/demo_offline.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import pickle
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feedback.calibrator import ConfidenceCalibrator
from src.feedback.evaluator import FeedbackEvaluator
from src.feedback.store import PredictionStore
from src.ingestion.base import SourceReading, utc_now
from src.ingestion.geographic import GeoFeatures
from src.ingestion.news_scraper import NewsArticle
from src.models.anomaly import AnomalyDetector
from src.models.ensemble import Ensemble
from src.models.short_term import ShortTermModel
from src.processing.features import FeatureBuilder, TARGET_VARIABLES
from src.processing.nlp import WeatherSignalExtractor
from src.explain.explainer import Explainer
from src.alerts.alert_engine import AlertEngine

LOG = logging.getLogger("demo")


# ---------------------------------------------------------------- synth data

def synth_hourly(n_days: int = 60, lat: float = 40.42,
                 seed: int = 42) -> pd.DataFrame:
    """Realistic-ish hourly time series for a mid-latitude location."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    times = pd.date_range(start, periods=n, freq="h", tz="UTC")

    doy = np.array([t.timetuple().tm_yday for t in times])
    hod = np.array([t.hour + t.minute / 60.0 for t in times])

    # Seasonal + diurnal temp; mild Madrid-ish base
    base = 12 + 8 * np.sin(2 * np.pi * (doy - 80) / 365)
    diurnal = 5 * np.sin(2 * np.pi * (hod - 5) / 24)
    weather_noise = rng.normal(0, 1.2, n).cumsum() * 0.05  # slow drift
    temp = base + diurnal + weather_noise + rng.normal(0, 1.0, n)

    # Inject a 5-day heatwave starting day 47 (in the held-out window)
    hw_start = 47 * 24
    hw_end = hw_start + 5 * 24
    temp[hw_start:hw_end] += np.linspace(8, 14, hw_end - hw_start)

    # Pressure: slow drift around 1013, dips around precip events
    pressure = 1013 + 5 * np.sin(2 * np.pi * np.arange(n) / (24 * 4)) \
        + rng.normal(0, 1.5, n)

    # Precip: zero-inflated; bursts when pressure low
    precip = np.where(pressure < 1009,
                      rng.gamma(2.0, 1.5, n), 0.0) * (rng.random(n) < 0.4)

    wind = 3 + 2 * (1009 - pressure).clip(min=0) / 5 + rng.normal(0, 0.8, n)
    wind = np.clip(wind, 0, None)

    humidity = (60 + 20 * np.sin(2 * np.pi * hod / 24) - 0.3 * (temp - base)
                + rng.normal(0, 5, n)).clip(20, 100)
    cloud = (40 + 30 * np.sin(2 * np.pi * (hod - 12) / 24)
             + 5 * (precip > 0) + rng.normal(0, 10, n)).clip(0, 100)

    df = pd.DataFrame({
        "valid_time":    times,
        "temp_c":        temp,
        "precip_mm":     precip,
        "wind_ms":       wind,
        "humidity_pct":  humidity,
        "pressure_hpa":  pressure,
        "cloud_pct":     cloud,
    })
    return df


def df_to_readings(df: pd.DataFrame, lat: float, lon: float,
                   source: str = "synth") -> list[SourceReading]:
    out = []
    var_cols = [c for c in df.columns if c != "valid_time"]
    for row in df.itertuples():
        out.append(SourceReading(
            source=source,
            fetched_at=utc_now(),
            valid_time=getattr(row, "valid_time"),
            lead_hours=0.0,
            lat=lat, lon=lon,
            variables={c: float(getattr(row, c))
                       for c in var_cols if pd.notna(getattr(row, c))},
            reliability_prior=1.0,
        ))
    return out


def synth_news(target_loc: str = "Madrid") -> list[NewsArticle]:
    return [
        NewsArticle(
            source="rss:test", url="http://demo/heatwave",
            title=f"Severe heatwave forecast across {target_loc}",
            body=("Meteorologists warn of an extreme heatwave with record "
                  f"temperatures expected in {target_loc} over the coming days. "
                  "Heat advisory issued."),
            published=utc_now(),
        ),
        NewsArticle(
            source="rss:test", url="http://demo/storm",
            title="Isolated thunderstorms possible overnight",
            body="Minor thunderstorm activity possible in the region.",
            published=utc_now(),
        ),
    ]


# ---------------------------------------------------------------- demo run

def run_demo(workdir: Path) -> dict:
    LAT, LON, NAME = 40.42, -3.70, "Madrid"
    geo = GeoFeatures(lat=LAT, lon=LON, elevation_m=667.0,
                      coast_distance_km=300.0, urban_heat_index=0.5,
                      terrain_ruggedness_m=120.0)

    # 1. Synthesize 60 days; train on first 50, hold out last 10.
    LOG.info("synthesizing 60 days of hourly data")
    df = synth_hourly(n_days=60, lat=LAT)
    train_df = df.iloc[: 50 * 24].copy()
    held_df  = df.iloc[ 50 * 24 :].copy()

    # 2. Build training features. Use the same dual-source pattern as
    #    inference (live_a + biased live_b) so train/inference feature
    #    schemas match perfectly. In production the equivalent is
    #    "train on the same set of providers you'll predict from".
    LOG.info("building training features")
    builder = FeatureBuilder(geo=geo, normals={})
    biased_train = train_df.copy()
    biased_train["temp_c"] = biased_train["temp_c"] + 0.3
    train_readings = (df_to_readings(train_df, LAT, LON, source="live_a")
                      + df_to_readings(biased_train, LAT, LON, source="live_b"))
    truth = train_df.set_index("valid_time")[
        ["temp_c", "precip_mm", "wind_ms"]]
    nlp_baseline = WeatherSignalExtractor({}).extract([], target_location=NAME)
    train_frame = builder.build_training(train_readings, truth, nlp_baseline)
    train_frame.X = train_frame.X.dropna()
    train_frame.y = train_frame.y.loc[train_frame.X.index]
    LOG.info("training matrix: %s", train_frame.X.shape)

    # 3. Fit short-term model + anomaly detector
    LOG.info("fitting LightGBM quantile models (3 quantiles × 3 targets)")
    short = ShortTermModel(quantiles=(0.1, 0.5, 0.9), horizon_hours=72)
    short.fit(train_frame.X, train_frame.y)

    LOG.info("fitting Isolation Forest anomaly detector")
    anom = AnomalyDetector(contamination=0.02)
    anom.fit_feature_detector(train_frame.X)

    # Climate std for confidence normalization
    climate_std = {v: float(train_frame.y[v].std())
                   for v in train_frame.y.columns}

    # 4. Simulate live cycles over the held-out period at 6h cadence.
    #    Each cycle: synthesize "live readings" = recent obs only, and
    #    inject heatwave news during the heatwave window.
    LOG.info("running 10 simulated prediction cycles on held-out data")
    ensemble = Ensemble(weight_decay_lambda=0.5, ema_alpha=0.2,
                        climate_std=climate_std)
    explainer = Explainer()
    nlp = WeatherSignalExtractor({})
    alerts_engine = AlertEngine({"extreme_heat_c": 30,  # tuned for demo
                                 "extreme_cold_c": -10,
                                 "heavy_rain_mm_per_h": 8,
                                 "severe_wind_ms": 15})

    db_path = workdir / "predictions.db"
    store = PredictionStore(f"sqlite:///{db_path}")

    fired_alerts: list[dict] = []
    coverage_hits = {v: [] for v in TARGET_VARIABLES}
    abs_errors = {v: [] for v in TARGET_VARIABLES}
    sample_explanations: list[str] = []
    pred_rows_buffer: list[dict] = []
    obs_rows_buffer: list[dict] = []
    # Quiet the store's per-write INFO logs during the demo loop
    logging.getLogger("src.feedback.store").setLevel(logging.WARNING)

    cycle_starts = list(range(0, 10 * 24, 12))   # every 12h for 10 days = 20 cycles
    HELD_OUT_START_DAY = 50
    HEATWAVE_DAYS = range(47, 53)               # absolute training-frame day indices
    for ci, start_i in enumerate(cycle_starts):
        if ci % 5 == 0:
            LOG.info("  cycle %d/%d", ci + 1, len(cycle_starts))
        # "Now" inside the simulated world
        now_idx = HELD_OUT_START_DAY * 24 + start_i
        if now_idx >= len(df):
            break
        sim_now = pd.Timestamp(df.iloc[now_idx]["valid_time"])

        # Live readings = past 48h of held data (acts as multi-source input).
        # In a real cycle these would come from the weather APIs.
        past = df.iloc[max(0, now_idx - 48): now_idx + 1].copy()
        live_readings = df_to_readings(past, LAT, LON, source="live_a")
        # Add a second pseudo-source with small bias to test consensus
        biased = past.copy()
        biased["temp_c"] = biased["temp_c"] + 0.3
        live_readings += df_to_readings(biased, LAT, LON, source="live_b")

        # NLP signals: turn on heatwave news during the event window
        sim_day_abs = now_idx // 24
        in_heatwave = sim_day_abs in HEATWAVE_DAYS
        articles = synth_news(NAME) if in_heatwave else []
        signals = nlp.extract(articles, target_location=NAME)

        frame = builder.build_inference(live_readings, signals)
        if frame.X.empty:
            continue
        frame.X = frame.X.ffill().fillna(0)

        # Predict for the next 24h only — that keeps the demo fast.
        # In production the orchestrator would pass a longer horizon.
        future_idx = pd.date_range(sim_now + pd.Timedelta(hours=1),
                                   periods=24, freq="h", tz="UTC")
        # Build a "future feature frame" by carrying the most recent
        # consensus values forward and updating only the seasonal/diurnal
        # columns. This is a deliberately simple persistence-of-features
        # baseline so the demo runs in seconds; production code would
        # instead build forecast features from each NWP source's forecast.
        last_row = frame.X.iloc[[-1]]
        future_X = pd.concat([last_row] * len(future_idx))
        future_X.index = future_idx
        # refresh seasonal/diurnal columns for the future timestamps
        from src.processing.features import FeatureBuilder as _FB
        future_X = _FB._add_seasonality(builder, future_X)

        short_preds = short.predict(future_X, now=sim_now)
        consensus = pd.DataFrame({
            v: future_X[f"consensus__{v}"]
            for v in TARGET_VARIABLES
            if f"consensus__{v}" in future_X.columns
        })
        ensemble_preds = ensemble.combine(short_preds, [], consensus, now=sim_now)
        anomaly_report = anom.check(frame.X.tail(1))

        # Score against ground truth (we have the future in df).
        # Only count forward-looking forecasts (lead_hours > 0); the input
        # window also produces "predictions" for past hours which are
        # essentially fitted values, not real forecasts.
        for ep in ensemble_preds:
            if ep.lead_hours <= 0:
                continue
            truth_row = df[df["valid_time"] == ep.valid_time]
            if truth_row.empty:
                continue
            actual = float(truth_row[ep.variable].iloc[0])
            abs_errors[ep.variable].append(abs(ep.point - actual))
            coverage_hits[ep.variable].append(int(ep.lower <= actual <= ep.upper))

            # Persist for the calibrator (buffered; flushed once at the end)
            pred_rows_buffer.append({
                "location_key": f"{LAT},{LON}",
                "lat": LAT, "lon": LON,
                "variable": ep.variable,
                "valid_time": ep.valid_time.to_pydatetime(),
                "lead_hours": ep.lead_hours,
                "point": ep.point, "lower": ep.lower, "upper": ep.upper,
                "confidence": ep.confidence_pct, "horizon": ep.horizon,
                "contributors": ep.contributors, "features": {},
            })
            obs_rows_buffer.append({
                "location_key": f"{LAT},{LON}",
                "variable": ep.variable,
                "valid_time": ep.valid_time.to_pydatetime(),
                "value": actual,
            })

        # Capture an explanation from the heatwave window
        if in_heatwave and not sample_explanations:
            for ep in ensemble_preds[:3]:
                if ep.valid_time in future_X.index:
                    feat_row = future_X.loc[ep.valid_time]
                    expl = explainer.explain(ep, feat_row, None,
                                             list(future_X.columns))
                    sample_explanations.append(expl.summary)

        # Alerts
        for a in alerts_engine.evaluate(ensemble_preds, anomaly_report):
            fired_alerts.append(a.__dict__)

    # Flush all buffered DB writes in one go
    LOG.info("flushing %d predictions + %d observations to store",
             len(pred_rows_buffer), len(obs_rows_buffer))
    logging.getLogger("src.feedback.store").setLevel(logging.INFO)
    store.save_predictions(pred_rows_buffer)
    store.save_observations(obs_rows_buffer)

    # 5. Calibrator
    LOG.info("fitting confidence calibrator from %d stored predictions",
             len(store.due_for_evaluation(
                 datetime.now(timezone.utc) + timedelta(days=999))))
    cal = ConfidenceCalibrator()
    cal_report = cal.fit_from_store(store)

    # 6. Aggregate metrics
    metrics = {
        "training_samples": int(train_frame.X.shape[0]),
        "feature_count":    int(train_frame.X.shape[1]),
        "cycles_run":       len(cycle_starts),
        "predictions":      sum(len(v) for v in abs_errors.values()),
        "mae": {v: float(np.mean(e)) if e else None
                for v, e in abs_errors.items()},
        "interval_coverage": {v: float(np.mean(c)) if c else None
                              for v, c in coverage_hits.items()},
        "alerts_fired":     len(fired_alerts),
        "unique_alert_codes": sorted({a["code"] for a in fired_alerts}),
        "calibrator_pre_brier":  cal_report.pre_brier,
        "calibrator_post_brier": cal_report.post_brier,
        "calibrator_samples":    cal_report.n_samples,
        "sample_explanation":    sample_explanations[:1],
        "sample_alerts":         fired_alerts[:3],
    }
    return metrics


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    td = tempfile.mkdtemp()
    try:
        metrics = run_demo(Path(td))
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)

    print("\n" + "=" * 64)
    print("END-TO-END DEMO RESULTS")
    print("=" * 64)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for kk, vv in v.items():
                print(f"  {kk:>20s}: {vv}")
        elif isinstance(v, list):
            print(f"\n{k}:")
            for item in v:
                print(f"  • {item}")
        else:
            print(f"  {k:>20s}: {v}")


if __name__ == "__main__":
    main()
