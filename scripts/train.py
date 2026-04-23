"""
Train the short-term (LGBM quantile), mid-term (Prophet), and anomaly
(IsolationForest) models from historical data.

Inputs:
  --hourly-csv    CSV with columns [valid_time, temp_c, precip_mm, wind_ms,
                  humidity_pct, pressure_hpa, cloud_pct]
  --lat, --lon    location of the series (used for geo features)
  --output-dir    where to write fitted models (default: models_store/)

Output: pickled models written to models_store/{short.pkl, mid.pkl, anomaly.pkl}
which the Orchestrator auto-loads at startup.

Usage:
  python scripts/train.py --hourly-csv data/madrid_hourly.csv \
      --lat 40.42 --lon -3.70 --output-dir models_store/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.base import SourceReading, utc_now
from src.ingestion.geographic import GeographicAdapter
from src.ingestion.historical import HistoricalAdapter
from src.processing.features import FeatureBuilder, TARGET_VARIABLES
from src.models.short_term import ShortTermModel
from src.models.mid_term import MidTermModel
from src.models.anomaly import AnomalyDetector

log = logging.getLogger("train")


def _hourly_df_to_readings(df: pd.DataFrame, lat: float,
                           lon: float, source: str = "history") -> list[SourceReading]:
    """Convert a wide hourly DataFrame back into per-hour SourceReading objects.

    This is the bridge between offline training data and the runtime
    feature builder, which expects the same SourceReading interface that
    live adapters produce.
    """
    out: list[SourceReading] = []
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


def train(hourly_csv: str, lat: float, lon: float,
          output_dir: str = "models_store") -> None:
    df = pd.read_csv(hourly_csv, parse_dates=["valid_time"])
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df = df.sort_values("valid_time").reset_index(drop=True)

    log.info("loaded %d hourly rows from %s", len(df), hourly_csv)

    # 1. Geographic features (one-shot for this lat/lon).
    geo = asyncio.run(GeographicAdapter().fetch(lat, lon))

    # 2. Climate normals from the data itself (quick + works offline).
    df["date"] = df["valid_time"]
    daily = df.groupby(df["valid_time"].dt.floor("D")).agg(
        temp_c=("temp_c", "mean"),
        precip_mm=("precip_mm", "sum"),
    ).reset_index().rename(columns={"valid_time": "date"})
    normals = HistoricalAdapter({}).climate_normals(daily.assign(
        date=pd.to_datetime(daily["date"], utc=True)))

    # 3. Build features from history. The "readings" come from a single
    #    pseudo-source (this is just training data); at inference time
    #    multiple sources contribute. Either way, the feature pipeline
    #    is identical.
    readings = _hourly_df_to_readings(df, lat, lon)
    builder = FeatureBuilder(geo=geo, normals=normals)
    nlp_signals = {  # static for training; replaced by live signals at inference
        "hurricane_signal": 0.0, "flood_signal": 0.0, "heatwave_signal": 0.0,
        "cold_front_signal": 0.0, "storm_general_signal": 0.0,
        "tornado_signal": 0.0, "blizzard_signal": 0.0, "drought_signal": 0.0,
        "alert_signal": 0.0, "signal_article_count": 0.0,
    }

    truth = df.set_index("valid_time")[
        [c for c in TARGET_VARIABLES if c in df.columns]]
    frame = builder.build_training(readings, truth, nlp_signals)

    # Drop early rows where lag features are NaN
    frame.X = frame.X.dropna()
    frame.y = frame.y.loc[frame.X.index]
    log.info("training matrix: %s, targets: %s",
             frame.X.shape, list(frame.y.columns))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 4. Short-term model (LightGBM quantile)
    short = ShortTermModel(quantiles=(0.1, 0.5, 0.9), horizon_hours=72)
    short.fit(frame.X, frame.y)
    short.save(out / "short.pkl")

    # 5. Mid-term model (Prophet); skip if not installed
    try:
        from src.models.mid_term import MidTermModel
        extra_regs = [c for c in ["consensus__pressure_hpa",
                                  "nlp__hurricane_signal"] if c in frame.X.columns]
        mid = MidTermModel(horizon_days=7, extra_regressors=extra_regs)
        mid.fit(frame.X, frame.y)
        with open(out / "mid.pkl", "wb") as f:
            pickle.dump(mid, f)
        log.info("mid-term model saved")
    except Exception as e:
        log.warning("skipping mid-term model: %s", e)

    # 6. Anomaly detector
    anom = AnomalyDetector(contamination=0.02)
    anom.fit_feature_detector(frame.X)
    with open(out / "anomaly.pkl", "wb") as f:
        pickle.dump(anom, f)

    # 7. Save climate std (used by ensemble for confidence normalization)
    climate_std = {v: float(frame.y[v].std()) for v in frame.y.columns}
    with open(out / "climate_std.pkl", "wb") as f:
        pickle.dump(climate_std, f)

    log.info("training done. artifacts in %s", out.resolve())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hourly-csv", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--output-dir", default="models_store")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args.hourly_csv, args.lat, args.lon, args.output_dir)


if __name__ == "__main__":
    main()
