"""
Orchestrator: runs one full prediction cycle.

Order of operations:
  1. Ingest — parallel fan-out to all enabled sources
  2. Process — build feature matrix, extract NLP signals
  3. Predict — short-term + mid-term + ensemble + anomaly check
  4. Explain — SHAP + rule-based → human text
  5. Alerts — apply rules over ensemble output
  6. Persist — save predictions to the feedback store

Training is intentionally separate (scripts/train.py, not shown) —
the orchestrator assumes models are already loaded from disk.
"""
from __future__ import annotations

import asyncio
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.ingestion.weather_api import OpenMeteoAdapter, OpenWeatherMapAdapter
from src.ingestion.news_scraper import NewsAPIAdapter, RSSFeedAdapter
from src.ingestion.geographic import GeographicAdapter, GeoFeatures
from src.ingestion.historical import HistoricalAdapter
from src.ingestion.satellite import SatelliteRadarAdapter

from src.processing.features import FeatureBuilder, TARGET_VARIABLES
from src.processing.nlp import WeatherSignalExtractor

from src.models.short_term import ShortTermModel
from src.models.mid_term import MidTermModel
from src.models.ensemble import Ensemble, EnsemblePrediction
from src.models.anomaly import AnomalyDetector

from src.explain.explainer import Explainer
from src.alerts.alert_engine import AlertEngine
from src.feedback.store import PredictionStore

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    predictions: list[EnsemblePrediction]
    explanations: list[str]
    alerts: list[dict]
    anomaly: dict
    geo: dict


class Orchestrator:

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Ingestion adapters (respect enabled flags)
        s_cfg = self.cfg["sources"]
        self.weather_adapters = [
            OpenMeteoAdapter(s_cfg["open_meteo"]),
        ]
        if s_cfg.get("openweathermap", {}).get("enabled"):
            self.weather_adapters.append(
                OpenWeatherMapAdapter(s_cfg["openweathermap"]))

        self.geo_adapter = GeographicAdapter()
        self.historical = HistoricalAdapter(s_cfg.get("meteostat", {}))
        self.satellite = SatelliteRadarAdapter(s_cfg.get("rainviewer", {}))

        news_feeds = []
        if s_cfg.get("rss_weather", {}).get("enabled"):
            news_feeds.append(RSSFeedAdapter(s_cfg["rss_weather"]))
        if s_cfg.get("newsapi", {}).get("enabled"):
            news_feeds.append(NewsAPIAdapter(s_cfg["newsapi"]))
        self.news_adapters = news_feeds

        # Processing
        self.nlp = WeatherSignalExtractor(self.cfg.get("nlp", {}))

        # Models — auto-load from disk if available, else fall back to
        # raw-consensus-only mode (still produces useful intervals).
        self.short_model: ShortTermModel | None = None
        self.mid_model: MidTermModel | None = None
        self.calibrator = None  # ConfidenceCalibrator | None
        self._load_artifacts(Path("models_store"))

        ens_kwargs = dict(self.cfg["models"]["ensemble"])
        # If we trained, the per-variable climate std went to disk too —
        # use it for sharper confidence normalization.
        std_path = Path("models_store/climate_std.pkl")
        if std_path.exists():
            with open(std_path, "rb") as f:
                ens_kwargs["climate_std"] = pickle.load(f)
        self.ensemble = Ensemble(**ens_kwargs)

        self.anomaly = AnomalyDetector(
            contamination=self.cfg["models"]["anomaly"]
                                  ["isolation_forest_contamination"],
            cusum_threshold=self.cfg["models"]["anomaly"]["cusum_threshold"],
        )
        anom_path = Path("models_store/anomaly.pkl")
        if anom_path.exists():
            with open(anom_path, "rb") as f:
                self.anomaly = pickle.load(f)
            log.info("loaded anomaly detector from disk")

        # Output stages
        self.explainer = Explainer()
        self.alert_engine = AlertEngine(self.cfg["alerts"])
        self.store = PredictionStore(self.cfg["feedback"]["database_url"])

    # ------------------------------------------------------------------

    def _load_artifacts(self, models_dir: Path) -> None:
        """Best-effort load of trained models + calibrator from disk."""
        short_p = models_dir / "short.pkl"
        if short_p.exists():
            try:
                self.short_model = ShortTermModel.load(short_p)
                log.info("loaded short-term model from %s", short_p)
            except Exception as e:
                log.warning("could not load short-term model: %s", e)

        mid_p = models_dir / "mid.pkl"
        if mid_p.exists():
            try:
                with open(mid_p, "rb") as f:
                    self.mid_model = pickle.load(f)
                log.info("loaded mid-term model from %s", mid_p)
            except Exception as e:
                log.warning("could not load mid-term model: %s", e)

        cal_p = models_dir / "calibrator.pkl"
        if cal_p.exists():
            try:
                from src.feedback.calibrator import ConfidenceCalibrator
                self.calibrator = ConfidenceCalibrator.load(cal_p)
                log.info("loaded confidence calibrator from %s", cal_p)
            except Exception as e:
                log.warning("could not load calibrator: %s", e)

    async def run_cycle(self, lat: float, lon: float,
                        location_name: str = "") -> CycleResult:
        log.info("=== cycle start: (%.3f, %.3f) ===", lat, lon)

        # 1. Ingest
        readings_lists, geo, history, radar, articles = await asyncio.gather(
            asyncio.gather(*[a.fetch(lat, lon) for a in self.weather_adapters]),
            self.geo_adapter.fetch(lat, lon),
            self.historical.recent_daily(lat, lon, days=30),
            self.satellite.fetch(lat, lon),
            asyncio.gather(*[a.fetch(location_name or "")
                             for a in self.news_adapters]),
        )
        readings = [r for sub in readings_lists for r in sub]
        news = [a for sub in articles for a in sub]

        log.info("ingest: %d readings, %d news items, %d radar frames, "
                 "%d history rows",
                 len(readings), len(news), len(radar), len(history))

        # 2. Process
        signals = self.nlp.extract(news, target_location=location_name)
        normals = self.historical.climate_normals(history)
        builder = FeatureBuilder(geo=geo, normals=normals)
        frame = builder.build_inference(readings, signals)

        # Raw consensus per variable (input to ensemble)
        consensus = pd.DataFrame({
            v: frame.X[f"consensus__{v}"]
            for v in TARGET_VARIABLES
            if f"consensus__{v}" in frame.X.columns
        })

        # 3. Predict
        short_preds = (self.short_model.predict(frame.X)
                       if self.short_model else [])
        mid_preds = (self.mid_model.predict(frame.X)
                     if self.mid_model else [])
        ensemble_preds = self.ensemble.combine(short_preds, mid_preds, consensus)

        # Post-hoc isotonic recalibration of confidence (if calibrator is
        # available — otherwise the raw sigmoid score is kept).
        if self.calibrator is not None:
            for ep in ensemble_preds:
                ep.confidence_pct = self.calibrator.apply(
                    ep.variable, ep.confidence_pct)

        anomaly_report = self.anomaly.check(frame.X.tail(1))

        # 4. Explain (batched SHAP per variable, then indexed back)
        explanations: list[str] = []
        shap_cache: dict[str, Any] = {}
        if self.short_model is not None:
            for var in TARGET_VARIABLES:
                if var in self.short_model.models:
                    try:
                        shap_cache[var] = self.short_model.shap_values(
                            frame.X, variable=var)
                    except Exception as e:
                        log.warning("shap failed for %s: %s", var, e)

        for ep in ensemble_preds:
            if ep.valid_time in frame.X.index:
                feat_row = frame.X.loc[ep.valid_time]
                shap_row = None
                if ep.variable in shap_cache:
                    row_pos = frame.X.index.get_loc(ep.valid_time)
                    shap_row = shap_cache[ep.variable][row_pos]
                expl = self.explainer.explain(
                    ep, feat_row, shap_row, frame.feature_names)
                explanations.append(expl.summary)

        # 5. Alerts
        alerts = self.alert_engine.evaluate(ensemble_preds, anomaly_report)

        # 6. Persist
        location_key = f"{round(lat, 2)},{round(lon, 2)}"
        self.store.save_predictions([
            {
                "location_key": location_key, "lat": lat, "lon": lon,
                "variable": p.variable, "valid_time": p.valid_time.to_pydatetime(),
                "lead_hours": p.lead_hours, "point": p.point,
                "lower": p.lower, "upper": p.upper,
                "confidence": p.confidence_pct, "horizon": p.horizon,
                "contributors": p.contributors,
                "features": {},   # populate if you want post-hoc debugging
            }
            for p in ensemble_preds
        ])

        log.info("=== cycle done: %d preds, %d alerts ===",
                 len(ensemble_preds), len(alerts))

        return CycleResult(
            predictions=ensemble_preds,
            explanations=explanations,
            alerts=[a.__dict__ for a in alerts],
            anomaly={
                "is_anomaly": anomaly_report.is_anomaly,
                "reasons": anomaly_report.reasons,
            },
            geo=geo.as_dict(),
        )
