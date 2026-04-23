"""
Continuous-operation scheduler.

Three jobs:

  • predict   — every 30min: run a fresh prediction cycle for each
                tracked location and persist results.
  • observe   — every 60min: ingest the most recent obs (lead=0 readings
                from the weather adapter) and store as ground truth.
  • evaluate  — every 60min, offset by 5min: run the feedback evaluator
                to score predictions whose valid_time is now in the past,
                update ensemble skill EMAs, and refit the confidence
                calibrator if enough new samples accumulated.

Locations are loaded from config/locations.yaml — a separate file so it
can be edited without touching the main config.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.feedback.calibrator import ConfidenceCalibrator
from src.feedback.evaluator import FeedbackEvaluator
from src.orchestrator import Orchestrator

log = logging.getLogger("scheduler")


class WeatherBotScheduler:

    def __init__(self, config_path: str = "config/config.yaml",
                 locations_path: str = "config/locations.yaml"):
        self.orch = Orchestrator(config_path)
        self.evaluator = FeedbackEvaluator(
            self.orch.store, self.orch.ensemble,
            lag_hours=self.orch.cfg["feedback"]["evaluation_lag_hours"],
        )
        self.locations = self._load_locations(locations_path)
        self.scheduler = AsyncIOScheduler()
        self._cycles_since_recalibration = 0
        self._recalibration_every_n_evals = 24   # ≈ once per day

    @staticmethod
    def _load_locations(path: str) -> list[dict]:
        if not Path(path).exists():
            log.warning("no locations file at %s — using a default", path)
            return [{"name": "Madrid", "lat": 40.42, "lon": -3.70}]
        with open(path) as f:
            return yaml.safe_load(f).get("locations", [])

    # -------------------------------------------------- jobs

    async def predict_job(self):
        for loc in self.locations:
            try:
                await self.orch.run_cycle(loc["lat"], loc["lon"], loc["name"])
            except Exception as e:
                log.exception("predict failed for %s: %s", loc["name"], e)

    async def observe_job(self):
        """Ingest current obs (lead=0 readings) and write to obs table."""
        for loc in self.locations:
            try:
                rows = []
                for adapter in self.orch.weather_adapters:
                    readings = await adapter.fetch(loc["lat"], loc["lon"],
                                                   horizon_hours=1)
                    for r in readings:
                        if r.lead_hours > 1:
                            continue
                        for var, val in r.variables.items():
                            if var in ("temp_c", "precip_mm", "wind_ms"):
                                rows.append({
                                    "location_key":
                                        f"{round(loc['lat'], 2)},"
                                        f"{round(loc['lon'], 2)}",
                                    "variable": var,
                                    "valid_time": r.valid_time,
                                    "value": val,
                                })
                if rows:
                    self.orch.store.save_observations(rows)
            except Exception as e:
                log.exception("observe failed for %s: %s", loc["name"], e)

    def evaluate_job(self):
        try:
            summary = self.evaluator.run()
            log.info("eval: %s", summary)
            self._cycles_since_recalibration += 1
            if self._cycles_since_recalibration >= self._recalibration_every_n_evals:
                self._recalibrate()
                self._cycles_since_recalibration = 0
        except Exception as e:
            log.exception("evaluate failed: %s", e)

    def _recalibrate(self):
        cal = ConfidenceCalibrator()
        rep = cal.fit_from_store(self.orch.store)
        if rep.n_samples >= 50:
            cal.save("models_store/calibrator.pkl")
            self.orch.calibrator = cal
            log.info("recalibrated confidence: pre-Brier=%.3f post-Brier=%.3f "
                     "(n=%d)", rep.pre_brier, rep.post_brier, rep.n_samples)
        else:
            log.info("recalibration skipped — only %d samples", rep.n_samples)

    # -------------------------------------------------- lifecycle

    def start(self):
        self.scheduler.add_job(
            self.predict_job, IntervalTrigger(minutes=30),
            id="predict", next_run_time=datetime.now(),
        )
        self.scheduler.add_job(
            self.observe_job, IntervalTrigger(minutes=60),
            id="observe", next_run_time=datetime.now(),
        )
        self.scheduler.add_job(
            self.evaluate_job, IntervalTrigger(minutes=60),
            id="evaluate",
        )
        self.scheduler.start()
        log.info("scheduler started: %d jobs, %d locations",
                 len(self.scheduler.get_jobs()), len(self.locations))


async def _amain():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sched = WeatherBotScheduler()
    sched.start()
    # keep the loop alive
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(_amain())
