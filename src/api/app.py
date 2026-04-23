"""
FastAPI service.

Endpoints:
  GET  /predict?lat=..&lon=..&name=..   → current prediction + explanation
  POST /feedback/observations           → record actual observations
  POST /feedback/evaluate               → trigger evaluator manually
  GET  /health                          → liveness (public)
  GET  /metrics                         → Prometheus exposition (public)

Auth: every request except /health and /metrics requires an X-API-Key
header. See src/api/middleware.py for details.
"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from src.orchestrator import Orchestrator
from src.feedback.evaluator import FeedbackEvaluator
from src.api.middleware import (APIKeyRegistry, ProductionMiddleware,
                                 RateLimiter, Metrics)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Weather Prediction Bot", version="0.5.0")
orch = Orchestrator("config/config.yaml")
evaluator = FeedbackEvaluator(orch.store, orch.ensemble,
                              lag_hours=orch.cfg["feedback"]["evaluation_lag_hours"],
                              anomaly=orch.anomaly,
                              climate_std=orch.ensemble.climate_std)

# Production middleware: API key auth, rate limiting, metrics.
_registry = APIKeyRegistry("config/api_keys.yaml")
_limiter = RateLimiter()
_metrics = Metrics()
app.add_middleware(ProductionMiddleware,
                   registry=_registry, limiter=_limiter, metrics=_metrics)


class ObservationIn(BaseModel):
    lat: float
    lon: float
    variable: str = Field(..., pattern="^(temp_c|precip_mm|wind_ms)$")
    valid_time: datetime
    value: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=_metrics.render(),
                    media_type="text/plain; version=0.0.4")


@app.get("/predict")
async def predict(lat: float, lon: float, name: str = "",
                  explain: bool = True, max_explanations: int = 8):
    try:
        result = await orch.run_cycle(
            lat, lon, location_name=name,
            compute_explanations=explain,
            max_explanations=max_explanations,
        )
    except Exception as e:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "predictions": [
            {
                "variable": p.variable,
                "valid_time": p.valid_time.isoformat(),
                "lead_hours": round(p.lead_hours, 1),
                "point": round(p.point, 2),
                "interval": [round(p.lower, 2), round(p.upper, 2)],
                "confidence_pct": round(p.confidence_pct, 1),
                "horizon": p.horizon,
                "contributors": {k: round(v, 3) for k, v in p.contributors.items()},
            }
            for p in result.predictions
        ],
        "explanations": result.explanations[:10],
        "alerts": result.alerts,
        "anomaly": result.anomaly,
        "geo": result.geo,
    }


@app.post("/feedback/observations")
def add_observations(obs: list[ObservationIn]):
    rows = [{"location_key": f"{round(o.lat, 2)},{round(o.lon, 2)}",
             "variable": o.variable,
             "valid_time": o.valid_time,
             "value": o.value} for o in obs]
    orch.store.save_observations(rows)
    return {"saved": len(rows)}


@app.post("/feedback/evaluate")
def evaluate():
    summary = evaluator.run()
    return {
        "n_scored": summary.n_scored,
        "per_var_mae": summary.per_var_mae,
        "per_var_coverage": summary.per_var_coverage,
    }
