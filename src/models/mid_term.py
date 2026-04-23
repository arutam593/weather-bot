"""
Mid-term prediction (3-7 days). Optional Prophet implementation;
if Prophet isn't installed, this becomes a no-op stub so the rest
of the system still runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MidTermPrediction:
    variable: str
    valid_time: pd.Timestamp
    lead_hours: float
    yhat: float
    yhat_lower: float
    yhat_upper: float


class MidTermModel:
    """Stub mid-term model. Returns no predictions but keeps the
    interface alive so the orchestrator and ensemble work unchanged."""

    def __init__(self, *args, **kwargs):
        log.info("MidTermModel: stub mode (Prophet integration optional)")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "MidTermModel":
        return self

    def predict(self, X: pd.DataFrame, now=None) -> list[MidTermPrediction]:
        return []
