"""
Historical weather & climate normals.

Uses Meteostat (free JSON API). Provides:
  • Recent daily obs (last 30 days) for feedback-loop ground truth
  • 30-year climate normals per day-of-year  → used for anomaly detection
    and as a seasonal baseline feature
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from .base import utc_now

log = logging.getLogger(__name__)


@dataclass
class ClimateNormal:
    """30-year average + std for a day-of-year at a location."""
    doy: int
    temp_mean_c: float
    temp_std_c: float
    precip_mean_mm: float


class HistoricalAdapter:
    name = "meteostat"
    # Meteostat has a free JSON endpoint via RapidAPI, but also a keyless
    # point API at https://dev.meteostat.net/ — we use the Python library
    # pattern here to keep the example self-contained. In production,
    # swap in the keyed endpoint or the `meteostat` PyPI package.
    BASE = "https://meteostat.p.rapidapi.com/point/daily"

    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.reliability_prior = float(config.get("default", 1.0))

    async def recent_daily(self, lat: float, lon: float,
                           days: int = 30) -> pd.DataFrame:
        """Daily obs for the last `days` — used as feedback ground truth."""
        end = date.today()
        start = end - timedelta(days=days)
        params = {"lat": lat, "lon": lon,
                  "start": start.isoformat(), "end": end.isoformat()}
        headers = {"X-RapidAPI-Key": self.api_key} if self.api_key else {}
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(self.BASE, params=params, headers=headers)
            if r.status_code != 200:
                log.warning("meteostat unavailable (%d). Returning empty frame.",
                            r.status_code)
                return pd.DataFrame()
            data = r.json().get("data", [])

        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.rename(columns={"tavg": "temp_c", "prcp": "precip_mm",
                                  "wspd": "wind_kmh"})

    def climate_normals(self, df_history: pd.DataFrame) -> dict[int, ClimateNormal]:
        """Aggregate many years of history into per-DOY normals."""
        if df_history.empty:
            return {}
        df = df_history.copy()
        df["doy"] = df["date"].dt.dayofyear
        agg = df.groupby("doy").agg(
            temp_mean_c=("temp_c", "mean"),
            temp_std_c=("temp_c", "std"),
            precip_mean_mm=("precip_mm", "mean"),
        ).fillna(0.0)
        return {
            int(doy): ClimateNormal(
                doy=int(doy),
                temp_mean_c=row.temp_mean_c,
                temp_std_c=row.temp_std_c or 3.0,       # guard against 0 std
                precip_mean_mm=row.precip_mean_mm,
            )
            for doy, row in agg.iterrows()
        }
