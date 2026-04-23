"""
Weather forecast API adapters.

Open-Meteo is the primary source — free, keyless, and exposes multiple NWP
models (ECMWF, GFS, ICON). OpenWeatherMap is an optional second source for
ensembling / cross-validation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import SourceAdapter, SourceReading, utc_now

log = logging.getLogger(__name__)


class OpenMeteoAdapter(SourceAdapter):
    """Open-Meteo: free, keyless, multi-model."""

    name = "open_meteo"
    BASE = "https://api.open-meteo.com/v1/forecast"

    async def fetch(
        self, lat: float, lon: float, *, horizon_hours: int = 72
    ) -> list[SourceReading]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m", "pressure_msl",
                "cloud_cover", "cape",
            ]),
            "forecast_days": max(1, (horizon_hours + 23) // 24),
            "timezone": "UTC",
            # Stacked ensemble of NWPs — Open-Meteo returns the blend:
            "models": "best_match",
        }
        async with httpx.AsyncClient() as client:
            data = await self._get(client, self.BASE, params=params)

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        readings: list[SourceReading] = []
        now = utc_now()

        for i, t in enumerate(times):
            valid = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            lead = (valid - now).total_seconds() / 3600.0
            if lead < -1 or lead > horizon_hours:
                continue
            readings.append(SourceReading(
                source=self.name,
                fetched_at=now,
                valid_time=valid,
                lead_hours=max(0.0, lead),
                lat=lat, lon=lon,
                variables={
                    "temp_c":        hourly["temperature_2m"][i],
                    "humidity_pct":  hourly["relative_humidity_2m"][i],
                    "precip_mm":     hourly["precipitation"][i],
                    "wind_ms":       hourly["wind_speed_10m"][i] / 3.6,  # km/h → m/s
                    "wind_dir_deg":  hourly["wind_direction_10m"][i],
                    "pressure_hpa":  hourly["pressure_msl"][i],
                    "cloud_pct":     hourly["cloud_cover"][i],
                    "cape_j_kg":     hourly.get("cape", [0] * len(times))[i] or 0.0,
                },
                reliability_prior=self.reliability_prior,
            ))
        log.info("open_meteo: %d readings for (%.3f, %.3f)", len(readings), lat, lon)
        return readings


class OpenWeatherMapAdapter(SourceAdapter):
    """OpenWeatherMap One Call API. Requires API key."""

    name = "openweathermap"
    BASE = "https://api.openweathermap.org/data/3.0/onecall"

    async def fetch(
        self, lat: float, lon: float, *, horizon_hours: int = 72
    ) -> list[SourceReading]:
        key = self.config.get("api_key", "")
        if not key:
            log.warning("openweathermap: no API key configured; skipping")
            return []

        params = {"lat": lat, "lon": lon, "appid": key, "units": "metric",
                  "exclude": "minutely,daily,alerts"}
        async with httpx.AsyncClient() as client:
            data = await self._get(client, self.BASE, params=params)

        readings: list[SourceReading] = []
        now = utc_now()
        for item in data.get("hourly", [])[:horizon_hours]:
            valid = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            lead = (valid - now).total_seconds() / 3600.0
            readings.append(SourceReading(
                source=self.name,
                fetched_at=now,
                valid_time=valid,
                lead_hours=max(0.0, lead),
                lat=lat, lon=lon,
                variables={
                    "temp_c":       item["temp"],
                    "humidity_pct": item["humidity"],
                    "precip_mm":    item.get("rain", {}).get("1h", 0.0)
                                    + item.get("snow", {}).get("1h", 0.0),
                    "wind_ms":      item["wind_speed"],
                    "wind_dir_deg": item["wind_deg"],
                    "pressure_hpa": item["pressure"],
                    "cloud_pct":    item["clouds"],
                },
                reliability_prior=self.reliability_prior,
            ))
        return readings
