"""
Satellite & radar adapter.

RainViewer aggregates global weather-radar mosaics and publishes a free,
keyless JSON index of recent (past 2h) and near-future (next 30min
nowcast) frames. We don't download the images themselves here — just the
timestamps and the location's intensity at each frame, which we use as:

  • short-term precipitation nowcasting feature
  • motion vector estimate (simple optical flow, out of scope here)

For production nowcasting, fetch the PNG tiles and run a ConvLSTM
or DGMR-style model. This module is a stub that returns a timeseries
of the radar reflectivity at the query point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .base import utc_now

log = logging.getLogger(__name__)


@dataclass
class RadarFrame:
    valid_time: datetime
    is_nowcast: bool
    reflectivity_dbz_estimate: float      # 0 if no echo; >30 = heavy rain


class SatelliteRadarAdapter:
    name = "rainviewer"
    INDEX_URL = "https://api.rainviewer.com/public/weather-maps.json"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.reliability_prior = float((config or {}).get("default", 1.0))

    async def fetch(self, lat: float, lon: float, *,
                    horizon_hours: int = 2) -> list[RadarFrame]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(self.INDEX_URL)
                r.raise_for_status()
                idx = r.json()
        except Exception as e:
            log.warning("rainviewer index failed: %s", e)
            return []

        frames: list[RadarFrame] = []
        for frame in idx.get("radar", {}).get("past", []):
            frames.append(RadarFrame(
                valid_time=datetime.fromtimestamp(frame["time"], tz=timezone.utc),
                is_nowcast=False,
                # Without downloading the tile we can't read the pixel value.
                # Production: fetch the 256x256 PNG tile that contains (lat,lon),
                # look up the pixel, translate the color index to dBZ.
                reflectivity_dbz_estimate=0.0,
            ))
        for frame in idx.get("radar", {}).get("nowcast", []):
            frames.append(RadarFrame(
                valid_time=datetime.fromtimestamp(frame["time"], tz=timezone.utc),
                is_nowcast=True,
                reflectivity_dbz_estimate=0.0,
            ))
        log.info("rainviewer: %d frames", len(frames))
        return frames
