"""Geographic context features with real coast distance + UHI."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict

import httpx

log = logging.getLogger(__name__)


_COAST_ANCHORS = (
    (44.65, -63.57), (40.71, -74.01), (35.78, -75.50), (28.54, -80.65),
    (25.77, -80.19), (29.96, -90.07), (27.50, -97.40), (32.72, -117.16),
    (34.05, -118.25), (37.77, -122.42), (47.61, -122.33), (60.13, -149.43),
    (49.29, -123.12),
    (10.66, -71.61), (-12.05, -77.05), (-23.55, -46.63), (-34.61, -58.40),
    (-33.45, -70.67), (-41.86, -73.85), (-54.81, -68.30), (-2.20, -80.90),
    (38.71, -9.14), (43.30, -2.00), (48.39, -4.49), (50.91, 1.85),
    (51.50, 0.00), (53.55, 9.99), (55.68, 12.57), (59.91, 10.75),
    (60.17, 24.94), (45.44, 12.32), (37.97, 23.73), (36.72, -4.42),
    (41.01, 28.97),
    (33.59, -7.62), (14.69, -17.44), (6.45, 3.40), (-4.05, 39.66),
    (-33.92, 18.42), (-25.97, 32.57), (15.59, 32.53), (30.04, 31.24),
    (35.69, -5.32),
    (24.47, 54.37), (29.37, 47.99), (25.20, 55.27), (12.78, 45.04),
    (19.08, 72.88), (13.08, 80.27), (22.57, 88.36), (6.93, 79.86),
    (16.04, 108.22),
    (1.35, 103.82), (3.14, 101.69), (14.60, 120.98), (22.32, 114.17),
    (31.23, 121.47), (39.04, 117.20), (35.68, 139.65), (43.06, 141.35),
    (37.57, 126.98), (10.82, 106.63),
    (-33.87, 151.21), (-37.81, 144.96), (-31.95, 115.86), (-12.46, 130.84),
    (-36.85, 174.76), (-41.29, 174.78),
    (64.15, -21.94), (78.22, 15.65), (-77.85, 166.67), (21.31, -157.86),
    (-22.27, 166.46), (-17.74, 168.32), (-4.62, 55.45), (4.18, 73.51),
)


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


_CITY_ANCHORS = (
    (40.71, -74.01, 3), (34.05, -118.25, 3), (41.88, -87.63, 2),
    (29.76, -95.37, 2), (33.45, -112.07, 2), (37.77, -122.42, 2),
    (47.61, -122.33, 1), (45.50, -73.57, 2), (43.65, -79.38, 2),
    (19.43, -99.13, 3), (-23.55, -46.63, 3), (-34.61, -58.40, 3),
    (-22.91, -43.17, 3), (4.71, -74.07, 2), (-12.05, -77.05, 3),
    (51.51, -0.13, 3), (48.86, 2.35, 3), (40.42, -3.70, 2),
    (52.52, 13.40, 2), (41.90, 12.50, 2), (45.46, 9.19, 2),
    (50.85, 4.35, 1), (52.37, 4.90, 1), (59.33, 18.07, 1),
    (55.75, 37.62, 3), (50.45, 30.52, 2), (41.01, 28.97, 3),
    (35.69, 51.39, 3), (24.71, 46.68, 2), (30.04, 31.24, 3),
    (-26.20, 28.05, 2), (6.52, 3.38, 3), (-1.29, 36.82, 2),
    (19.08, 72.88, 3), (28.61, 77.21, 3), (12.97, 77.59, 2),
    (13.08, 80.27, 2), (22.57, 88.36, 3), (23.81, 90.41, 3),
    (39.91, 116.41, 3), (31.23, 121.47, 3), (22.32, 114.17, 2),
    (35.68, 139.65, 3), (37.57, 126.98, 3), (1.35, 103.82, 2),
    (14.60, 120.98, 3), (10.82, 106.63, 3), (-6.20, 106.85, 3),
    (-33.87, 151.21, 2), (-37.81, 144.96, 2),
)


@dataclass(frozen=True)
class GeoFeatures:
    lat: float
    lon: float
    elevation_m: float
    coast_distance_km: float
    urban_heat_index: float
    terrain_ruggedness_m: float

    def as_dict(self):
        return asdict(self)


class GeographicAdapter:
    name = "geographic"
    ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

    def __init__(self, config=None):
        self.config = config or {}

    async def fetch(self, lat, lon):
        elevation = await self._elevation(lat, lon)
        terrain = await self._terrain_ruggedness(lat, lon)
        coast = self._coast_distance_km(lat, lon)
        uhi = self._urban_heat_proxy(lat, lon)
        return GeoFeatures(lat=lat, lon=lon, elevation_m=elevation,
                           coast_distance_km=coast, urban_heat_index=uhi,
                           terrain_ruggedness_m=terrain)

    async def _elevation(self, lat, lon):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(self.ELEVATION_URL,
                                     params={"locations": str(lat) + "," + str(lon)})
                r.raise_for_status()
                return float(r.json()["results"][0]["elevation"])
        except Exception as e:
            log.warning("elevation lookup failed: %s", e)
            return 0.0

    async def _terrain_ruggedness(self, lat, lon):
        import numpy as np
        offsets = 0.1
        coords = [(lat + dy * offsets, lon + dx * offsets)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        locs = "|".join(str(la) + "," + str(lo) for la, lo in coords)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(self.ELEVATION_URL, params={"locations": locs})
                r.raise_for_status()
                elevs = [p["elevation"] for p in r.json()["results"]]
                return float(np.std(elevs))
        except Exception:
            return 0.0

    @staticmethod
    def _coast_distance_km(lat, lon):
        return min(_haversine_km(lat, lon, alat, alon)
                   for alat, alon in _COAST_ANCHORS)

    @staticmethod
    def _urban_heat_proxy(lat, lon):
        best = 0.0
        for clat, clon, pop_class in _CITY_ANCHORS:
            d_km = _haversine_km(lat, lon, clat, clon)
            radius = (60.0, 120.0, 200.0, 350.0)[pop_class]
            if d_km > radius:
                continue
            class_weight = (0.4, 0.6, 0.8, 1.0)[pop_class]
            score = class_weight * (1.0 - d_km / radius)
            best = max(best, score)
        return float(best)