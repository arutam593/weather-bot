"""
Base abstractions for ingestion adapters.

Every adapter returns `SourceReading` objects with a common schema so the
downstream pipeline is source-agnostic. Adapters are async-first so that
ingestion can be fan-out parallel.
"""
from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass
class SourceReading:
    """Normalized reading from any data source."""
    source: str                              # e.g. "open_meteo"
    fetched_at: datetime                     # UTC
    valid_time: datetime                     # the time this reading describes
    lead_hours: float                        # 0 for obs, >0 for forecast
    lat: float
    lon: float
    variables: dict[str, float] = field(default_factory=dict)
    # e.g. {"temp_c": 21.3, "precip_mm": 0.0, "wind_ms": 5.1, "pressure_hpa": 1012}
    reliability_prior: float = 1.0           # set from config; updated by feedback
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_forecast(self) -> bool:
        return self.lead_hours > 0


class SourceAdapter(abc.ABC):
    """All adapters share retry, timeout, and lifecycle."""

    name: str = "base"
    timeout_s: float = 15.0
    max_retries: int = 3

    def __init__(self, config: dict):
        self.config = config
        self.reliability_prior = float(config.get("default", 1.0))

    @abc.abstractmethod
    async def fetch(
        self, lat: float, lon: float, *, horizon_hours: int = 72
    ) -> list[SourceReading]:
        ...

    async def _get(self, client: httpx.AsyncClient, url: str, **kwargs) -> dict:
        """GET with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                r = await client.get(url, timeout=self.timeout_s, **kwargs)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError) as e:
                wait = 2 ** attempt
                log.warning(
                    "%s fetch failed (attempt %d/%d): %s — retrying in %ds",
                    self.name, attempt + 1, self.max_retries, e, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"{self.name}: exhausted retries for {url}")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
