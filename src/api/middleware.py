"""
Production middleware for the FastAPI service.

Three layers, each independently toggleable via config:

  • **API key auth** — every request must carry an `X-API-Key` header
    that matches one of the keys in `config/api_keys.yaml`. Keys are
    associated with a `tier` (e.g. free / paid / internal) which the
    rate limiter uses to decide quotas. Keys themselves are stored
    hashed (SHA-256) — the file never holds plaintext.

  • **Token-bucket rate limiter** — per-key, in-memory. Two buckets per
    key (per-minute and per-day) so a burst doesn't immediately drain
    the daily quota. Switch the storage to Redis to share state across
    multiple API instances; the algorithm is identical.

  • **Prometheus metrics** — request count + latency histogram + by-key
    quota usage, exposed at `/metrics`. Designed to be scraped by a
    Prometheus server in the same way every other production HTTP
    service is monitored.

`/health` and `/metrics` are public — every other endpoint is gated.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)

PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}


# ─────────────────────────────────────────────────── API keys

@dataclass
class APIKey:
    name: str        # human label, e.g. "internal-dashboard"
    tier: str        # "free" | "paid" | "internal"
    hashed: str      # SHA-256 of the raw key


class APIKeyRegistry:
    def __init__(self, path: str | Path = "config/api_keys.yaml"):
        self.path = Path(path)
        self._keys: dict[str, APIKey] = {}        # hashed → APIKey
        self._reload()

    def _reload(self):
        if not self.path.exists():
            log.warning("no api_keys.yaml at %s — auth will reject all requests",
                        self.path)
            return
        data = yaml.safe_load(self.path.read_text()) or {}
        for entry in data.get("keys", []):
            # Either the file holds raw keys (dev) or pre-hashed (prod).
            if "key_hash" in entry:
                h = entry["key_hash"]
            elif "key" in entry:
                h = self.hash_key(entry["key"])
            else:
                continue
            self._keys[h] = APIKey(name=entry["name"], tier=entry["tier"],
                                    hashed=h)
        log.info("loaded %d API keys", len(self._keys))

    @staticmethod
    def hash_key(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()

    def lookup(self, raw_key: str | None) -> APIKey | None:
        if not raw_key:
            return None
        return self._keys.get(self.hash_key(raw_key))


# ─────────────────────────────────────────────────── rate limiting

@dataclass
class _Bucket:
    capacity: int
    refill_per_sec: float
    tokens: float
    last_refill: float = field(default_factory=time.time)

    def take(self, n: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


# (per-minute capacity, per-day capacity) per tier
_TIER_QUOTAS: dict[str, tuple[int, int]] = {
    "free":     (10,    500),
    "paid":     (100,  20_000),
    "internal": (1_000, 1_000_000),
}


class RateLimiter:
    def __init__(self):
        # buckets[hashed_key] = (minute_bucket, day_bucket)
        self._buckets: dict[str, tuple[_Bucket, _Bucket]] = {}

    def _get_buckets(self, api_key: APIKey) -> tuple[_Bucket, _Bucket]:
        if api_key.hashed in self._buckets:
            return self._buckets[api_key.hashed]
        per_min, per_day = _TIER_QUOTAS.get(api_key.tier, _TIER_QUOTAS["free"])
        minute_b = _Bucket(capacity=per_min,
                           refill_per_sec=per_min / 60,
                           tokens=per_min)
        day_b = _Bucket(capacity=per_day,
                        refill_per_sec=per_day / 86400,
                        tokens=per_day)
        self._buckets[api_key.hashed] = (minute_b, day_b)
        return minute_b, day_b

    def consume(self, api_key: APIKey) -> tuple[bool, str | None]:
        minute_b, day_b = self._get_buckets(api_key)
        if not minute_b.take():
            return False, "minute"
        if not day_b.take():
            # We took from minute; refund it.
            minute_b.tokens = min(minute_b.capacity, minute_b.tokens + 1)
            return False, "day"
        return True, None

    def remaining(self, api_key: APIKey) -> dict[str, int]:
        minute_b, day_b = self._get_buckets(api_key)
        return {"per_minute_remaining": int(minute_b.tokens),
                "per_day_remaining":    int(day_b.tokens)}


# ─────────────────────────────────────────────────── Prometheus metrics

class Metrics:
    """In-process Prometheus-compatible text exposition.

    Avoids the prometheus_client dependency; the format is simple enough
    to emit by hand. For multi-process deployments use prometheus_client
    with the multiprocess directory pattern instead.
    """

    def __init__(self):
        self.req_total: dict[tuple[str, int], int] = defaultdict(int)
        self.latency_sum_ms: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        # Histogram buckets in ms
        self.buckets = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
        self.latency_hist: dict[tuple[str, int], int] = defaultdict(int)
        # Rate-limit rejections
        self.rate_limited: dict[tuple[str, str], int] = defaultdict(int)

    def record(self, path: str, status: int, latency_ms: float):
        self.req_total[(path, status)] += 1
        self.latency_sum_ms[path] += latency_ms
        self.latency_count[path] += 1
        for bucket in self.buckets:
            if latency_ms <= bucket:
                self.latency_hist[(path, bucket)] += 1

    def record_rate_limit(self, key_name: str, window: str):
        self.rate_limited[(key_name, window)] += 1

    def render(self) -> str:
        lines: list[str] = []
        lines.append("# HELP weather_bot_requests_total Total HTTP requests")
        lines.append("# TYPE weather_bot_requests_total counter")
        for (path, status), n in self.req_total.items():
            lines.append(
                f'weather_bot_requests_total{{path="{path}",status="{status}"}} {n}')
        lines.append("# HELP weather_bot_request_duration_ms Request latency")
        lines.append("# TYPE weather_bot_request_duration_ms histogram")
        for path, total in self.latency_sum_ms.items():
            count = self.latency_count[path]
            lines.append(
                f'weather_bot_request_duration_ms_sum{{path="{path}"}} {total:.1f}')
            lines.append(
                f'weather_bot_request_duration_ms_count{{path="{path}"}} {count}')
            for bucket in self.buckets:
                v = self.latency_hist.get((path, bucket), 0)
                lines.append(
                    f'weather_bot_request_duration_ms_bucket'
                    f'{{path="{path}",le="{bucket}"}} {v}')
        lines.append("# HELP weather_bot_rate_limited_total Rate-limit rejections")
        lines.append("# TYPE weather_bot_rate_limited_total counter")
        for (name, window), n in self.rate_limited.items():
            lines.append(
                f'weather_bot_rate_limited_total{{key="{name}",window="{window}"}} {n}')
        return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────── middleware

class ProductionMiddleware(BaseHTTPMiddleware):
    """Combined auth + rate-limit + metrics middleware.

    Order of checks:
      1. /health and /metrics → bypass auth+limit; still recorded.
      2. Look up X-API-Key. Missing or bad → 401.
      3. Take a token from the user's bucket. Empty → 429.
      4. Run the handler, record metrics on the way out.
    """

    def __init__(self, app, *, registry: APIKeyRegistry,
                 limiter: RateLimiter, metrics: Metrics):
        super().__init__(app)
        self.registry = registry
        self.limiter = limiter
        self.metrics = metrics

    async def dispatch(self, request: Request,
                       call_next: Callable) -> Response:
        path = request.url.path
        t0 = time.perf_counter()

        if path in PUBLIC_PATHS:
            response = await call_next(request)
            self.metrics.record(path, response.status_code,
                                (time.perf_counter() - t0) * 1000)
            return response

        # 1. Auth
        api_key = self.registry.lookup(request.headers.get("X-API-Key"))
        if api_key is None:
            self.metrics.record(path, 401, (time.perf_counter() - t0) * 1000)
            return Response(content='{"detail":"missing or invalid X-API-Key"}',
                            media_type="application/json", status_code=401)

        # 2. Rate limit
        ok, window = self.limiter.consume(api_key)
        if not ok:
            self.metrics.record(path, 429, (time.perf_counter() - t0) * 1000)
            self.metrics.record_rate_limit(api_key.name, window or "unknown")
            return Response(
                content=f'{{"detail":"rate limit exceeded ({window})"}}',
                media_type="application/json", status_code=429,
                headers={"Retry-After": "60" if window == "minute" else "3600"})

        # 3. Run handler
        request.state.api_key = api_key
        response = await call_next(request)

        # 4. Metrics + rate-limit headers
        latency_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record(path, response.status_code, latency_ms)
        rem = self.limiter.remaining(api_key)
        response.headers["X-RateLimit-Remaining-Minute"] = str(rem["per_minute_remaining"])
        response.headers["X-RateLimit-Remaining-Day"]    = str(rem["per_day_remaining"])
        response.headers["X-Response-Time-ms"] = f"{latency_ms:.1f}"
        return response
