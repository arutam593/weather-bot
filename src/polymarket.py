"""
Polymarket weather-market analyzer.

Fetches active markets, finds weather-related ones, and compares the
market-implied probability against this bot's forecast.

Statistics ONLY — no bet recommendations.
"""
from __future__ import annotations
import asyncio
import functools
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
import httpx
import pandas as pd
log = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
WEATHER_KEYWORDS = (
    "temperature", "hottest", "coldest", "heat wave", "heatwave",
    "hurricane", "tropical storm", "cyclone", "typhoon",
    "snow", "snowfall", "rainfall", "precipitation",
    "tornado", "blizzard", "wildfire", "drought",
    "warmest", "record high", "record low", "celsius",
    "fahrenheit", " °c", " °f",
)


@dataclass
class WeatherMarket:
    market_id: str
    question: str
    slug: str
    end_date: datetime
    yes_price: float            # implied probability of YES, 0..1
    no_price: float
    volume: float
    parsed: Optional["ParsedQuestion"] = None
    bot_probability: Optional[float] = None
    note: str = ""


@dataclass
class ParsedQuestion:
    """What we managed to extract from a market question."""
    location: Optional[str] = None
    variable: Optional[str] = None       # "temp_max", "temp_min", "precip_total"
    threshold: Optional[float] = None
    threshold_unit: Optional[str] = None  # "C" or "F"
    direction: Optional[str] = None       # "above" or "below"
    deadline: Optional[datetime] = None


# ───────────────────────────────────────────────────── fetch

async def fetch_active_markets(limit: int = 500) -> list[dict]:
    """Pull active, open markets from Polymarket Gamma API."""
    out: list[dict] = []
    offset = 0
    page_size = 100
    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(out) < limit:
            r = await client.get(GAMMA_URL, params={
                "active": "true", "closed": "false",
                "limit": page_size, "offset": offset,
                "order": "volume", "ascending": "false",
            })
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            out.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
    log.info("fetched %d active markets", len(out))
    return out[:limit]


def is_weather_market(market: dict) -> bool:
    q = (market.get("question") or "").lower()
    desc = (market.get("description") or "").lower()
    blob = q + " " + desc
    return any(k in blob for k in WEATHER_KEYWORDS)


# ───────────────────────────────────────────────── parsing

# Temperature threshold patterns:
#   "reach 100°F", "exceed 35 C", "above 90 degrees"
_TEMP_PATTERN = re.compile(
    r"(?:reach|exceed|above|hit|surpass|over|higher than|>)\s*"
    r"(\d{1,3}(?:\.\d+)?)\s*(?:°|degrees?|deg)?\s*(C|F|Celsius|Fahrenheit)?",
    re.IGNORECASE,
)

# Some major cities we can map for the bot. Extend freely.
_KNOWN_CITIES = {
    "new york": (40.71, -74.01),     "nyc": (40.71, -74.01),
    "manhattan": (40.71, -74.01),    "central park": (40.78, -73.97),
    "los angeles": (34.05, -118.25), "la": (34.05, -118.25),
    "chicago": (41.88, -87.63),      "miami": (25.77, -80.19),
    "houston": (29.76, -95.37),      "phoenix": (33.45, -112.07),
    "boston": (42.36, -71.06),       "dallas": (32.78, -96.80),
    "denver": (39.74, -104.99),      "seattle": (47.61, -122.33),
    "london": (51.51, -0.13),        "paris": (48.86, 2.35),
    "madrid": (40.42, -3.70),        "berlin": (52.52, 13.40),
    "rome": (41.90, 12.50),          "tokyo": (35.68, 139.65),
    "delhi": (28.61, 77.21),         "mumbai": (19.08, 72.88),
    "beijing": (39.91, 116.41),      "shanghai": (31.23, 121.47),
    "sydney": (-33.87, 151.21),      "hong kong": (22.32, 114.17),
}


@functools.lru_cache(maxsize=500)
def _geocode_lookup(query: str) -> Optional[tuple[float, float, str]]:
    """Look up a place name via Open-Meteo geocoding. Cached so we don't
    hammer the API for the same name twice."""
    if not query or len(query.strip()) < 3:
        return None
    try:
        r = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1, "language": "en"},
            timeout=10.0,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return None
        c = results[0]
        return (float(c["latitude"]), float(c["longitude"]), c["name"])
    except Exception as e:
        log.warning("geocode failed for %r: %s", query, e)
        return None


def _extract_candidate_locations(q: str) -> list[str]:
    """Pull capitalized multi-word phrases out of a question — these are
    the most likely place names. Filters out common non-place words."""
    # Strip the question down to "Will X reach Y" patterns and grab nouns.
    # Regex finds runs of capitalized words: "New York", "Hong Kong", "Cape Town".
    candidates = re.findall(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", q)
    # Drop obvious non-places
    stopwords = {
        "Will", "Yes", "No", "January", "February", "March", "April", "May",
        "June", "July", "August", "September", "October", "November", "December",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday", "Hurricane", "Storm", "Tropical", "Heat", "Wave",
    }
    return [c for c in candidates if c not in stopwords]


def parse_question(q: str) -> ParsedQuestion:
    """Heuristic parser for common weather market formats."""
    p = ParsedQuestion()
    ql = q.lower()

# Location: try the built-in list first (fast, no API call)
    for name, (lat, lon) in _KNOWN_CITIES.items():
        if name in ql:
            p.location = name
            break

    # Fallback: extract candidate place names and geocode them
    if not p.location:
        for candidate in _extract_candidate_locations(q):
            geo = _geocode_lookup(candidate)
            if geo:
                lat, lon, canonical = geo
                # Add to the runtime cache so we don't re-geocode it
                key = canonical.lower()
                _KNOWN_CITIES[key] = (lat, lon)
                p.location = key
                break

    # Temperature threshold
    m = _TEMP_PATTERN.search(q)
    if m:
        p.variable = "temp_max"
        p.threshold = float(m.group(1))
        unit_raw = (m.group(2) or "").upper()
        if unit_raw.startswith("C") or "celsius" in (m.group(2) or "").lower():
            p.threshold_unit = "C"
        elif unit_raw.startswith("F") or "fahrenheit" in (m.group(2) or "").lower():
            p.threshold_unit = "F"
        else:
            # Heuristic: thresholds 50-130 with no unit are probably °F (US-centric)
            p.threshold_unit = "F" if 50 <= p.threshold <= 130 else "C"
        p.direction = "above"

    # "below" / "under" inverts direction
    if re.search(r"\b(below|under|lower than|<)\b", ql) and p.threshold is not None:
        p.direction = "below"

    return p


def f_to_c(f: float) -> float:
    return (f - 32) * 5.0 / 9.0


# ──────────────────────────────────────────── bot probability

def probability_from_forecast(predictions: list, parsed: ParsedQuestion,
                              market_end: datetime) -> Optional[float]:
    """Estimate the bot's P(YES) for a parsed temperature-threshold question.

    Treats each hour's q10/q50/q90 as defining a triangular-ish distribution
    and asks: what fraction of forecast hours up to the deadline are
    expected to exceed the threshold?

    This is a rough approximation. CRPS-proper probabilistic resolution
    would integrate the predictive CDF; for a 3-quantile sparse forecast
    it's enough to fit a piecewise-linear CDF and read it off.
    """
    if (parsed.variable not in ("temp_max", "temp_min")
            or parsed.threshold is None):
        return None

    threshold_c = (f_to_c(parsed.threshold)
                   if parsed.threshold_unit == "F"
                   else parsed.threshold)

    # Filter to predictions before the market deadline + temp variable
    relevant = [p for p in predictions
                if p.variable == "temp_c" and p.valid_time <= market_end]
    if not relevant:
        return None

    # For each hour: probability that temperature >= threshold
    # using the 3-quantile piecewise-linear CDF approximation.
    def p_above(q10: float, q50: float, q90: float, x: float) -> float:
        # Treat q10..q90 as the 10th and 90th percentiles of a piecewise-linear CDF
        # linking (q10, 0.1), (q50, 0.5), (q90, 0.9).
        if x <= q10:                return 0.95   # cap, not 1.0 (sparse intervals)
        if x >= q90:                return 0.05
        if x <= q50:
            # interpolate between 0.1 and 0.5
            t = (x - q10) / max(q50 - q10, 1e-9)
            return 1.0 - (0.1 + 0.4 * t)
        t = (x - q50) / max(q90 - q50, 1e-9)
        return 1.0 - (0.5 + 0.4 * t)

    # For "max temperature exceeds X by deadline" — at least once → use the max
    # of per-hour probabilities (Bonferroni-ish lower bound; in reality the
    # probabilities are correlated across nearby hours so this is okay).
    if parsed.variable == "temp_max":
        per_hour = [p_above(p.lower, p.point, p.upper, threshold_c)
                    for p in relevant]
        # P(at least one hour exceeds) under independence = 1 - prod(1 - p_i).
        # But hourly temp is highly autocorrelated, so independence vastly
        # overstates it. As a practical compromise, use the maximum hourly p
        # plus a small autocorrelation adjustment.
        prob_at_least_one = 1.0 - 1.0
        for ph in per_hour:
            prob_at_least_one = 1.0 - (1.0 - prob_at_least_one) * (1.0 - ph * 0.4)
        prob = max(max(per_hour), prob_at_least_one)
        if parsed.direction == "below":
            prob = 1.0 - prob
        return float(prob)

    return None


# ───────────────────────────────────────── summary table

@dataclass
class MarketStat:
    question: str
    location: Optional[str]
    deadline: str
    market_yes_pct: float
    bot_yes_pct: Optional[float]
    gap_pct: Optional[float]
    volume_usd: float
    note: str

    def as_row(self) -> dict:
        return {
            "question": self.question[:80],
            "location": self.location or "—",
            "deadline": self.deadline,
            "market %": "{:.0f}".format(self.market_yes_pct * 100),
            "bot %": ("{:.0f}".format(self.bot_yes_pct * 100)
                      if self.bot_yes_pct is not None else "—"),
            "gap (bot - market)": ("{:+.0f}pp".format(self.gap_pct * 100)
                                   if self.gap_pct is not None else "—"),
            "volume $": "{:,.0f}".format(self.volume_usd),
            "note": self.note,
        }


async def analyze() -> pd.DataFrame:
    """Top-level: fetch, filter, parse, score, return a DataFrame."""
    raw = await fetch_active_markets(limit=500)
    weather = [m for m in raw if is_weather_market(m)]
    log.info("found %d weather markets out of %d active", len(weather), len(raw))

    if not weather:
        return pd.DataFrame()

    # Lazy import to avoid circulars
    from src.orchestrator import Orchestrator
    orch = Orchestrator("config/config.yaml")

    rows: list[MarketStat] = []
    for m in weather:
        try:
            outcome_prices = m.get("outcomePrices") or "[]"
            if isinstance(outcome_prices, str):
                import json
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices
            yes_price = float(prices[0]) if prices else 0.5
        except Exception:
            yes_price = 0.5

        end_str = m.get("endDate") or ""
        try:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except Exception:
            end_dt = datetime.now(timezone.utc) + timedelta(days=30)

        parsed = parse_question(m.get("question", ""))
        bot_p = None
        note = ""

        if parsed.location and parsed.threshold is not None:
            try:
                lat, lon = _KNOWN_CITIES[parsed.location]
                result = await orch.run_cycle(
                    lat, lon, parsed.location,
                    compute_explanations=False,
                )
                bot_p = probability_from_forecast(
                    result.predictions, parsed, end_dt)
                if bot_p is None:
                    note = "could not compute prob"
            except Exception as e:
                note = f"forecast failed: {e}"
        else:
            if not parsed.location:
                note = "location not recognized"
            elif parsed.threshold is None:
                note = "could not parse threshold"

        gap = (bot_p - yes_price) if bot_p is not None else None

        rows.append(MarketStat(
            question=m.get("question", "")[:200],
            location=parsed.location,
            deadline=end_dt.strftime("%Y-%m-%d"),
            market_yes_pct=yes_price,
            bot_yes_pct=bot_p,
            gap_pct=gap,
            volume_usd=float(m.get("volume") or 0),
            note=note,
        ))

    df = pd.DataFrame([r.as_row() for r in rows])
    # Sort by absolute gap (markets where bot disagrees most go first),
    # but put unparseable ones last.
    df["_gap_sort"] = [abs(r.gap_pct) if r.gap_pct is not None else -1
                       for r in rows]
    df = df.sort_values("_gap_sort", ascending=False).drop(columns=["_gap_sort"])
    return df


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = asyncio.run(analyze())
    if df.empty:
        print("\nNo active weather markets found on Polymarket right now.")
        print("Weather markets typically appear during hurricane season,")
        print("heatwaves, and around major weather events.")
        return
    print("\n══════════ POLYMARKET WEATHER MARKETS ══════════\n")
    print(df.to_string(index=False))
    print("\nNote: 'gap' is the bot's probability minus the market's implied")
    print("probability. Positive = bot thinks YES is undervalued by the market.")
    print("\nThis is statistics only. No bet recommendations.")


if __name__ == "__main__":
    main()