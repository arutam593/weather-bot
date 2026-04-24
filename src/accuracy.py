"""
Accuracy tracking for the bot's Polymarket weather predictions.

Each day, we:
  1. Snapshot current predictions for every parseable market
  2. Look up yesterday's (and older) snapshots — if their target date has
     passed, fetch what the weather actually did from Open-Meteo's archive
  3. Score: was the bot right? By how much? Compute Brier score.
  4. Store the result so the dashboard can show trends

Persistence: SQLite in the system temp dir. On Streamlit Cloud free tier
the file will be wiped when the app restarts (~12h of idle), so early on
we'll have partial data. That's OK — we note it in the UI.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from sqlalchemy import (create_engine, Column, String, Float, DateTime,
                          Integer, Boolean, select)
from sqlalchemy.orm import sessionmaker, declarative_base

log = logging.getLogger(__name__)
Base = declarative_base()


def _db_path() -> str:
    return os.path.join(tempfile.gettempdir(), "weather_accuracy.db")


def _engine():
    return create_engine("sqlite:///" + _db_path(), future=True)


class Snapshot(Base):
    __tablename__ = "snapshots"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    snap_time       = Column(DateTime, index=True)
    market_id       = Column(String, index=True)
    question        = Column(String)
    city            = Column(String, index=True)
    lat             = Column(Float)
    lon             = Column(Float)
    variable        = Column(String)            # temp_max_exact, etc.
    threshold       = Column(Float)
    threshold_unit  = Column(String)
    bot_prob_yes    = Column(Float)             # 0..1
    market_prob_yes = Column(Float)
    target_date     = Column(DateTime)          # market deadline
    resolved        = Column(Boolean, default=False)


class Resolution(Base):
    __tablename__ = "resolutions"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    market_id     = Column(String, index=True)
    question      = Column(String)
    city          = Column(String)
    variable      = Column(String)
    threshold     = Column(Float)
    bot_prob_yes  = Column(Float)               # bot's prediction at snapshot time
    actual_value  = Column(Float)               # e.g. actual daily high °C
    resolved_yes  = Column(Boolean)             # did the YES outcome happen?
    brier_score   = Column(Float)               # (bot_prob - outcome)^2
    correct       = Column(Boolean)             # binary: bot_prob≥.5 matches outcome
    resolved_at   = Column(DateTime)


def init_db() -> None:
    Base.metadata.create_all(_engine())


# ──────────────────────────────────── snapshot current predictions

def record_snapshot(market_id: str, question: str, city: str,
                    lat: float, lon: float, variable: str,
                    threshold: float, threshold_unit: str,
                    bot_prob_yes: float, market_prob_yes: float,
                    target_date: datetime) -> None:
    """Save one snapshot. De-dupes: if we already snapshotted this market
    today, update it instead of adding a new row."""
    init_db()
    Session = sessionmaker(bind=_engine())
    with Session() as s:
        today = datetime.now(timezone.utc).date()
        # Find existing snapshot for this market today
        existing = s.execute(select(Snapshot).where(
            Snapshot.market_id == market_id
        )).scalars().all()
        today_snap = None
        for snap in existing:
            if snap.snap_time.date() == today:
                today_snap = snap
                break
        if today_snap:
            today_snap.bot_prob_yes = bot_prob_yes
            today_snap.market_prob_yes = market_prob_yes
        else:
            s.add(Snapshot(
                snap_time=datetime.now(timezone.utc),
                market_id=market_id, question=question, city=city,
                lat=lat, lon=lon, variable=variable,
                threshold=threshold, threshold_unit=threshold_unit,
                bot_prob_yes=bot_prob_yes, market_prob_yes=market_prob_yes,
                target_date=target_date, resolved=False,
            ))
        s.commit()


# ────────────────────────────────────── resolve past snapshots

def _f_to_c(f: float) -> float:
    return (f - 32) * 5.0 / 9.0


async def _fetch_actual_high(lat: float, lon: float,
                              target_date: datetime) -> Optional[float]:
    """Fetch the actual daily max temp °C for target_date from Open-Meteo."""
    d = target_date.date().isoformat()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": lat, "longitude": lon,
                        "start_date": d, "end_date": d,
                        "daily": "temperature_2m_max", "timezone": "UTC"})
            r.raise_for_status()
            data = r.json()
            highs = data.get("daily", {}).get("temperature_2m_max", [])
            if highs and highs[0] is not None:
                return float(highs[0])
    except Exception as e:
        log.warning("archive fetch failed for %.3f,%.3f on %s: %s",
                    lat, lon, d, e)
    return None


async def _fetch_actual_low(lat: float, lon: float,
                              target_date: datetime) -> Optional[float]:
    d = target_date.date().isoformat()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": lat, "longitude": lon,
                        "start_date": d, "end_date": d,
                        "daily": "temperature_2m_min", "timezone": "UTC"})
            r.raise_for_status()
            data = r.json()
            lows = data.get("daily", {}).get("temperature_2m_min", [])
            if lows and lows[0] is not None:
                return float(lows[0])
    except Exception as e:
        log.warning("archive fetch failed for %.3f,%.3f on %s: %s",
                    lat, lon, d, e)
    return None


async def resolve_due_snapshots() -> int:
    """Find all unresolved snapshots whose target_date is at least 2 days
    old (so Open-Meteo archive has the data), fetch actual weather, score.
    Returns number of new resolutions created."""
    init_db()
    Session = sessionmaker(bind=_engine())
    now = datetime.now(timezone.utc)
    count = 0

    with Session() as s:
        # Get the earliest snapshot per market_id that hasn't been resolved
        # (we score against the FIRST prediction, not last — that's a real test)
        unresolved = s.execute(select(Snapshot).where(
            Snapshot.resolved == False,
            Snapshot.target_date < now - timedelta(days=2),
        )).scalars().all()

        # Group by market_id, keep earliest snap per market
        earliest = {}
        for snap in unresolved:
            if snap.market_id not in earliest:
                earliest[snap.market_id] = snap
            elif snap.snap_time < earliest[snap.market_id].snap_time:
                earliest[snap.market_id] = snap

        for market_id, snap in earliest.items():
            # Skip if already have a resolution for this market
            existing_res = s.execute(select(Resolution).where(
                Resolution.market_id == market_id
            )).first()
            if existing_res:
                # Mark all snapshots for this market resolved and skip
                for other in unresolved:
                    if other.market_id == market_id:
                        other.resolved = True
                continue

            # Fetch ground truth
            if snap.variable == "temp_max_exact":
                actual_c = await _fetch_actual_high(snap.lat, snap.lon,
                                                     snap.target_date)
            elif snap.variable == "temp_min_exact":
                actual_c = await _fetch_actual_low(snap.lat, snap.lon,
                                                     snap.target_date)
            else:
                continue

            if actual_c is None:
                continue

            # Determine YES/NO
            # Market asks: will the daily high be exactly N degrees?
            # Conventional Polymarket resolution: high falls in [N, N+1)
            threshold_c = (_f_to_c(snap.threshold)
                           if snap.threshold_unit == "F"
                           else snap.threshold)
            resolved_yes = (threshold_c <= actual_c < threshold_c + 1.0)

            outcome = 1.0 if resolved_yes else 0.0
            brier = (snap.bot_prob_yes - outcome) ** 2
            correct = (snap.bot_prob_yes >= 0.5) == resolved_yes

            s.add(Resolution(
                market_id=market_id, question=snap.question, city=snap.city,
                variable=snap.variable, threshold=snap.threshold,
                bot_prob_yes=snap.bot_prob_yes, actual_value=actual_c,
                resolved_yes=resolved_yes, brier_score=brier, correct=correct,
                resolved_at=now,
            ))
            # Mark all snapshots for this market as resolved
            for other in unresolved:
                if other.market_id == market_id:
                    other.resolved = True
            count += 1
            log.info("resolved %s: actual=%.1f°C, threshold=%.1f, yes=%s, "
                     "bot=%.2f, brier=%.3f",
                     snap.city, actual_c, threshold_c, resolved_yes,
                     snap.bot_prob_yes, brier)

        s.commit()
    return count


# ──────────────────────────────────────── read-side helpers

def get_stats() -> dict:
    """Return overall accuracy stats for the dashboard."""
    init_db()
    Session = sessionmaker(bind=_engine())
    with Session() as s:
        rows = s.execute(select(Resolution)).scalars().all()
        if not rows:
            return {"n_resolved": 0, "n_snapshots": _count_snapshots()}
        n = len(rows)
        brier_avg = sum(r.brier_score for r in rows) / n
        correct = sum(1 for r in rows if r.correct)
        # Calibration — among predictions ≥70%, how many happened?
        high_conf = [r for r in rows if r.bot_prob_yes >= 0.7]
        high_conf_hit = sum(1 for r in high_conf if r.resolved_yes)
        return {
            "n_resolved":      n,
            "n_snapshots":     _count_snapshots(),
            "brier_avg":       brier_avg,
            "binary_acc":      correct / n,
            "n_correct":       correct,
            "n_high_conf":     len(high_conf),
            "n_high_conf_hit": high_conf_hit,
        }


def _count_snapshots() -> int:
    Session = sessionmaker(bind=_engine())
    with Session() as s:
        return len(s.execute(select(Snapshot)).scalars().all())


def get_resolutions_df() -> pd.DataFrame:
    """All resolutions as a DataFrame for the dashboard."""
    init_db()
    Session = sessionmaker(bind=_engine())
    with Session() as s:
        rows = s.execute(select(Resolution)).scalars().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "city":         r.city,
            "question":     r.question,
            "target":       r.threshold,
            "actual":       r.actual_value,
            "bot_prob_yes": r.bot_prob_yes,
            "resolved_yes": r.resolved_yes,
            "correct":      r.correct,
            "brier":        r.brier_score,
            "resolved_at":  r.resolved_at,
        } for r in rows])


def get_daily_brier() -> pd.DataFrame:
    """Return (date, mean_brier, n) grouped by resolution day."""
    df = get_resolutions_df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["resolved_at"]).dt.date
    return (df.groupby("date")
              .agg(brier=("brier", "mean"), n=("brier", "size"))
              .reset_index())


def should_run_daily_job() -> bool:
    """Returns True if we haven't run snapshots+resolutions in the last 23h."""
    init_db()
    Session = sessionmaker(bind=_engine())
    with Session() as s:
        newest = s.execute(select(Snapshot).order_by(
            Snapshot.snap_time.desc())).first()
        if not newest:
            return True
        last = newest[0].snap_time
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last) > timedelta(hours=23)