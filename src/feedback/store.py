"""
Persistence layer for the feedback loop.

We persist three things:
  • Predictions  — what we said and when
  • Observations — what actually happened (once observable)
  • Source skill — rolling error stats per (source, variable, lead_bucket)

SQLite is fine for single-node use. Swap the URL for Postgres to scale.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, Integer, String, JSON,
                        create_engine, Index)
from sqlalchemy.orm import DeclarativeBase, Session

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    __tablename__ = "predictions"
    id            = Column(Integer, primary_key=True)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)
    location_key  = Column(String, index=True)      # "lat,lon" rounded
    lat           = Column(Float)
    lon           = Column(Float)
    variable      = Column(String, index=True)      # temp_c / precip_mm / wind_ms
    valid_time    = Column(DateTime, index=True)
    lead_hours    = Column(Float)
    point         = Column(Float)
    lower         = Column(Float)
    upper         = Column(Float)
    confidence    = Column(Float)
    horizon       = Column(String)                  # "short" | "mid"
    contributors  = Column(JSON)                    # {expert: weight}
    features      = Column(JSON)                    # for post-hoc SHAP etc.


class ObservationRow(Base):
    __tablename__ = "observations"
    id            = Column(Integer, primary_key=True)
    location_key  = Column(String, index=True)
    variable      = Column(String, index=True)
    valid_time    = Column(DateTime, index=True)
    value         = Column(Float)


class SourceSkillRow(Base):
    __tablename__ = "source_skill"
    id            = Column(Integer, primary_key=True)
    source        = Column(String, index=True)
    variable      = Column(String, index=True)
    lead_bucket   = Column(String, index=True)      # "0-6h", "6-24h", "24-72h", ...
    ema_abs_error = Column(Float, default=1.0)
    n_samples     = Column(Integer, default=0)
    updated_at    = Column(DateTime, default=datetime.utcnow)


Index("idx_obs_loc_var_time",
      ObservationRow.location_key, ObservationRow.variable,
      ObservationRow.valid_time)


# --------------------------------------------------------------------------

class PredictionStore:

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self):
        s = Session(self.engine, future=True)
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ------------------------------------------------------------- writes

    def save_predictions(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.session() as s:
            s.add_all([PredictionRow(**r) for r in rows])
        log.info("feedback: saved %d predictions", len(rows))

    def save_observations(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.session() as s:
            s.add_all([ObservationRow(**r) for r in rows])
        log.info("feedback: saved %d observations", len(rows))

    # -------------------------------------------------------------- reads

    def due_for_evaluation(self, older_than: datetime) -> list[PredictionRow]:
        """Predictions whose valid_time has passed and are not yet scored."""
        with self.session() as s:
            # In a full implementation we'd LEFT JOIN observations and
            # filter out scored rows via a separate evaluation table.
            return s.query(PredictionRow).filter(
                PredictionRow.valid_time <= older_than
            ).all()

    def observation_at(self, location_key: str, variable: str,
                       valid_time: datetime) -> float | None:
        with self.session() as s:
            row = s.query(ObservationRow).filter(
                ObservationRow.location_key == location_key,
                ObservationRow.variable == variable,
                ObservationRow.valid_time == valid_time,
            ).first()
            return row.value if row else None
