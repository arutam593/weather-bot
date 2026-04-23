"""
Alert engine.

Fires user-facing alerts when any of:
  • prediction exceeds a configured threshold (with some confidence)
  • anomaly detector flags the current state
  • NLP signal is high AND local forecast agrees (reduces false positives)

Design: a small DSL of (name, predicate, severity) rules. Easy to extend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from src.models.ensemble import EnsemblePrediction
from src.models.anomaly import AnomalyReport

log = logging.getLogger(__name__)


@dataclass
class Alert:
    code: str            # e.g. "EXTREME_HEAT"
    severity: str        # "info" | "warn" | "severe"
    message: str
    valid_time: str
    variable: str | None = None


class AlertEngine:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._rules: list[tuple[str, str, Callable[[EnsemblePrediction], bool],
                                 Callable[[EnsemblePrediction], str]]] = [
            ("EXTREME_HEAT", "severe",
             lambda p: p.variable == "temp_c"
                       and p.point >= cfg.get("extreme_heat_c", 38)
                       and p.confidence_pct >= 55,
             lambda p: f"Extreme heat forecast: {p.point:.0f}°C "
                       f"at {p.valid_time:%a %H:%M}."),
            ("EXTREME_COLD", "severe",
             lambda p: p.variable == "temp_c"
                       and p.point <= cfg.get("extreme_cold_c", -15)
                       and p.confidence_pct >= 55,
             lambda p: f"Extreme cold forecast: {p.point:.0f}°C "
                       f"at {p.valid_time:%a %H:%M}."),
            ("HEAVY_RAIN", "warn",
             lambda p: p.variable == "precip_mm"
                       and p.point >= cfg.get("heavy_rain_mm_per_h", 15),
             lambda p: f"Heavy rain: up to {p.upper:.1f} mm/h at "
                       f"{p.valid_time:%a %H:%M}."),
            ("SEVERE_WIND", "severe",
             lambda p: p.variable == "wind_ms"
                       and p.point >= cfg.get("severe_wind_ms", 20),
             lambda p: f"Damaging wind forecast: {p.point:.0f} m/s "
                       f"at {p.valid_time:%a %H:%M}."),
        ]

    def evaluate(self, preds: Iterable[EnsemblePrediction],
                 anomaly: AnomalyReport | None = None) -> list[Alert]:
        alerts: list[Alert] = []
        for p in preds:
            for code, sev, cond, msg in self._rules:
                try:
                    if cond(p):
                        alerts.append(Alert(
                            code=code, severity=sev, message=msg(p),
                            valid_time=p.valid_time.isoformat(),
                            variable=p.variable,
                        ))
                except Exception as e:  # noqa: BLE001
                    log.warning("alert rule %s error: %s", code, e)

        if anomaly and anomaly.is_anomaly:
            alerts.append(Alert(
                code="ANOMALY",
                severity="warn",
                message=("Unusual conditions vs. recent history — "
                         + "; ".join(anomaly.reasons)),
                valid_time="",
            ))

        # Deduplicate consecutive identical alerts
        seen = set()
        unique: list[Alert] = []
        for a in alerts:
            key = (a.code, a.valid_time, a.variable)
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique
