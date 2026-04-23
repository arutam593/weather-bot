"""
NLP signal extraction.

Converts a list of NewsArticle → dict of weather-event signals in [0, 1].
Design choices:

  • Fast path (default): spaCy NER for locations + keyword severity scoring.
    Zero GPU, runs in milliseconds. Good enough for most signals.
  • Slow path (optional): zero-shot classification with a transformer
    (e.g. `facebook/bart-large-mnli`) for nuanced signals like
    "incoming cold front" vs "heat advisory". Gated behind a flag because
    it pulls ~1.5GB of weights.

Output example:
  {
    "hurricane_signal": 0.0,
    "flood_signal": 0.4,
    "heatwave_signal": 0.2,
    "storm_general_signal": 0.6,
    "signal_count": 3,          # how many articles contributed
  }
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Iterable

from src.ingestion.news_scraper import NewsArticle

log = logging.getLogger(__name__)

# Map keyword → (signal name, base severity)
_KEYWORD_MAP = {
    r"\bhurricane|cyclone|typhoon\b":    ("hurricane_signal", 1.0),
    r"\btornado|twister\b":              ("tornado_signal", 0.9),
    r"\bblizzard|ice storm|snowstorm\b": ("blizzard_signal", 0.8),
    r"\bflood(ing)?|flash flood\b":      ("flood_signal", 0.7),
    r"\bheatwave|heat wave|extreme heat\b": ("heatwave_signal", 0.7),
    r"\bcold (wave|front|snap)\b":       ("cold_front_signal", 0.6),
    r"\bdrought\b":                      ("drought_signal", 0.5),
    r"\bthunderstorm|severe storm\b":    ("storm_general_signal", 0.5),
    r"\badvisory|warning|watch\b":       ("alert_signal", 0.3),
}

# Modifiers bumping severity up or down
_INTENSIFIERS = [
    (r"\b(record|historic|unprecedented|catastrophic)\b", 1.3),
    (r"\b(severe|major|dangerous|extreme)\b",             1.2),
    (r"\b(minor|isolated|scattered|possible)\b",          0.7),
]


class WeatherSignalExtractor:

    def __init__(self, config: dict):
        self.config = config
        self._nlp = None

        # Lazy-load spaCy only if installed; fall back to regex-only.
        try:
            import spacy
            model = config.get("spacy_model", "en_core_web_sm")
            try:
                self._nlp = spacy.load(model)
            except OSError:
                log.warning("spaCy model '%s' not installed; using regex only."
                            " Run: python -m spacy download %s", model, model)
        except ImportError:
            log.info("spaCy not available; regex-only NLP path.")

    def extract(self, articles: Iterable[NewsArticle],
                target_location: str | None = None) -> dict[str, float]:
        """Aggregate signals across articles. Location-filter softly."""
        agg: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        n_articles = 0

        for art in articles:
            text = f"{art.title}. {art.body}".lower()
            if not text.strip():
                continue

            # Locality filter: if an article doesn't mention the target
            # location (and spaCy sees no matching GPE), penalize it.
            local_weight = self._locality_weight(text, target_location)

            n_articles += 1
            for pattern, (signal, base) in _KEYWORD_MAP.items():
                if re.search(pattern, text):
                    sev = base * self._intensifier(text) * local_weight
                    agg[signal] = max(agg[signal], min(1.0, sev))
                    counts[signal] += 1

        out = dict(agg)
        out["signal_article_count"] = float(n_articles)
        # Ensure the standard signals are always present (0 if absent) —
        # keeps the feature vector shape stable for the model.
        for _, (name, _) in _KEYWORD_MAP.items():
            out.setdefault(name, 0.0)
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _intensifier(text: str) -> float:
        mult = 1.0
        for pattern, m in _INTENSIFIERS:
            if re.search(pattern, text):
                mult *= m
        return mult

    def _locality_weight(self, text: str,
                         target_location: str | None) -> float:
        if not target_location:
            return 1.0
        if target_location.lower() in text:
            return 1.0
        if self._nlp is None:
            return 0.5   # can't verify, partial credit
        # spaCy: if GPEs exist and none match, strongly penalize.
        doc = self._nlp(text[:2000])  # cap for speed
        gpes = {ent.text.lower() for ent in doc.ents if ent.label_ in ("GPE", "LOC")}
        if not gpes:
            return 0.8
        if any(target_location.lower() in g or g in target_location.lower()
               for g in gpes):
            return 1.0
        return 0.3
