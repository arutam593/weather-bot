"""
Sanity tests that don't require external network or heavy ML dependencies.
Run: pytest -q tests/
"""
from datetime import datetime, timezone

import pandas as pd

from src.ingestion.base import SourceReading
from src.processing.features import FeatureBuilder
from src.processing.nlp import WeatherSignalExtractor
from src.ingestion.news_scraper import NewsArticle


def test_feature_builder_basic_shape():
    # Two sources, same time, same variable → consensus produced
    now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    readings = [
        SourceReading(source="a", fetched_at=now, valid_time=now, lead_hours=0,
                      lat=0, lon=0,
                      variables={"temp_c": 20.0, "pressure_hpa": 1013.0},
                      reliability_prior=1.0),
        SourceReading(source="b", fetched_at=now, valid_time=now, lead_hours=0,
                      lat=0, lon=0,
                      variables={"temp_c": 22.0, "pressure_hpa": 1012.0},
                      reliability_prior=0.5),
    ]
    builder = FeatureBuilder()
    frame = builder.build_inference(readings, nlp_signals={"flood_signal": 0.2})
    assert not frame.X.empty
    # Reliability-weighted consensus = (20*1 + 22*0.5)/(1+0.5) = 20.666...
    assert abs(frame.X["consensus__temp_c"].iloc[0] - 20.666666) < 0.01
    assert "nlp__flood_signal" in frame.X.columns
    assert "season_sin" in frame.X.columns


def test_nlp_extractor_regex_path():
    ext = WeatherSignalExtractor({"spacy_model": "this_model_does_not_exist"})
    articles = [
        NewsArticle(
            source="rss:test", url="http://x",
            title="Severe hurricane approaching Miami",
            body="A major hurricane is expected to make landfall in Miami.",
            published=datetime.now(timezone.utc),
        ),
        NewsArticle(
            source="rss:test", url="http://y",
            title="Minor thunderstorm advisory",
            body="Isolated thunderstorms possible overnight.",
            published=datetime.now(timezone.utc),
        ),
    ]
    sigs = ext.extract(articles, target_location="Miami")
    # Severe + major => intensifier multiplier > 1 → clamped at 1.0
    assert sigs["hurricane_signal"] >= 0.9
    # Minor advisory weakens storm_general below its base of 0.5
    assert 0 < sigs["storm_general_signal"] < 0.5
    assert sigs["signal_article_count"] == 2
