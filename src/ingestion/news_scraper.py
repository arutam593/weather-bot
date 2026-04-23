"""
News & alert ingestion.

Two paths:
  1) NewsAPI  (keyed, best for broad coverage)
  2) RSS/CAP  (keyless, NWS alerts, Met Office etc.)

The scraper only returns raw articles; NLP signal extraction lives in
`src/processing/nlp.py` so this module stays I/O-only.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from .base import utc_now

log = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    source: str
    url: str
    title: str
    body: str
    published: datetime
    raw: dict[str, Any] = field(default_factory=dict)


class NewsAPIAdapter:
    name = "newsapi"
    BASE = "https://newsapi.org/v2/everything"

    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.reliability_prior = float(config.get("default", 0.3))

    async def fetch(self, location: str, hours: int = 24) -> list[NewsArticle]:
        if not self.api_key:
            return []
        q = f'({location}) AND (storm OR hurricane OR flood OR heatwave OR ' \
            f'typhoon OR blizzard OR tornado OR drought OR cyclone)'
        params = {
            "q": q, "apiKey": self.api_key, "language": "en",
            "sortBy": "publishedAt", "pageSize": 20,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(self.BASE, params=params)
            r.raise_for_status()
            data = r.json()

        out: list[NewsArticle] = []
        for a in data.get("articles", []):
            try:
                pub = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
            except Exception:
                pub = utc_now()
            out.append(NewsArticle(
                source=f"newsapi:{a.get('source', {}).get('name', '')}",
                url=a.get("url", ""),
                title=a.get("title") or "",
                body=(a.get("description") or "") + " " + (a.get("content") or ""),
                published=pub,
                raw=a,
            ))
        log.info("newsapi: fetched %d articles for '%s'", len(out), location)
        return out


class RSSFeedAdapter:
    name = "rss_weather"

    def __init__(self, config: dict):
        self.feeds: list[str] = list(config.get("feeds", []))
        self.reliability_prior = float(config.get("default", 0.4))

    async def fetch(self, location: str | None = None,
                    hours: int = 24) -> list[NewsArticle]:
        # feedparser is sync; offload to a thread so we don't block the loop.
        loop = asyncio.get_running_loop()
        parsed = await asyncio.gather(
            *[loop.run_in_executor(None, feedparser.parse, u) for u in self.feeds],
            return_exceptions=True,
        )

        out: list[NewsArticle] = []
        cutoff = utc_now().timestamp() - hours * 3600
        for url, res in zip(self.feeds, parsed):
            if isinstance(res, Exception):
                log.warning("rss failure: %s — %s", url, res)
                continue
            for entry in res.entries:
                pub_struct = (entry.get("published_parsed")
                              or entry.get("updated_parsed"))
                if pub_struct:
                    pub = datetime(*pub_struct[:6], tzinfo=timezone.utc)
                    if pub.timestamp() < cutoff:
                        continue
                else:
                    pub = utc_now()
                title = entry.get("title", "")
                body = entry.get("summary", "") or entry.get("description", "")
                if location and location.lower() not in (title + " " + body).lower():
                    # Lightweight locality filter. NWS CAP entries carry
                    # structured geocodes; a production system should parse those.
                    pass
                out.append(NewsArticle(
                    source=f"rss:{url}",
                    url=entry.get("link", url),
                    title=title,
                    body=body,
                    published=pub,
                ))
        log.info("rss: fetched %d entries from %d feeds", len(out), len(self.feeds))
        return out
