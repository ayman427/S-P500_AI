"""Date-matched company news with a shared Convex archive and honest coverage gaps."""
from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import pandas as pd
import requests

from .artifacts import exclusive_file
from .data import _normalize_ticker
from .paths import NEWS_DIR


def normalized_headlines(headlines, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    result = {}
    for item in headlines:
        published = pd.to_datetime(item.get("published_at_utc"), utc=True, errors="coerce")
        title, url = str(item.get("title", "")).strip(), str(item.get("url", "")).strip()
        parsed = urlparse(url)
        if pd.isna(published) or published > now or not title or parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
            continue
        row = {"published_at_utc": published.isoformat(), "title": title, "url": url,
               "publisher": str(item.get("publisher", "Unknown")),
               "source": item["source"] if item.get("source") in {"Finnhub", "API Ninjas"} else "Yahoo Finance feed"}
        if isinstance(item.get("excerpt"), str) and item["excerpt"].strip():
            row["excerpt"] = item["excerpt"].strip()[:12000]
        result[(url, published.isoformat())] = row
    return sorted(result.values(), key=lambda row: row["published_at_utc"], reverse=True)


def archive_headlines(ticker, headlines, directory=None):
    """Append/reconcile articles without overwriting previous retrievals."""
    ticker = _normalize_ticker(ticker)
    directory = NEWS_DIR if directory is None else directory
    path = directory / "archive" / f"{ticker}.json"
    with exclusive_file(directory / f".{ticker}.news-archive.lock"):
        existing = json.loads(path.read_text()).get("headlines", []) if path.exists() else []
        merged = normalized_headlines([*existing, *headlines])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ticker": ticker, "headlines": merged}))


def fetch_yahoo_headlines(ticker):
    from .news import _parse_yahoo_news
    response = requests.get("https://query1.finance.yahoo.com/v1/finance/search",
        params={"q": ticker, "newsCount": 100, "quotesCount": 0},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    response.raise_for_status()
    return _parse_yahoo_news(response.json(), ticker, limit=100)


def match_event_news(headlines, event_date, window_days=0, now=None):
    if window_days not in (0, 3):
        raise ValueError("News window must be the same day or within three days.")
    if not isinstance(event_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date):
        raise ValueError("Use a calendar event date in YYYY-MM-DD format.")
    date = pd.Timestamp(event_date)
    if date.tzinfo is not None or date != date.normalize():
        raise ValueError("Use a calendar event date in YYYY-MM-DD format.")
    start, end = date - pd.Timedelta(days=window_days), date + pd.Timedelta(days=window_days)
    matches = []
    for item in normalized_headlines(headlines, now):
        published = pd.Timestamp(item["published_at_utc"])
        local_day = published.tz_convert("America/New_York").tz_localize(None).normalize()
        if start <= local_day <= end:
            matches.append({**item, "publicationDate": local_day.date().isoformat(),
                            "sameDay": local_day == date, "daysFromEvent": int((local_day - date).days)})
    matches.sort(key=lambda row: (abs(row["daysFromEvent"]), row["published_at_utc"]))
    return matches


def read_event_news(ticker, event_date, window_days=0):
    ticker = _normalize_ticker(ticker)
    # Validate the requested date/window before cloud writes or provider requests.
    match_event_news([], event_date, window_days)
    if pd.Timestamp(event_date).date() > pd.Timestamp.now(tz="America/New_York").date():
        raise ValueError("News event date cannot be in the future.")
    from .prediction_store import load_environment
    load_environment()
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if api_key:
        return _read_finnhub_event_news(ticker, event_date, window_days, api_key)
    cache_path = NEWS_DIR / f"{ticker}.json"
    archive_path = NEWS_DIR / "archive" / f"{ticker}.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    headlines = list(cached.get("headlines", []))
    fetched = pd.to_datetime(cached.get("fetched_at_utc"), utc=True, errors="coerce")
    provider_status = "cached"
    if pd.isna(fetched) or pd.Timestamp.now(tz="UTC") - fetched >= pd.Timedelta(hours=6):
        try:
            fresh = fetch_yahoo_headlines(ticker)
        except (requests.RequestException, ValueError):
            provider_status = "unavailable"
        else:
            from .news import _save_news_cache
            # Archive the previous cache too, including data predating this feature.
            cached = _save_news_cache(ticker, fresh)
            headlines.extend(fresh)
            provider_status = "refreshed"
    if archive_path.exists():
        headlines.extend(json.loads(archive_path.read_text()).get("headlines", []))
    available = normalized_headlines(headlines)
    matched = match_event_news(available, event_date, window_days)
    from .news_analysis import rank_news
    matched = rank_news(matched)
    return {"ticker": ticker, "eventDate": event_date, "windowDays": window_days,
        "timezone": "America/New_York", "headlines": matched[:20], "matchCount": len(matched),
        "provider": "Yahoo Finance", "providerStatus": provider_status, "fetchedAt": cached.get("fetched_at_utc"),
        "archiveCount": len(available),
        "earliestArticle": min((row["published_at_utc"] for row in available), default=None),
        "message": "Headlines are matched by publication date, not verified causes of the decline." if matched else
            "No date-matched Yahoo headlines are available in our archive. Yahoo's free feed does not provide a historical date-range archive. This does not mean no news occurred.",
    }


def _read_finnhub_event_news(ticker, event_date, window_days, api_key):
    from .finnhub_news import FinnhubNewsError, fetch_company_news
    path = NEWS_DIR / "events" / "finnhub" / event_date / str(window_days) / f"{ticker}.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    fetched = pd.to_datetime(cached.get("fetched_at_utc"), utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    identity_valid = (cached.get("ticker") == ticker and cached.get("eventDate") == event_date
                      and cached.get("windowDays") == window_days)
    cache_valid = (identity_valid and cached.get("schemaVersion") == 2 and not pd.isna(fetched)
                   and pd.Timedelta(0) <= now - fetched < pd.Timedelta(hours=6))
    status, message = "cached", ""
    headlines = list(cached.get("headlines", [])) if identity_valid else []
    if not cache_valid:
        try:
            headlines = fetch_company_news(ticker, event_date, window_days, api_key)
        except FinnhubNewsError as error:
            status, message = error.status, str(error)
        else:
            # Archive first; a failed upload must not look like a successful save.
            archive_headlines(ticker, headlines)
            cached = {"ticker": ticker, "eventDate": event_date, "windowDays": window_days,
                "headlines": headlines, "fetched_at_utc": now.isoformat(), "schemaVersion": 2}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cached))
            status = "refreshed"
    archive_path = NEWS_DIR / "archive" / f"{ticker}.json"
    if archive_path.exists():
        headlines.extend(json.loads(archive_path.read_text()).get("headlines", []))
    legacy_path = NEWS_DIR / f"{ticker}.json"
    if legacy_path.exists():
        headlines.extend(json.loads(legacy_path.read_text()).get("headlines", []))
    available = normalized_headlines(headlines)
    matched = match_event_news(available, event_date, window_days)
    from .news_analysis import rank_news
    matched = rank_news(matched)
    return {"ticker": ticker, "eventDate": event_date, "windowDays": window_days,
        "timezone": "America/New_York", "provider": "Finnhub", "providerStatus": status,
        "providerMessage": message, "fetchedAt": cached.get("fetched_at_utc"),
        "headlines": matched[:20], "matchCount": len(matched), "archiveCount": len(available),
        "earliestArticle": min((row["published_at_utc"] for row in available), default=None),
        "message": "Headlines are date-matched context, not verified causes." if matched else
            "No date-matched headlines found. The date may be outside your plan's coverage; no news has been substituted.",
    }
