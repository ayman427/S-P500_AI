"""Server-only, date-scoped Finnhub news. Credentials never enter URLs or records."""
from __future__ import annotations

import pandas as pd
import requests

from .data import _normalize_ticker


class FinnhubNewsError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def fetch_company_news(ticker, event_date, window_days, api_key):
    date = pd.Timestamp(event_date)
    headlines = fetch_company_news_range(ticker, (date - pd.Timedelta(days=window_days)).date().isoformat(),
                                        (date + pd.Timedelta(days=window_days)).date().isoformat(), api_key)
    from .risk_news import match_event_news
    return match_event_news(headlines, event_date, window_days)


def fetch_company_news_range(ticker, from_date, to_date, api_key):
    ticker = _normalize_ticker(ticker)
    # Finnhub's range is day-based. Expand to cover the entire New York day
    # across UTC midnight, then enforce the requested local dates below.
    start, last = pd.Timestamp(from_date), pd.Timestamp(to_date)
    if start.tzinfo or last.tzinfo or start > last or last - start > pd.Timedelta(days=31):
        raise ValueError("Use an ordered news date range of at most 31 days.")
    end = last + pd.Timedelta(days=1)
    try:
        response = requests.get("https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": start.date().isoformat(), "to": end.date().isoformat()},
            headers={"X-Finnhub-Token": api_key, "Accept": "application/json"}, timeout=15)
        errors = {
            401: ("unauthorized", "Finnhub rejected the API key. Check FINNHUB_API_KEY on the server."),
            403: ("forbidden", "Your Finnhub plan does not allow this request. Try a more recent event date."),
            429: ("rate_limited", "Finnhub's request limit was reached. Try again shortly."),
        }
        if response.status_code in errors:
            raise FinnhubNewsError(*errors[response.status_code])
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise FinnhubNewsError("unavailable", "Finnhub news is temporarily unavailable. Showing saved articles only.") from None
    if not isinstance(payload, list):
        # HTTP 200 is not proof of a successful provider operation.
        raise FinnhubNewsError("unavailable", "Finnhub returned an unexpected response. Showing saved articles only.")
    headlines = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        related = {value.strip().upper().replace(".", "-") for value in str(item.get("related", "")).split(",") if value.strip()}
        if related and ticker not in related:
            continue
        published = pd.to_datetime(item.get("datetime"), unit="s", utc=True, errors="coerce")
        if pd.isna(published):
            continue
        headlines.append({"published_at_utc": published.isoformat(), "title": item.get("headline", ""),
            "url": item.get("url", ""), "publisher": item.get("source", "Unknown"), "source": "Finnhub",
            "excerpt": item.get("summary", "")})
    from .risk_news import normalized_headlines
    return [row for row in normalized_headlines(headlines)
            if from_date <= pd.Timestamp(row["published_at_utc"]).tz_convert("America/New_York").date().isoformat() <= to_date]
