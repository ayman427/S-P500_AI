"""Bounded historical-news coverage trial; results and evidence live in Convex."""
import json
import os
import re

import pandas as pd
import requests

from .data import _normalize_ticker
from .news_analysis import _company_aliases
from .paths import NEWS_DIR
from .prediction_store import load_environment
from .primary_evidence import research_window
from .risk_news import normalized_headlines
from .spike_evidence import load_evidence, save_evidence
from .spike_research import classify_sources, research_spike
from .spikes import read_spikes


class ApiNinjasError(Exception):
    pass


def fetch_news(ticker, start, end, api_key):
    ticker = _normalize_ticker(ticker)
    if not all(isinstance(day, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in (start, end)):
        raise ValueError("Use calendar dates in YYYY-MM-DD format.")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last or last - first > pd.Timedelta(days=31):
        raise ValueError("Use an ordered date range of at most 31 days.")
    rows = []
    # shortcut: cap at 100 headlines per event, increase only if trial coverage is truncated.
    for offset in (0, 50):
        try:
            response = requests.get("https://api.api-ninjas.com/v1/stocknews",
                params={"ticker": ticker, "start_date": start,
                        "end_date": (last + pd.Timedelta(days=1)).date().isoformat(), "offset": offset},
                headers={"X-Api-Key": api_key, "Accept": "application/json"}, timeout=15)
            if response.status_code in (400, 401, 403, 429):
                raise ApiNinjasError({400: "API Ninjas rejected the request; check Stock News subscription access (premium required).",
                    401: "API Ninjas rejected the key.",
                    403: "API Ninjas denied access; Stock News requires a paid plan.",
                    429: "API Ninjas request quota reached; trial stopped."}[response.status_code])
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or len(payload) > 50 or any(not isinstance(item, dict) for item in payload):
                raise ValueError("Unexpected provider response")
        except (requests.RequestException, ValueError):
            raise ApiNinjasError("API Ninjas request failed; no successful coverage result was recorded.") from None
        for item in payload:
            tags = item.get("tickers")
            if not isinstance(tags, list) or ticker not in {str(tag).upper().replace(".", "-") for tag in tags}:
                continue
            published = pd.to_datetime(item.get("published"), unit="s", utc=True, errors="coerce")
            if pd.isna(published):
                continue
            rows.append({"published_at_utc": published.isoformat(), "title": item.get("title", ""),
                         "url": item.get("url", ""), "publisher": item.get("source", "Unknown"), "source": "API Ninjas"})
        if len(payload) < 50:
            break
    return {"headlines": [row for row in normalized_headlines(rows)
            if start <= pd.Timestamp(row["published_at_utc"]).tz_convert("America/New_York").date().isoformat() <= end],
            "truncated": len(payload) == 50 and offset == 50}


def select_events(events, count=4):
    ordered = sorted(events, key=lambda row: row["date"])
    if len(ordered) <= count:
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]


def main():
    load_environment()
    key = os.environ.get("API_NINJAS_API_KEY", "").strip()
    if not key:
        raise SystemExit("Add API_NINJAS_API_KEY to .env.local first. Stock News requires paid access.")
    plan = [(ticker, event) for ticker in ("AAPL", "NVDA", "MSFT", "AMZN", "TSLA")
            for event in select_events(read_spikes(ticker, years=0)["events"])]
    if len(plan) != 20:
        raise SystemExit("Trial needs four detected spikes per stock; available history is insufficient. No news requests sent.")
    report = {"provider": "API Ninjas", "startedAt": pd.Timestamp.now(tz="UTC").isoformat(),
              "plannedEvents": 20, "selection": "Four evenly spaced detected events by date per ticker; exploratory sample.",
              "usableForTraining": False, "events": []}
    path = NEWS_DIR / "spike_trials" / ("api_ninjas_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    for ticker, event in plan:
        baseline = research_spike(ticker, event["date"])
        archive = load_evidence(ticker, event["date"])
        start, end = research_window(event["previousDate"], event["date"])
        try:
            result = fetch_news(ticker, start, end, key)
        except ApiNinjasError as error:
            report["stoppedReason"] = str(error)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report))
            raise SystemExit(str(error)) from None
        now = pd.Timestamp.now(tz="UTC").isoformat()
        matches = classify_sources(result["headlines"], _company_aliases(ticker), event["previousDate"], event["date"], now)
        known_urls = {row["url"] for row in archive["headlines"] + archive["documents"]}
        added = [row for row in matches if row["url"] not in known_urls]
        save_evidence(ticker, event["date"], result["headlines"], [], now)
        row = {"ticker": ticker, "eventDate": event["date"], "windowFrom": start, "windowTo": end,
               "baselineMatched": baseline["documentsMatched"], "headlinesRetrieved": len(result["headlines"]),
               "headlinesMatched": len(matches), "newMatchedUrls": len(added), "sources": matches,
               "truncated": result["truncated"], "retrievedAt": now}
        report["events"].append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        print(json.dumps({field: row[field] for field in ("ticker", "eventDate", "headlinesRetrieved", "headlinesMatched", "newMatchedUrls", "truncated")}))
    print("Trial complete. Evidence and report saved in Convex; matches are topic context, not verified causes.")


if __name__ == "__main__":
    main()
