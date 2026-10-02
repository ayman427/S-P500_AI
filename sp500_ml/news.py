"""Headline fetching and brittle, model-independent counterparty extraction.

Kept isolated in its own module: it's regex-based, not used as a model
feature, and only feeds the dashboard's informational "not used by the
model" context panel and paper-trade signal log.
"""
from __future__ import annotations

import html
import json
import re

import pandas as pd
import requests

from .data import _normalize_ticker
from .paths import NEWS_DIR, ROOT


def _parse_yahoo_news(
    payload: dict,
    ticker: str,
    now: pd.Timestamp | None = None,
    limit: int = 5,
) -> list[dict]:
    ticker = _normalize_ticker(ticker)
    cutoff = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    headlines = []
    for item in payload.get("news", []):
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        related_tickers = {
            str(symbol).upper().replace(".", "-")
            for symbol in item.get("relatedTickers", [])
        }
        if related_tickers and ticker not in related_tickers:
            continue
        if not related_tickers and ticker not in title.upper():
            continue

        published = pd.to_datetime(
            item.get("providerPublishTime"), unit="s", utc=True, errors="coerce"
        )
        if pd.isna(published) or published > cutoff:
            continue
        headlines.append(
            {
                "published_at_utc": published.isoformat(),
                "publisher": str(item.get("publisher", "Unknown")),
                "title": title,
                "url": str(item.get("link", "")),
                "related_tickers": sorted(related_tickers),
            }
        )

    headlines.sort(key=lambda item: item["published_at_utc"], reverse=True)
    return headlines[:limit]


def _extract_headline_relationships(title: str, ticker: str) -> list[dict]:
    patterns = [
        (
            "acquires_company_founded_by",
            r"\b(?:buys?|acquires?|purchases?)\s+(?:the\s+)?(?:firm|company|startup|business)\s+founded\s+by\b.*?\b(?P<entity>[A-Z][a-z]+(?:-[A-Z][a-z]+)*\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)*)\s*$",
        ),
        (
            "acquires",
            r"\b(?:acquire|acquires|acquired|buy|buys|bought|purchase|purchases|purchased)\s+(?:the\s+)?(?P<entity>[A-Z][A-Za-z0-9&'’.-]*(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,4})",
        ),
        (
            "acquires",
            r"\b(?:acquisition|purchase)\s+of\s+(?P<entity>[A-Z][A-Za-z0-9&'’.-]*(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,4})",
        ),
        (
            "invests_in",
            r"\b(?:investment|invests?|invested)\s+in\s+(?P<entity>[A-Z][A-Za-z0-9&'’.-]*(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,4})",
        ),
        (
            "bet_on",
            r"\bbet\s+on\s+(?P<entity>[A-Z][A-Za-z0-9&'’.-]*(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,4})",
        ),
        (
            "partners_with",
            r"\b(?:partners?|partnered|partnership)\s+with\s+(?P<entity>[A-Z][A-Za-z0-9&'’.-]*(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,4})",
        ),
        (
            "merges_with",
            r"\b(?:merges?|merged|merger)\s+with\s+(?P<entity>[A-Z][A-Za-z0-9&'’.-]*(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,4})",
        ),
        (
            "counterparty_of_lawsuit",
            r"\b(?:sues?|sued)\s+(?P<entity>[A-Z][A-Za-z0-9&'’.-]*(?:\s+[A-Z][A-Za-z0-9&'’.-]*){0,4})",
        ),
        (
            "founded_by",
            r"\bfounded\s+by\b.*?\b(?P<entity>[A-Z][a-z]+(?:-[A-Z][a-z]+)*\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)*)\s*$",
        ),
    ]
    primary = _normalize_ticker(ticker)
    relationships = []
    seen = set()
    for predicate, pattern in patterns:
        for match in re.finditer(pattern, title):
            entity = match.group("entity").strip(" .,:;!?'")
            entity = re.sub(r"^.*?[’']s\s+", "", entity)
            if not entity or entity.upper().replace(".", "-") == primary:
                continue
            if predicate == "founded_by" and any(
                item["predicate"] == "acquires_company_founded_by"
                and item["counterparty_candidate"].casefold() == entity.casefold()
                for item in relationships
            ):
                continue
            key = (predicate, entity.casefold())
            if key in seen:
                continue
            seen.add(key)
            relationships.append(
                {
                    "subject_ticker": primary,
                    "predicate": predicate,
                    "counterparty_candidate": entity,
                    "evidence": title,
                    "extraction_method": "headline_phrase_rule; candidate requires source verification",
                }
            )
    return relationships


def _parse_wikipedia_entity_context(payload: dict, entity: str, limit: int = 3) -> list[dict]:
    normalized_entity = re.sub(r"[^a-z0-9]+", " ", entity.casefold()).strip()
    entity_tokens = set(normalized_entity.split())
    results = []
    for result in payload.get("query", {}).get("search", []):
        title = str(result.get("title", "")).strip()
        snippet = html.unescape(re.sub(r"<[^>]+>", "", str(result.get("snippet", ""))))
        searchable = re.sub(r"[^a-z0-9]+", " ", f"{title} {snippet}".casefold())
        if not entity_tokens or not entity_tokens.issubset(set(searchable.split())):
            continue
        title_match = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip() == normalized_entity
        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": f"https://en.wikipedia.org/?curid={result.get('pageid', '')}",
                "match": "exact_title" if title_match else "search_snippet_match",
            }
        )
    results.sort(key=lambda item: item["match"] != "exact_title")
    return results[:limit]


def _search_entity_context(entity: str, limit: int = 3) -> list[dict]:
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": f'"{entity}" company',
                "format": "json",
            },
            headers={"User-Agent": "Mozilla/5.0 S&P500_AI/1.0"},
            timeout=20,
        )
        response.raise_for_status()
        return _parse_wikipedia_entity_context(response.json(), entity, limit)
    except (requests.RequestException, ValueError) as error:
        print(f"Entity context search unavailable for {entity}: {error}")
        return []


def _enrich_news_relationships(headlines: list[dict], ticker: str) -> list[dict]:
    context_cache = {}
    enriched = []
    for headline in headlines:
        item = dict(headline)
        relationships = _extract_headline_relationships(item["title"], ticker)
        for relationship in relationships:
            entity = relationship["counterparty_candidate"]
            if entity not in context_cache:
                context_cache[entity] = _search_entity_context(entity)
            relationship["counterparty_context"] = context_cache[entity]
        item["relationships"] = relationships
        enriched.append(item)
    return enriched


def _fetch_recent_news(ticker: str, limit: int = 5) -> list[dict]:
    try:
        response = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": ticker, "newsCount": 20, "quotesCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        headlines = _parse_yahoo_news(response.json(), ticker, limit=limit)
        return _enrich_news_relationships(headlines, ticker)
    except (requests.RequestException, ValueError) as error:
        print(f"Recent Yahoo Finance news unavailable for {ticker}: {error}")
        return []


def _save_news_cache(ticker: str, headlines: list[dict]) -> dict:
    ticker = _normalize_ticker(ticker)
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "headlines": headlines,
    }
    path = NEWS_DIR / f"{ticker}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def fetch_news(ticker: str) -> None:
    ticker = _normalize_ticker(ticker)
    payload = _save_news_cache(ticker, _fetch_recent_news(ticker))
    print(
        f"Fetched {len(payload['headlines'])} recent headlines for {ticker}; "
        f"saved to {(NEWS_DIR / f'{ticker}.json').relative_to(ROOT)}."
    )
    for headline in payload["headlines"]:
        print(
            f"{headline['published_at_utc']} | {headline['publisher']} | "
            f"{headline['title']} | {headline['url']}"
        )
        for relationship in headline.get("relationships", []):
            print(
                f"  Candidate relationship: {ticker} "
                f"{relationship['predicate']} {relationship['counterparty_candidate']}"
            )
            for context in relationship.get("counterparty_context", []):
                print(f"  Context: {context['title']} | {context['snippet']} | {context['url']}")
