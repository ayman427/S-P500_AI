"""Source-backed retrospective spike context; never a predictive training label."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

import pandas as pd
import requests

from .finnhub_news import fetch_company_news_range, FinnhubNewsError
from .news_analysis import _company_aliases, _mentions, clean_text, OPINION
from .prediction_store import configured, load_environment
from .risk_news import normalized_headlines
from .sessions import execution_window
from .spikes import read_spikes
from .primary_evidence import fetch_sec_evidence, fetch_company_announcements, research_window, sources_from_documents
from .spike_evidence import collect_documents, load_evidence, save_evidence
from .rise_risk import assess_rise

METHOD = "dated_topic_rules_v4"
TOPICS = (
    ("Earnings / guidance", r"\b(?:earnings|profit|revenue)\b.{0,60}\b(?:beat|beats|exceed\w*|positive surprise)|\b(?:beat|beats)\b.{0,40}\b(?:estimates|expectations)|\brais\w*\b.{0,30}\b(?:guidance|outlook)|financial results|quarterly (?:earnings|results)|earnings (?:report|release)"),
    ("Buybacks / dividends", r"\b(?:buyback|share repurchase|stock repurchase)\b|\b(?:rais\w*|increas\w*)\b.{0,30}\bdividend"),
    ("Acquisition / deal", r"\b(?:takeover|merger|acquisition|buyout|acquire[sd]?|tender offer)\b"),
    ("Products / approvals / contracts", r"\b(?:FDA approv\w*|regulatory approv\w*|product launch|wins? .{0,25}contract|new partnership|strategic (?:partnership|collaboration)|unveils? .{0,30}(?:chip|platform|product))\b"),
    ("Market / economy / analyst", r"\b(?:rate cut|cuts? rates|stimulus|tariff pause|easing inflation|inflation cool\w*|upgrades?|price.target (?:hike|raise)|raises? .{0,20}price.target)\b"),
    ("Speculative activity", r"\b(?:short squeeze|short covering|meme.stock|retail frenzy|social.media hype|pump.and.dump)\b"),
    ("Political / public statements", r"\b(?:Trump|Biden|(?:U\.?S\.? |United States )president|White House)\b.{0,90}\b(?:said|says|statement|remarks|posts?|announc\w*|pledge\w*|threaten\w*|promis\w*)|\b(?:statement|remarks|posts?)\b.{0,40}\b(?:Trump|Biden|(?:U\.?S\.? |United States )president|White House)\b"),
)
RISE = r"(?:rall\w*|surg\w*|jump\w*|soar\w*|rise|rose|gain\w*)"
LINK = r"\b(?:after|following|on news|driven by|as|amid)\b"


def _rpc(kind, function, arguments):
    load_environment()
    if not configured():
        raise RuntimeError("Convex research storage is not configured.")
    from urllib.parse import urlparse
    url = os.environ["CONVEX_URL"].rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("CONVEX_URL must use HTTPS")
    try:
        response = requests.post(url + "/api/" + kind, json={"path": "spikeResearch:" + function,
            "args": {**arguments, "secret": os.environ["PREDICTION_STORE_SECRET"]}, "format": "json"}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise ValueError("Unconfirmed database operation")
        return payload["value"]
    except (requests.RequestException, ValueError, KeyError):
        raise RuntimeError("Convex spike research operation failed; no save was confirmed.") from None


def classify_sources(headlines, company_names, previous_date, event_date, retrieved_at):
    previous_close = execution_window(previous_date)[0]
    event_close = execution_window(event_date)[0]
    sources, seen = [], set()
    company = "(?:" + "|".join(re.escape(name) for name in company_names) + ")"
    start, end = research_window(previous_date, event_date)
    observed = {}
    for item in headlines:
        stamp = pd.to_datetime(item.get("published_at_utc"), utc=True, errors="coerce")
        if not pd.isna(stamp):
            observed.setdefault((item.get("url"), stamp.isoformat(), clean_text(item.get("title", "")), clean_text(item.get("excerpt", ""))), item.get("retrievedAt", retrieved_at))
    price_rise = r"\b(?:shares?|stock)\s+(?:(?:price|prices)\s+)?(?:\w+\s+){0,2}" + RISE + r"\b|\b" + company + r"\s+" + RISE + r"\b"
    for row in normalized_headlines(headlines):
        title, excerpt = clean_text(row["title"]), clean_text(row.get("excerpt", ""))
        identity = re.sub(r"\W+", " ", title.casefold()).strip()
        if identity in seen or re.search(OPINION, title, re.I):
            continue
        seen.add(identity)
        published = pd.Timestamp(row["published_at_utc"])
        local_date = published.tz_convert("America/New_York").date().isoformat()
        if not start <= local_date <= end:
            continue
        # Keep only sentences about this company; provider ticker tags alone are insufficient.
        sentences = [text for text in re.split(r"(?<=[.!?])\s+", excerpt) if _mentions(text, company_names)]
        focused_title = _mentions(title, company_names)
        if focused_title:
            sentences = [text for text in re.split(r"(?<=[.!?])\s+", excerpt) if _mentions(text, company_names) or re.match(r"(?:The company|Its? |The board|Management )", text)]
        text = (title if focused_title else "") + " " + " ".join(sentences)
        categories = [label for label, pattern in TOPICS if re.search(pattern, text, re.I)]
        market = re.search(r"S&P 500|Nasdaq|Wall Street|U\.S\. stocks", title, re.I)
        market_categories = [label for label, pattern in (TOPICS[4], TOPICS[6]) if re.search(pattern, title, re.I)]
        if not categories and market and market_categories:
            categories = market_categories
            text = title
        if not categories:
            continue
        relevant_sentences = [sentence for sentence in sentences if any(re.search(pattern, sentence, re.I) for _, pattern in TOPICS)]
        summary = (relevant_sentences[0] if relevant_sentences else title)[:350]
        reported_link = any(re.search(price_rise, sentence, re.I) and re.search(LINK, sentence, re.I)
            and any(re.search(pattern, sentence, re.I) for _, pattern in TOPICS)
            for sentence in [title if focused_title or market else "", *relevant_sentences])
        reported_link = bool(reported_link and previous_close < published <= event_close)
        sources.append({"title": title, "url": row["url"], "publisher": row["publisher"],
            "publishedAt": published.isoformat(), "retrievedAt": observed.get((row["url"], row["published_at_utc"], title, excerpt), retrieved_at), "summary": summary,
            "basis": "provider_excerpt" if relevant_sentences else "headline_only", "categories": categories,
            "reportedLink": reported_link,
            "timing": "before_move" if published <= previous_close else "during_move" if published <= event_close else "after_close",
            "contextPhase": "earlier_context" if local_date < previous_date else "later_reporting" if local_date > event_date else "move_window"})
    return sorted(sources, key=lambda row: (not row["reportedLink"], abs((pd.Timestamp(row["publishedAt"]).tz_convert("America/New_York").tz_localize(None).normalize() - pd.Timestamp(event_date)).days)))


def read_investigation(ticker, event_date):
    """Read cloud evidence without contacting news, SEC, issuer or model providers."""
    return research_spike(ticker, event_date, collect=False)


def research_spike(ticker, event_date, *, collect=True):
    if not isinstance(event_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date):
        raise ValueError("Use a spike date in YYYY-MM-DD format.")
    spikes = read_spikes(ticker, years=0)
    event = next((row for row in spikes["events"] if row["date"] == event_date), None)
    if event is None:
        raise ValueError("This date is not a detected spike in the saved, completed price history.")
    ticker = spikes["ticker"]
    fingerprint = hashlib.sha256(json.dumps({field: event[field] for field in
        ("date", "previousDate", "previousClose", "close", "return", "threshold")}, sort_keys=True).encode()).hexdigest()
    key = f"{ticker}:{event_date}:{METHOD}:{fingerprint}"
    def response(record, **extra):
        return {**record, **extra, "assessment": assess_rise(event, spikes.get("history", []), record.get("sources", []), record.get("coverage", []))}
    cached = _rpc("query", "get", {"key": key})
    now = pd.Timestamp.now(tz="UTC")
    fetched = pd.to_datetime(cached.get("fetchedAt"), utc=True, errors="coerce") if cached else pd.NaT
    cache_complete = cached and (cached.get("sources") or all(row["status"] in {"ok", "empty", "not_needed", "not_supported", "not_listed"} for row in cached.get("coverage", [])))
    cache_complete = cache_complete and not any(row.get("status") == "not_collected" for row in cached.get("coverage", [])) if cached else False
    if cached and (not collect or (cache_complete and not pd.isna(fetched) and pd.Timedelta(0) <= now - fetched < pd.Timedelta(hours=6))):
        return response(cached, recordId=cached["_id"], saved=True, cached=True)
    load_environment()
    start, end = research_window(event["previousDate"], event_date)
    archive = load_evidence(ticker, event_date)
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    status, message, headlines = "not_configured", "Finnhub is not configured on the server.", []
    if not collect:
        status, message = "not_collected", "Collection runs separately from viewing results."
    elif api_key:
        try:
            headlines = fetch_company_news_range(ticker, start, end, api_key)
            status = "ok" if headlines else "empty"
            message = "" if headlines else "No articles returned for this window; historical coverage is unverified."
        except FinnhubNewsError as error:
            status, message = error.status, str(error)
    sources = classify_sources(archive["headlines"] + headlines, _company_aliases(ticker), event["previousDate"], event_date, now.isoformat())
    sources.extend(sources_from_documents(archive["documents"], event["previousDate"], event_date))
    coverage = [{"provider": "Finnhub", "status": status, "message": message, "documentsChecked": len(headlines), "truncated": False}]
    archived_count = len(archive["headlines"]) + len(archive["documents"])
    if archived_count:
        coverage.append({"provider": "Saved evidence", "status": "ok", "message": "Previously retrieved evidence was rechecked.", "documentsChecked": archived_count, "truncated": any(row.get("textTruncated") for row in archive["documents"])})
    with collect_documents() as documents:
        for provider, fetch in (("SEC EDGAR", fetch_sec_evidence), ("Issuer archive", fetch_company_announcements)):
            result = fetch(ticker, event["previousDate"], event_date, now.isoformat()) if collect else {
                "status": "not_collected", "message": "Collection runs separately from viewing results.", "documentsChecked": 0, "truncated": False, "sources": []}
            sources.extend(result["sources"])
            coverage.append({"provider": provider, **{field: result[field] for field in ("status", "message", "documentsChecked", "truncated")}})
    if collect:
        save_evidence(ticker, event_date, headlines, documents, now.isoformat())
    sources = list({(row["url"], row["publishedAt"]): row for row in sources}.values())
    sources.sort(key=lambda row: (not row["reportedLink"], abs((pd.Timestamp(row["publishedAt"]).date() - pd.Timestamp(event_date).date()).days)))
    matched_count = len(sources)
    retrieved_count = len({(row["url"], row["published_at_utc"]) for row in archive["headlines"] + headlines}) + len({row["url"] for row in archive["documents"] + documents})
    assessment = assess_rise(event, [], sources, coverage)
    priority = assessment["narrativeSources"][:1] + assessment["businessSources"][:1] + assessment["reportedBusinessSources"][:1]
    sources = [sources[i] for i in dict.fromkeys(priority + list(range(len(sources))))][:5]
    if cached and cached.get("sources") and not sources:
        return response({**cached, "coverage": coverage}, recordId=cached["_id"], saved=True, cached=True,
                        providerStatus=status, providerMessage=message + " Showing previously saved research.")
    categories = sorted({category for row in sources for category in row["categories"]})
    evidence_status = "possible_explanation" if sources else "inconclusive" if any(row["documentsChecked"] for row in coverage) else "no_documents_found" if any(row["status"] in {"ok", "empty"} for row in coverage) else "sources_unavailable"
    unknown_overview = {"inconclusive": "Unknown: documents were found, but no sufficiently relevant explanation was identified.",
        "no_documents_found": "Unknown: no dated evidence was returned. This does not establish that no announcement occurred.",
        "sources_unavailable": "Unknown: the evidence sources could not be accessed."}
    record = {"key": key, "ticker": ticker, "eventDate": event_date, "method": METHOD,
        "priceFingerprint": fingerprint, "previousDate": event["previousDate"], "previousClose": event["previousClose"],
        "close": event["close"], "dailyReturn": event["return"], "categories": categories,
        "confidence": "moderate" if any(row["reportedLink"] and row["basis"] == "provider_excerpt" for row in sources) else "limited" if sources else "unknown",
        "overview": "Possible reported context: " + ", ".join(categories) + ". Causation is not verified." if sources else unknown_overview[evidence_status],
        "evidenceStatus": evidence_status, "coverage": coverage,
        "documentsRetrieved": retrieved_count, "documentsMatched": matched_count,
        "providerStatus": status, "providerMessage": message, "fetchedAt": now.isoformat(),
        "windowFrom": start, "windowTo": end,
        "articlesChecked": len(headlines), "sources": sources, "usableForTraining": False}
    receipt = _rpc("mutation", "save", {"record": record})
    if not isinstance(receipt, dict) or not receipt.get("id"):
        raise RuntimeError("Convex did not confirm the research record ID.")
    return response(record, recordId=receipt["id"], saved=True, cached=False)


def main():
    parser = argparse.ArgumentParser(description="Save retrospective spike research to Convex; no local output files.")
    parser.add_argument("tickers", nargs="+", help="Companies to collect, for example AAPL MSFT NVDA")
    parser.add_argument("--date", help="One detected spike date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=5, help="Largest spikes to research, 1-25 (default 5)")
    parser.add_argument("--recent-days", type=int, help="Collect only rises within this many calendar days")
    args = parser.parse_args()
    if not 1 <= args.limit <= 25:
        parser.error("limit must be between 1 and 25")
    if args.recent_days is not None and (not 1 <= args.recent_days <= 3650 or args.date):
        parser.error("recent-days must be 1-3650 and cannot be combined with date")
    failed = False
    for ticker in dict.fromkeys(args.tickers):
        try:
            cutoff = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=args.recent_days)).date().isoformat() if args.recent_days else None
            dates = [args.date] if args.date else [row["date"] for row in read_spikes(ticker, years=0)["events"] if not cutoff or row["date"] >= cutoff][:args.limit]
            for date in dates:
                row = research_spike(ticker, date)
                print(json.dumps({field: row[field] for field in ("ticker", "eventDate", "evidenceStatus", "confidence", "providerStatus", "recordId", "saved")}))
                if any(item.get("status") == "rate_limited" for item in row.get("coverage", [])):
                    raise SystemExit("Collection paused at provider rate limit. Retry on the next scheduled run.")
        except (ValueError, RuntimeError, requests.RequestException):
            failed = True
            print(json.dumps({"ticker": ticker, "saved": False, "error": "Collection failed; existing cloud evidence was retained."}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
