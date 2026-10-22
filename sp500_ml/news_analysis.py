"""Explainable relevance ranking and source-grounded extractive news summaries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import time
from urllib.parse import urljoin, urlparse

from lxml import html, etree
import pandas as pd
import requests

from .data import _normalize_ticker
from .paths import NEWS_DIR
from .paths import FUNDAMENTAL_DIR

TOPICS = (
    ("Earnings / guidance", r"earnings miss|miss(?:es|ed)? estimates|profit warning|weak earnings|cut\w* (?:its |their )?(?:guidance|outlook)|lower\w* (?:its |their )?(?:guidance|outlook)|revenue (?:fell|fall|declin|miss)|disappoint\w* (?:earnings|results)", 5),
    ("Analyst downgrade", r"downgrad\w*|price.target cut|cuts? (?:its |their )?price.target", 4),
    ("Legal / regulatory", r"antitrust|lawsuit|investigation|regulator|recall|fraud|sanctions", 3),
    ("Demand / operations", r"(?:weak|soft) demand|slow\w* sales|production delay|supply.chain disrupt|export (?:ban|restrict)|layoffs|data breach", 4),
    ("Market / economy", r"tariff|rate hike|inflation|recession|market.sell.off", 2),
)
DROP = r"\b(?:plung\w*|tumbl\w*|sell.off|slump\w*|crash\w*|shares? (?:fall\w*|fell|drop\w*|sink\w*|slid\w*)|stock (?:fall\w*|fell|drop\w*|sink\w*|slid\w*))\b"
OPINION = r"buy.the.dip|stocks? to buy|should you buy|buy (?:now|before)|best stocks|passive income|millionaire|price prediction|undervalued"
HOSTS = ("finnhub.io", "finance.yahoo.com", "reuters.com", "cnbc.com", "apnews.com",
         "bbc.com", "marketwatch.com", "businesswire.com", "fool.com", "investopedia.com",
         "barrons.com", "wsj.com", "bloomberg.com", "seekingalpha.com", "benzinga.com")


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip()


def _company_aliases(ticker):
    names = [ticker]
    try:
        payload = json.loads((FUNDAMENTAL_DIR / "company_tickers.json").read_text())
        for row in payload.values():
            if str(row.get("ticker", "")).upper().replace(".", "-") == ticker:
                name = re.sub(r"\b(?:incorporated|inc|corporation|corp|plc|ltd|limited|holdings)\b\.?", "", row.get("title", ""), flags=re.I).strip(" ,.")
                if name:
                    names.append(name)
                    first = re.split(r"[\s.]", name)[0]
                    if len(first) >= 4 and first.lower() not in {"general", "united", "american", "advanced", "international", "global", "first", "national", "bank"}:
                        names.append(first)
                break
    except (ValueError, OSError, RuntimeError, requests.RequestException):
        pass
    return names


def _mentions(text, names):
    return any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I) for name in names if len(name) >= 3) or any(
        re.search(r"(?:\$|\(|NYSE:\s*|NASDAQ:\s*)" + re.escape(name) + r"\b", text) for name in names if len(name) < 3)


def rank_news(headlines, company_names=None):
    ranked = []
    seen = set()
    for row in headlines:
        title = clean_text(row.get("title", ""))
        identity = re.sub(r"\W+", " ", title.casefold()).strip()
        if identity in seen:
            continue
        seen.add(identity)
        text = title + " " + clean_text(row.get("excerpt", ""))
        reasons, topics, score = [], [], 0
        if re.search(DROP, title, re.I):
            score += 5
            reasons.append("Reports a price decline")
        elif re.search(DROP, text, re.I):
            score += 3
            reasons.append("Discusses a price decline")
        for label, pattern, weight in TOPICS:
            if re.search(pattern, text, re.I):
                score += weight
                topics.append(label)
        if re.search(OPINION, title, re.I):
            score -= 7
            reasons.append("Investment opinion / listicle")
        if row.get("sameDay"):
            score += 1
        elif row.get("daysFromEvent", 0) > 0:
            score -= 1
        has_signal = bool(topics) or any("price decline" in reason for reason in reasons)
        focused = True
        if company_names:
            focused = _mentions(title, company_names)
            if focused:
                score += 5
                reasons.append("Selected company in headline")
            else:
                focused = any(_mentions(sentence, company_names) and
                    (re.search(DROP, sentence, re.I) or any(re.search(pattern, sentence, re.I) for _, pattern, _ in TOPICS))
                    for sentence in re.split(r"(?<=[.!?])\s+", clean_text(row.get("excerpt", ""))))
                if not focused and re.search(r"S&P 500|Nasdaq|Wall Street|Dow Jones|U\.S\. stocks", title, re.I) and re.search(DROP, title, re.I):
                    focused = True
                    reasons.append("Broad market context")
        ranked.append({**row, "relevanceScore": score, "relevanceReasons": [*reasons, *topics],
                       "topics": topics, "relevant": focused and has_signal and score >= 4,
                       **({"companyTerms": company_names} if company_names and "Broad market context" not in reasons else {})})
    return sorted(ranked, key=lambda row: (-row["relevanceScore"], abs(row.get("daysFromEvent", 0)), row.get("published_at_utc", "")))


def _allowed_url(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return (parsed.scheme == "https" and not parsed.username and not parsed.password
            and parsed.port in (None, 443) and any(host == domain or host.endswith("." + domain) for domain in HOSTS))


def read_article(url):
    """Read public article text only, with bounded downloads and checked redirects."""
    try:
        deadline = time.monotonic() + 8
        for _ in range(4):
            if time.monotonic() >= deadline or not _allowed_url(url):
                return ""
            with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(2, 3),
                              allow_redirects=False, stream=True) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("Location", ""))
                    continue
                if response.status_code != 200 or "html" not in response.headers.get("Content-Type", ""):
                    return ""
                content = bytearray()
                for chunk in response.iter_content(32768):
                    content.extend(chunk)
                    if len(content) > 1_000_000 or time.monotonic() >= deadline:
                        return ""
                tree = html.fromstring(bytes(content))
            for node in tree.xpath("//script[@type='application/ld+json']"):
                try:
                    objects = json.loads(node.text or "{}")
                except ValueError:
                    continue
                queue = objects if isinstance(objects, list) else [objects]
                while queue:
                    item = queue.pop()
                    if isinstance(item, dict):
                        if isinstance(item.get("articleBody"), str) and len(item["articleBody"]) > 200:
                            return clean_text(item["articleBody"])[:25000]
                        queue.extend(value for value in item.values() if isinstance(value, (dict, list)))
                    elif isinstance(item, list):
                        queue.extend(item)
            for node in tree.xpath("//script|//style|//nav|//footer|//aside|//form"):
                node.drop_tree()
            paragraphs = tree.xpath("//article//p | //*[@id='article-body']//p | //*[contains(@class, 'article-body')]//p")
            text = clean_text(" ".join(node.text_content() for node in paragraphs))
            return text[:25000] if len(text) > 200 else ""
    except (requests.RequestException, ValueError, TypeError, etree.ParserError):
        return ""
    return ""


def summarize_article(row):
    body = read_article(row["url"])
    names = row.get("companyTerms")
    if body and names and not _mentions(body, names):
        body = ""
    basis = "article" if body else "provider_excerpt" if row.get("excerpt") else "headline_only"
    text = clean_text(body or row.get("excerpt", ""))
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    if names:
        focused = {index for index, sentence in enumerate(sentences) if _mentions(sentence, names)}
        context = focused | {index + 1 for index in focused}
        sentences = [sentence for index, sentence in enumerate(sentences) if index in context]
        text = " ".join(sentences)
        if not sentences:
            text, basis = "", "headline_only"
    candidates = []
    for index, sentence in enumerate(sentences):
        if not 35 <= len(sentence) <= 600:
            continue
        weight = 3 if re.search(DROP, sentence, re.I) else 0
        weight += sum(value for _, pattern, value in TOPICS if re.search(pattern, sentence, re.I))
        if re.search(r"subscribe|sign up|cookies|all rights reserved", sentence, re.I):
            continue
        candidates.append((weight, index, sentence))
    best = sorted(sorted(candidates, key=lambda item: (-item[0], item[1]))[:2], key=lambda item: item[1])
    summary = " ".join(item[2] for item in best)
    if not summary and text:
        summary = text[:450].rsplit(" ", 1)[0] + "…" if len(text) > 450 else text
    return {key: row[key] for key in ("title", "url", "publisher", "published_at_utc", "publicationDate", "daysFromEvent", "relevanceReasons", "topics") if key in row} | {
        "summary": summary, "basis": basis,
    }


def analyze_event_news(ticker, event_date, window_days=0):
    from .risk_news import read_event_news
    ticker = _normalize_ticker(ticker)
    news = read_event_news(ticker, event_date, window_days)
    candidates = [row for row in rank_news(news["headlines"], _company_aliases(ticker)) if row["relevant"]][:3]
    fingerprint = hashlib.sha256(json.dumps({"revision": 2, "articles": candidates, "providerMessage": news.get("providerMessage", "")}, sort_keys=True).encode()).hexdigest()
    path = NEWS_DIR / "analysis" / "v1" / event_date / str(window_days) / f"{ticker}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        created = pd.to_datetime(cached.get("createdAt"), utc=True, errors="coerce")
        if cached.get("fingerprint") == fingerprint and not pd.isna(created) and pd.Timedelta(0) <= pd.Timestamp.now(tz="UTC") - created < pd.Timedelta(hours=6):
            return {**cached, "cached": True}
    with ThreadPoolExecutor(max_workers=3) as executor:
        articles = list(executor.map(summarize_article, candidates))
    readable = [row for row in articles if row["summary"]]
    if readable:
        articles = readable
    topics = list(dict.fromkeys(topic for row in articles for topic in row["topics"]))
    result = {"ticker": ticker, "eventDate": event_date, "windowDays": window_days,
        "fingerprint": fingerprint, "createdAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "cached": False, "method": "rule_ranked_extractive", "articles": articles,
        "overview": ("Most relevant reports discuss: " if any(row["basis"] != "headline_only" for row in articles) else "Headlines suggest these topics: ") + ", ".join(topics) + "." if topics else
            "Price-decline reports found; no specific catalyst identified." if articles else
            "No strong matches in the available news. Try the wider date window or a more recent event.",
        "providerMessage": news.get("providerMessage", ""),
    }
    if not news.get("providerMessage"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    return result
