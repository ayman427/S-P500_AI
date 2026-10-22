"""Bounded historical SEC filings and issuer announcements; no local storage."""
from __future__ import annotations

import json
import re
import threading
import time
import socket
import html as html_text
from contextlib import ExitStack
from urllib.parse import unquote, urljoin, urlparse

from lxml import html, etree
import pandas as pd
import requests

from .fundamentals import _get_sec_user_agent
from .news_analysis import clean_text
from .paths import FUNDAMENTAL_DIR
from .sessions import execution_window
from .sec_connection import SecDNSAdapter, SEC_HOSTS
from .spike_evidence import remember_document
from .rise_risk import positive_development

_request_lock = threading.Lock()
_last_request = 0.
PRIMARY_TOPICS = (
    ("Earnings / guidance", r"financial results|quarterly results|(?:first|second|third|fourth)[- ]quarter.{0,20}results|results of operations|announced.{0,50}results for the quarter|raises? .{0,25}(?:guidance|outlook)"),
    ("Buybacks / dividends", r"buyback|repurchas\w*.{0,140}(?:shares|common stock)|(?:increase|raise)\w*.{0,35}dividend"),
    ("Acquisition / deal", r"\b(?:announc\w*|agree\w*|complet\w*|propos\w*).{0,80}\b(?:merger|acquisition|takeover|buyout)\b|\bacquir(?:e|es|ed)\b.{0,60}(?:agreement|for \$)"),
    ("Products / approvals / contracts", r"FDA approv\w*|regulatory approv\w*|wins? .{0,25}contract|(?:announce\w*|enter\w*|expand\w*|new).{0,40}(?:partnership|collaboration)|(?:launch\w*|unveil\w*).{0,40}(?:product|chip|platform)"),
)


def research_window(previous_date, event_date):
    return min(previous_date, (pd.Timestamp(event_date) - pd.Timedelta(days=7)).date().isoformat()), (pd.Timestamp(event_date) + pd.Timedelta(days=3)).date().isoformat()


class PrimaryEvidenceError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _allowed_url(url):
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return False
    path = unquote(parsed.path)
    if parsed.scheme != "https" or parsed.username or parsed.password or port not in (None, 443) or parsed.fragment or ".." in path.split("/"):
        return False
    return ((parsed.hostname == "data.sec.gov" and re.fullmatch(r"/submissions/CIK\d{10}(?:-submissions-\d+)?\.json", path))
            or (parsed.hostname == "www.sec.gov" and (path == "/files/company_tickers.json" or re.fullmatch(r"/Archives/edgar/data/\d+/\d{18}/[A-Za-z0-9_.-]+", path)))
            or (parsed.hostname == "www.apple.com" and re.fullmatch(r"/newsroom/(?:archive/)?\d{4}/\d{2}/(?:[a-z0-9-]+/)?", path))
            or (parsed.hostname == "nvidianews.nvidia.com" and (path == "/helper-search-news" or re.fullmatch(r"/news/[a-z0-9-]+", path)))
            or (parsed.hostname == "www.microsoft.com" and re.fullmatch(r"/en-us/investor/earnings/FY-\d{4}-Q[1-4]/press-release-webcast", path)))


def _download(url, headers):
    global _last_request
    if not _allowed_url(url):
        raise PrimaryEvidenceError("unavailable", "Unsupported evidence URL.")
    try:
        # Serialize requests and stay below SEC's published per-user request limit.
        with _request_lock:
            time.sleep(max(0., .15 - (time.monotonic() - _last_request)))
            _last_request = time.monotonic()
        with ExitStack() as stack:
            options = dict(headers=headers, timeout=(4, 12), allow_redirects=False, stream=True)
            try:
                response = stack.enter_context(requests.get(url, **options))
            except requests.ConnectionError:
                host = urlparse(url).hostname
                if host not in SEC_HOSTS:
                    raise
                try:
                    socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
                except socket.gaierror:
                    session = stack.enter_context(requests.Session())
                    session.mount(f"https://{host}/", SecDNSAdapter())
                    response = stack.enter_context(session.get(url, **options))
                else:
                    raise
            if response.status_code in (403, 429):
                raise PrimaryEvidenceError("access_denied" if response.status_code == 403 else "rate_limited", "Evidence source denied or limited access.")
            response.raise_for_status()
            if response.is_redirect:
                raise PrimaryEvidenceError("unavailable", "Evidence source redirected; no unverified destination was fetched.")
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > 8_000_000:
                    raise PrimaryEvidenceError("unavailable", "Evidence document exceeds the download limit.")
            return bytes(body)
    except requests.RequestException:
        raise PrimaryEvidenceError("unavailable", "Evidence source is unreachable or returned an error.") from None


def _json(url, headers):
    try:
        payload = json.loads(_download(url, headers))
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
        return payload
    except (ValueError, TypeError):
        raise PrimaryEvidenceError("unavailable", "Evidence source returned invalid metadata.") from None


def _source(body, title, url, published, precision, previous_date, event_date, retrieved_at, basis):
    stamp = pd.Timestamp(published)
    day = stamp.tz_convert("America/New_York").date() if stamp.tzinfo is not None else stamp.date()
    start, end = research_window(previous_date, event_date)
    if not pd.Timestamp(start).date() <= day <= pd.Timestamp(end).date():
        return None
    if day > pd.Timestamp(retrieved_at).tz_convert("America/New_York").date():
        return None
    if stamp.tzinfo is not None and stamp > pd.Timestamp(retrieved_at):
        return None
    tree = html.fromstring(body)
    for node in tree.xpath("//script|//style|//nav|//footer|//aside"):
        node.drop_tree()
    nodes = tree.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' article-body ')]") or tree.xpath("//article | //*[contains(concat(' ', normalize-space(@class), ' '), ' article ')]")
    text = clean_text((nodes[0] if nodes else tree).text_content())
    remember_document(text, title=title, url=url, publishedAt=published, publicationPrecision=precision, retrievedAt=retrieved_at, basis=basis)
    text = re.split(r"forward[- ]looking statements|safe harbor statement|cautionary note", text, maxsplit=1, flags=re.I)[0]
    categories = [label for label, pattern in PRIMARY_TOPICS if re.search(pattern, text, re.I)]
    if not categories:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", text)
    relevant = [sentence for sentence in sentences if any(re.search(pattern, sentence, re.I) for _, pattern in PRIMARY_TOPICS) or positive_development(sentence)]
    summary = next((sentence for sentence in relevant if positive_development(sentence)), relevant[0] if relevant else text)[:350]
    timing = "unknown_time"
    if precision == "timestamp":
        stamp = pd.Timestamp(published)
        timing = "before_move" if stamp <= execution_window(previous_date)[0] else "during_move" if stamp <= execution_window(event_date)[0] else "after_close"
    return {"title": title, "url": url, "publisher": "SEC EDGAR / issuer" if basis == "sec_filing" else "Company announcement",
        "publishedAt": published, "retrievedAt": retrieved_at, "summary": summary, "basis": basis,
        "categories": categories, "reportedLink": False, "timing": timing, "publicationPrecision": precision,
        "timestampBasis": "sec_acceptance" if basis == "sec_filing" else "issuer_publication",
        "contextPhase": "earlier_context" if day < pd.Timestamp(previous_date).date() else "later_reporting" if day > pd.Timestamp(event_date).date() else "move_window"}


def sources_from_documents(documents, previous_date, event_date):
    sources = []
    for row in documents:
        source = _source(html_text.escape(row["text"]).encode(), row["title"], row["url"], row["publishedAt"],
            row["publicationPrecision"], previous_date, event_date, row["retrievedAt"], row["basis"])
        if source:
            sources.append(source)
    return sources


def _filing_rows(columns, start, end):
    rows = []
    for i, day in enumerate(columns.get("filingDate", [])):
        if not isinstance(day, str) or not start <= day <= end:
            continue
        try:
            row = {key: value[i] for key, value in columns.items() if isinstance(value, list) and i < len(value)}
            if row.get("form") in {"8-K", "8-K/A", "6-K", "6-K/A", "10-Q", "10-K"} and re.fullmatch(r"\d{10}-\d{2}-\d{6}", row.get("accessionNumber", "")):
                rows.append(row)
        except (TypeError, ValueError):
            continue
    return rows


def fetch_sec_evidence(ticker, previous_date, event_date, retrieved_at):
    result = {"status": "empty", "message": "No relevant current-report filings in this window.", "sources": [], "documentsChecked": 0, "truncated": False}
    user_agent = _get_sec_user_agent()
    if not user_agent:
        return {**result, "status": "not_configured", "message": "SEC_USER_AGENT is not configured."}
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    start, end = research_window(previous_date, event_date)
    try:
        path = FUNDAMENTAL_DIR / "company_tickers.json"
        mapping = json.loads(path.read_text()) if path.exists() else _json("https://www.sec.gov/files/company_tickers.json", headers)
        company = next((row for row in mapping.values() if str(row.get("ticker", "")).upper().replace(".", "-") == ticker), None)
        if company is None:
            return {**result, "status": "not_listed", "message": "No SEC issuer mapping for this ticker."}
        cik = int(company["cik_str"])
        if not 0 < cik < 10**10:
            raise ValueError("Invalid CIK")
        submissions = _json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers)
        filings = submissions.get("filings", {})
        rows = _filing_rows(filings.get("recent", {}), start, end)
        files = [row for row in filings.get("files", []) if row.get("filingFrom", "9999") <= end and row.get("filingTo", "") >= start]
        result["truncated"] = len(files) > 3
        for file in files[:3]:
            name = file.get("name", "")
            if not re.fullmatch(rf"CIK{cik:010d}-submissions-\d+\.json", name):
                continue
            rows.extend(_filing_rows(_json("https://data.sec.gov/submissions/" + name, headers), start, end))
        rows = list({row["accessionNumber"]: row for row in rows}.values())
        result["truncated"] |= len(rows) > 3
        for row in sorted(rows, key=lambda row: (abs((pd.Timestamp(row["filingDate"]) - pd.Timestamp(event_date)).days), row["form"] not in {"8-K", "6-K"}))[:3]:
            accession = row["accessionNumber"].replace("-", "")
            primary = row.get("primaryDocument", "")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", primary):
                continue
            root = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
            url = root + primary
            body = _download(url, headers)
            result["documentsChecked"] += 1
            tree = html.fromstring(body)
            stamp = pd.to_datetime(row.get("acceptanceDateTime"), errors="coerce")
            precision = "timestamp" if not pd.isna(stamp) and stamp.tzinfo is not None else "date"
            published = stamp.isoformat() if precision == "timestamp" else row["filingDate"]
            title = f"{company.get('title', ticker)}: {row['form']} filing"
            source = _source(body, title, url, published, precision, previous_date, event_date, retrieved_at, "sec_filing")
            if source:
                result["sources"].append(source)
            links = []
            for anchor in tree.xpath("//a[@href]"):
                link = urljoin(url, anchor.get("href"))
                description = clean_text(anchor.getparent().text_content())[:500]
                if link != url and link.startswith(root) and _allowed_url(link) and re.search(r"99[._-]?1|earnings|press release", anchor.get("href", "") + " " + description, re.I):
                    if link not in links:
                        links.append(link)
            result["truncated"] |= len(links) > 2
            for link in links[:2]:
                announcement = _download(link, headers)
                result["documentsChecked"] += 1
                source = _source(announcement, f"{company.get('title', ticker)}: company announcement attached to {row['form']}", link, published, precision, previous_date, event_date, retrieved_at, "sec_filing")
                if source:
                    result["sources"].append(source)
        if result["documentsChecked"]:
            result.update(status="ok", message="Dated SEC documents were checked; filing time is not necessarily the original announcement time.")
    except PrimaryEvidenceError as error:
        result.update(status="partial" if result["documentsChecked"] else error.status, message=str(error), truncated=bool(result["documentsChecked"]) or result["truncated"])
    except (ValueError, TypeError, KeyError, AttributeError, etree.ParserError):
        result.update(status="partial" if result["documentsChecked"] else "unavailable", message="SEC metadata or documents could not be parsed.", truncated=bool(result["documentsChecked"]) or result["truncated"])
    return result


def fetch_company_announcements(ticker, previous_date, event_date, retrieved_at):
    result = {"status": "not_supported", "message": "No direct issuer-archive adapter for this company; SEC exhibits remain available.", "sources": [], "documentsChecked": 0, "truncated": False}
    if ticker in {"NVDA", "MSFT"}:
        return _fetch_additional_issuer(ticker, previous_date, event_date, retrieved_at)
    # shortcut: Other issuers use SEC; add direct parsers as verified archive formats become available.
    if ticker != "AAPL":
        return result
    result.update(status="empty", message="No matching issuer announcements in this window.")
    start, end_date = research_window(previous_date, event_date)
    end = pd.Timestamp(end_date)
    months = pd.period_range(pd.Timestamp(start), end, freq="M")
    headers = {"User-Agent": "StockResearch/1.0", "Accept-Encoding": "gzip, deflate"}
    seen = set()
    try:
        for month in months:
            archive = f"https://www.apple.com/newsroom/archive/{month.year}/{month.month:02d}/"
            for page in range(1, 5):
                tree = html.fromstring(_download(archive + (f"?page={page}" if page > 1 else ""), headers))
                pages = tree.xpath("//*[contains(@class, 'pagination-total')]/text()")
                total = int(pages[0].strip()) if pages else 1
                result["truncated"] |= total > 4
                for anchor in tree.xpath("//a[@href]"):
                    link = urljoin(archive, anchor.get("href"))
                    if not re.fullmatch(r"https://www\.apple\.com/newsroom/\d{4}/\d{2}/[a-z0-9-]+/", link) or link in seen:
                        continue
                    seen.add(link)
                    label = clean_text(anchor.text_content())
                    match = re.search(r"[A-Z][a-z]+ \d{1,2}, \d{4}", label)
                    day = pd.to_datetime(match.group(), errors="coerce") if match else pd.NaT
                    if pd.isna(day) or not pd.Timestamp(start) <= day <= end:
                        continue
                    if result["documentsChecked"] >= 4:
                        result["truncated"] = True
                        continue
                    body = _download(link, headers)
                    result["documentsChecked"] += 1
                    doc = html.fromstring(body)
                    heading = doc.xpath("//h1")
                    title = clean_text(heading[0].text_content()) if heading else label
                    source = _source(body, title, link, day.date().isoformat(), "date", previous_date, event_date, retrieved_at, "company_announcement")
                    if source:
                        result["sources"].append(source)
                if page >= total:
                    break
        if result["documentsChecked"]:
            result.update(status="ok", message="Dated issuer announcements were checked; exact publication time is unavailable.")
    except PrimaryEvidenceError as error:
        result.update(status="partial" if result["documentsChecked"] else error.status, message=str(error), truncated=bool(result["documentsChecked"]) or result["truncated"])
    except (ValueError, TypeError, etree.ParserError):
        result.update(status="partial" if result["documentsChecked"] else "unavailable", message="The company archive could not be parsed.", truncated=bool(result["documentsChecked"]) or result["truncated"])
    return result


def _fetch_additional_issuer(ticker, previous_date, event_date, retrieved_at):
    result = {"status": "empty", "message": "No matching dated issuer releases.", "sources": [], "documentsChecked": 0, "truncated": False}
    start, end = research_window(previous_date, event_date)
    headers = {"User-Agent": "StockResearch/1.0"}
    seen = set()
    try:
        if ticker == "NVDA":
            for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
                for page in range(1, 5):
                    url = f"https://nvidianews.nvidia.com/helper-search-news?ct=releases&page={page}&year={year}&q=financial"
                    page_body = _download(url, headers)
                    if not page_body.strip():
                        break
                    tree = html.fromstring(page_body)
                    items = tree.xpath("//div[@class='index-item-text']")
                    if not items:
                        break
                    dates = []
                    for item in items:
                        dates_text = item.xpath(".//*[contains(@class,'index-item-text-info-date')]/text()")
                        day = pd.to_datetime(dates_text[0], errors="coerce") if dates_text else pd.NaT
                        if pd.isna(day):
                            continue
                        dates.append(day)
                        links = item.xpath(".//h3/a[@href]")
                        if not links or not pd.Timestamp(start) <= day <= pd.Timestamp(end):
                            continue
                        link = urljoin(url, links[0].get("href"))
                        if link in seen or not link.startswith("https://nvidianews.nvidia.com/news/") or not _allowed_url(link):
                            continue
                        seen.add(link)
                        if result["documentsChecked"] >= 4:
                            result["truncated"] = True
                            continue
                        body = _download(link, headers)
                        result["documentsChecked"] += 1
                        source = _source(body, clean_text(links[0].text_content()), link, day.date().isoformat(), "date", previous_date, event_date, retrieved_at, "company_announcement")
                        if source:
                            result["sources"].append(source)
                    if result["documentsChecked"] >= 4:
                        result["truncated"] = True
                        break
                    if dates and max(dates) < pd.Timestamp(start):
                        break
                    if page == 4:
                        result["truncated"] = True
        else:
            for month in pd.period_range(start, end, freq="M"):
                if month.month not in {1, 4, 7, 10}:
                    continue
                fiscal_year = month.year + (month.month == 10)
                quarter = {1: 2, 4: 3, 7: 4, 10: 1}[month.month]
                url = f"https://www.microsoft.com/en-us/investor/earnings/FY-{fiscal_year}-Q{quarter}/press-release-webcast"
                body = _download(url, headers)
                tree = html.fromstring(body)
                match = re.search(r"REDMOND.{0,80}?([A-Z][a-z]+ \d{1,2}, \d{4})", clean_text(tree.text_content()))
                day = pd.to_datetime(match.group(1), errors="coerce") if match else pd.NaT
                if pd.isna(day):
                    result.update(status="unavailable", message="Issuer publication date could not be verified.")
                    continue
                if not pd.Timestamp(start) <= day <= pd.Timestamp(end):
                    continue
                result["documentsChecked"] += 1
                source = _source(body, f"Microsoft FY{fiscal_year} Q{quarter} earnings release", url, day.date().isoformat(), "date", previous_date, event_date, retrieved_at, "company_announcement")
                if source:
                    result["sources"].append(source)
        if result["documentsChecked"]:
            result.update(status="ok", message="Dated issuer financial releases checked; exact publication time unavailable.")
    except PrimaryEvidenceError as error:
        result.update(status="partial" if result["documentsChecked"] else error.status, message=str(error), truncated=bool(result["documentsChecked"]) or result["truncated"])
    except (ValueError, TypeError, etree.ParserError):
        result.update(status="partial" if result["documentsChecked"] else "unavailable", message="Issuer archive could not be parsed.", truncated=bool(result["documentsChecked"]) or result["truncated"])
    return result
