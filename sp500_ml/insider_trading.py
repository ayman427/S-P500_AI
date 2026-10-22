"""API Ninjas discovery, SEC-verified purchases, and shared Convex retention."""
import os
import re
import hashlib
from urllib.parse import urlparse, unquote

from lxml import etree, html
import numpy as np
import pandas as pd
import requests

from .data import _normalize_ticker
from .fundamentals import _get_sec_user_agent
from .prediction_store import configured, load_environment
from .primary_evidence import _download, PrimaryEvidenceError
from .sessions import execution_window


class InsiderSourceError(Exception):
    pass


def _store(kind, name, arguments):
    load_environment()
    if not configured():
        raise RuntimeError("Convex insider storage is not configured.")
    url = os.environ["CONVEX_URL"].rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("CONVEX_URL must use HTTPS")
    try:
        response = requests.post(url + "/api/" + kind, json={"path": "insiderTransactions:" + name, "format": "json",
            "args": {"secret": os.environ["PREDICTION_STORE_SECRET"], **arguments}}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise ValueError("Unconfirmed database operation")
        return payload["value"]
    except (requests.RequestException, ValueError, KeyError):
        raise RuntimeError("Convex insider evidence operation was not confirmed.") from None


def fetch_candidates(ticker, start, end, key):
    try:
        response = requests.get("https://api.api-ninjas.com/v1/insidertransactions",
            params={"ticker": ticker, "form_type": "4", "transaction_code": "P",
                    "min_transaction_date": start, "max_transaction_date": end},
            headers={"X-Api-Key": key, "Accept": "application/json"}, timeout=15, allow_redirects=False)
        if response.is_redirect:
            raise InsiderSourceError("API Ninjas redirected the request. Coverage is incomplete.")
        if response.status_code in (400, 401, 403, 429):
            raise InsiderSourceError("API Ninjas denied or limited insider access. Coverage is incomplete.")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) > 100 or any(not isinstance(row, dict) for row in payload):
            raise ValueError("Unexpected response")
        return payload
    except (requests.RequestException, ValueError):
        raise InsiderSourceError("API Ninjas insider data is unavailable. Coverage is incomplete.") from None


def _date(text):
    try:
        return isinstance(text, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text)) and pd.Timestamp(text).date().isoformat() == text
    except ValueError:
        return False


def parse_filing(index_body, xml_body, ticker, cik, accession, filing_url, xml_url, retrieved_at):
    tree = html.fromstring(index_body)
    if accession not in tree.text_content():
        raise ValueError("Filing accession did not match")
    accepted = tree.xpath('//div[normalize-space(.)="Accepted"]/following-sibling::div[1]/text()')
    filed = tree.xpath('//div[normalize-space(.)="Filing Date"]/following-sibling::div[1]/text()')
    if len(accepted) != 1 or not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", accepted[0].strip()) or len(filed) != 1 or not _date(filed[0].strip()):
        raise ValueError("Filing publication metadata is missing")
    stamp = pd.Timestamp(accepted[0].strip()).tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    if stamp > pd.Timestamp(retrieved_at):
        raise ValueError("Filing publication is in the future")
    root = etree.fromstring(xml_body, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    if root.tag != "ownershipDocument" or root.findtext("documentType") != "4" or int(root.findtext("issuer/issuerCik", "0")) != int(cik):
        raise ValueError("Filing issuer or form did not match")
    owners = root.findall("reportingOwner")
    names = [node.findtext("reportingOwnerId/rptOwnerName", "").strip() for node in owners]
    ids = [str(int(node.findtext("reportingOwnerId/rptOwnerCik", "0"))) for node in owners]
    if not owners or not all(names) or "0" in ids:
        raise ValueError("Insider identity is missing")
    positions = [node.findtext("reportingOwnerRelationship/officerTitle", "").strip() or
                 ("Director" if node.findtext("reportingOwnerRelationship/isDirector") in {"1", "true"} else "Reporting owner") for node in owners]
    records = []
    for index, node in enumerate(root.findall("nonDerivativeTable/nonDerivativeTransaction")):
        if node.findtext("transactionCoding/transactionCode") != "P" or node.findtext("transactionAmounts/transactionAcquiredDisposedCode/value") != "A":
            continue
        transaction_date = node.findtext("transactionDate/value")
        if not _date(transaction_date) or pd.Timestamp(transaction_date).date() > stamp.tz_convert("America/New_York").date():
            continue
        try:
            shares = float(node.findtext("transactionAmounts/transactionShares/value", "nan"))
            raw_price = node.findtext("transactionAmounts/transactionPricePerShare/value")
            price = float(raw_price) if raw_price else None
        except (ValueError, TypeError):
            continue
        if not np.isfinite(shares) or shares <= 0 or (price is not None and (not np.isfinite(price) or price < 0)):
            continue
        value = shares * price if price is not None else None
        if value is not None and not np.isfinite(value):
            continue
        records.append({"key": f"{ticker}:{accession}:{index}", "ticker": ticker, "accession": accession, "cik": str(int(cik)),
            "transactionIndex": index, "transactionDate": transaction_date, "filingDate": filed[0].strip(), "acceptedAt": stamp.isoformat(),
            "insiderIds": ids, "insiderName": ", ".join(names)[:500], "position": ", ".join(positions)[:500],
            "shares": shares, "price": price, "value": value, "currency": "unverified", "transactionCode": "P", "form": "4",
            "securityTitle": node.findtext("securityTitle/value", "")[:300],
            "ownership": node.findtext("ownershipNature/directOrIndirectOwnership/value", "Unknown"),
            "filingUrl": filing_url, "sourceUrl": xml_url, "source": "SEC EDGAR", "discoveredBy": "API Ninjas",
            "filingFingerprint": hashlib.sha256(index_body + xml_body).hexdigest(),
            "firstRetrievedAt": retrieved_at, "usableForTraining": False})
    return records


def verify_filing(candidate, ticker, retrieved_at, user_agent):
    accession, cik = candidate.get("accession_number"), str(candidate.get("cik", ""))
    if (candidate.get("ticker", "").upper().replace(".", "-") != ticker or candidate.get("transaction_code") != "P"
            or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", str(accession)) or not re.fullmatch(r"\d{1,10}", cik)):
        raise ValueError("Invalid insider candidate identity")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    filing_url = base + accession + "-index.html"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    index_body = _download(filing_url, headers)
    tree = html.fromstring(index_body)
    links = tree.xpath('//table[contains(@class,"tableFile")]//tr[td[normalize-space(.)="4"]]//a/@href')
    files = [unquote(urlparse(link).path).split("/")[-1] for link in links]
    name = next((name for name in files if re.fullmatch(r"[A-Za-z0-9_.-]+\.xml", name)), None)
    if not name:
        raise ValueError("No Form 4 XML document found")
    xml_url = base + name
    return parse_filing(index_body, _download(xml_url, headers), ticker, cik, accession, filing_url, xml_url, retrieved_at)


def collect_insider_evidence(ticker, as_of, retrieved_at):
    ticker = _normalize_ticker(ticker)
    cutoff = execution_window(as_of)[0]
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=30)).date().isoformat()
    load_environment()
    retained = _store("query", "recent", {"ticker": ticker, "from": start + "T00:00:00+00:00", "to": cutoff.isoformat()})
    if not isinstance(retained, dict) or not isinstance(retained.get("rows"), list):
        raise RuntimeError("Convex did not return valid insider evidence.")
    rows = retained["rows"]
    key, agent = os.environ.get("API_NINJAS_API_KEY", "").strip(), _get_sec_user_agent()
    summary = {"status": "coverage_incomplete", "coverage": "partial", "message": "Free results do not establish complete coverage. Only verified non-derivative Form 4 purchases are shown; amendments are not reconciled.",
               "from": start, "to": cutoff.isoformat(), "retrievedAt": retrieved_at, "providerStatus": "not_configured",
               "candidatesReturned": 0, "filingsChecked": 0, "truncated": bool(retained.get("truncated")),
               "buyers": 0, "transactions": 0, "records": []}
    if key and agent:
        try:
            candidates = fetch_candidates(ticker, start, as_of, key)
            summary.update(providerStatus="ok", candidatesReturned=len(candidates), truncated=summary["truncated"] or len(candidates) >= 10)
            checked, fresh = set(), []
            for candidate in candidates:
                accession = candidate.get("accession_number")
                if not isinstance(accession, str) or accession in checked:
                    continue
                if len(checked) >= 3:
                    summary["truncated"] = True
                    break
                checked.add(accession)
                summary["filingsChecked"] += 1
                try:
                    verified = verify_filing(candidate, ticker, retrieved_at, agent)
                    fresh.extend(row for row in verified if start <= row["transactionDate"] <= as_of and start <= row["acceptedAt"][:10])
                except (PrimaryEvidenceError, ValueError, TypeError, etree.LxmlError):
                    summary["providerStatus"] = "verification_incomplete"
                    summary["message"] = "Some discovered filings could not be verified. Coverage is incomplete; no missing transaction is treated as absent."
            # Bound writes and retain a separate SEC row identity, not provider-row hashes.
            if len(fresh) > 150:
                fresh = fresh[:150]
                summary["truncated"] = True
            if fresh:
                receipt = _store("mutation", "save", {"records": fresh})
                if not isinstance(receipt, dict) or not isinstance(receipt.get("ids"), list) or len(receipt["ids"]) != len(fresh) or not all(isinstance(value, str) and value for value in receipt["ids"]):
                    raise RuntimeError("Convex did not confirm the insider transaction IDs.")
                rows = list({row["key"]: row for row in rows + fresh}.values())
        except InsiderSourceError as error:
            summary.update(providerStatus="unavailable", message=str(error))
    else:
        summary["message"] = "Configure API_NINJAS_API_KEY and SEC_USER_AGENT. Coverage is incomplete."
    rows = [row for row in rows if start <= row["transactionDate"] <= as_of and pd.Timestamp(row["acceptedAt"]) <= cutoff]
    rows.sort(key=lambda row: (row["acceptedAt"], row["key"]), reverse=True)
    summary.update(status="buying_observed" if rows else "coverage_incomplete",
                   buyers=len({owner for row in rows for owner in row["insiderIds"]}), transactions=len(rows),
                   records=[{field: row[field] for field in ("key", "insiderName", "securityTitle", "transactionDate", "acceptedAt", "shares", "price", "value", "filingUrl")} for row in rows[:5]])
    return summary
