"""Refresh -> completed prices -> shared rule -> confirmed Convex snapshot."""
import hashlib
import json
import os
from urllib.parse import urlparse

import requests

from .cloud_data import read_csv
from .data import _normalize_ticker, _ticker_file
from .prediction_store import load_environment, configured
from .search_prices import refresh_search_prices, latest_completed_session
from .sessions import utc_now
from .spike_checklist import evaluate_checklist
from .spike_rules import prepare_prices
from .insider_trading import collect_insider_evidence


def save_snapshot(record):
    load_environment()
    if not configured():
        raise RuntimeError("Convex watch storage is not configured.")
    url = os.environ["CONVEX_URL"].rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("CONVEX_URL must use HTTPS")
    try:
        response = requests.post(url + "/api/mutation", json={"path": "spikeWatch:save", "format": "json",
            "args": {"secret": os.environ["PREDICTION_STORE_SECRET"], "record": record}}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "success" or not isinstance(payload.get("value"), dict) or not payload["value"].get("id"):
            raise ValueError("Unconfirmed snapshot")
        return payload["value"]["id"]
    except (requests.RequestException, ValueError, KeyError):
        raise RuntimeError("Convex did not confirm the watch snapshot. No current warning was published.") from None


def run_watch(ticker):
    ticker = _normalize_ticker(ticker)
    refresh_search_prices(ticker, include_market_inputs=False)
    now = utc_now()
    expected = latest_completed_session(now)
    prices = prepare_prices(read_csv(_ticker_file(ticker), parse_dates=["Date"], index_col="Date"))
    prices = prices.loc[prices.index <= expected]
    result = evaluate_checklist(prices, now=now, expected_session=expected)
    result["insiderEvidence"] = collect_insider_evidence(ticker, result["asOf"], now.isoformat())
    insider = next(check for check in result["checks"] if check["id"] == "insiders")
    insider.update(source="API Ninjas / SEC EDGAR", rule="Observed purchases; excluded from alert decisions pending validation")
    fingerprint = hashlib.sha256(prices.to_csv(index=True).encode()).hexdigest()
    evidence_fingerprint = hashlib.sha256(json.dumps({key: value for key, value in result["insiderEvidence"].items() if key != "retrievedAt"}, sort_keys=True).encode()).hexdigest()
    record = {**result, "ticker": ticker, "evaluatedAt": now.isoformat(), "priceFingerprint": fingerprint,
              "evidenceFingerprint": evidence_fingerprint,
              "key": f"{ticker}:{result['expectedSession']}:{result['ruleVersion']}:{fingerprint}:{evidence_fingerprint}"}
    record_id = save_snapshot(record)
    return {**record, "saved": True, "recordId": record_id}
