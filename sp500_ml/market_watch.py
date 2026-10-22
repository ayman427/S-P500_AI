"""Shared, resumable S&P 500 checklist scan; every checkpoint lives in Convex."""
import os
import time
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from urllib.parse import urlparse

import requests
import pandas as pd

from .cloud_data import exclusive_cloud_operation, _lease
from .data import get_sp500_tickers, _download_batch, _extract_ticker
from .prediction_store import load_environment
from .search_prices import latest_completed_session
from .sessions import utc_now
from .spike_checklist import evaluate_checklist, RULE_VERSION


def _rpc(kind, name, **args):
    load_environment()
    url = os.environ.get("CONVEX_URL", "").rstrip("/")
    if urlparse(url).scheme != "https" or not os.environ.get("PREDICTION_STORE_SECRET"):
        raise RuntimeError("Convex market scan storage is not configured.")
    try:
        response = requests.post(url + "/api/" + kind, json={"path": "marketWatch:" + name,
            "format": "json", "args": {"secret": os.environ["PREDICTION_STORE_SECRET"], **args}}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "success" or "value" not in payload:
            raise ValueError("Unconfirmed market scan")
        if kind == "mutation" and (not isinstance(payload["value"], dict) or not payload["value"].get("id")):
            raise ValueError("Unconfirmed market scan write")
        return payload["value"]
    except (requests.RequestException, ValueError, KeyError):
        raise RuntimeError("Market scan storage is temporarily unavailable.") from None



_scan_cache = None
_scan_cache_lock = threading.Lock()
BATCH_SIZE = 15


def _remember_scan(scan, expected):
    global _scan_cache
    with _scan_cache_lock:
        _scan_cache = (os.environ.get("CONVEX_URL"), os.environ.get("PREDICTION_STORE_SECRET"), expected, time.monotonic(), deepcopy(scan))


def _forget_scan():
    global _scan_cache
    with _scan_cache_lock:
        _scan_cache = None


def _latest(expected):
    load_environment()
    with _scan_cache_lock:
        cached = _scan_cache
        if cached and cached[:3] == (os.environ.get("CONVEX_URL"), os.environ.get("PREDICTION_STORE_SECRET"), expected):
            scan = cached[4]
            ttl = 3600 if scan and scan.get("complete") and scan.get("expectedSession") == expected else 2
            if time.monotonic() - cached[3] < ttl:
                return deepcopy(scan)
    scan = _rpc("query", "latest")
    _remember_scan(scan, expected)
    return scan


def read_scan():
    expected = latest_completed_session(utc_now()).date().isoformat()
    return {"scan": _latest(expected), "expectedSession": expected}


def scan_batch(restart=False):
    if restart:
        _forget_scan()
    try:
        return _scan_batch(restart)
    except RuntimeError as error:
        if "holds this cloud lock" in str(error):
            _forget_scan()
            return {**read_scan(), "busy": True}
        raise


def _scan_stock(ticker, now, expected):
    # The scanner needs recent bars, not a download/rewrite of every cloud history.
    start = (expected - pd.Timedelta(days=180)).date().isoformat()
    prices = _extract_ticker(_download_batch([ticker], start), ticker)
    prices = prices.loc[prices.index <= expected]
    if prices.empty or prices.index.max() != expected:
        raise RuntimeError("Latest completed scan prices are unavailable.")
    # Only current conditions are ranked; historical validation is not a probability forecast.
    result = evaluate_checklist(prices.loc[prices.index <= expected].tail(80), now=now, expected_session=expected)
    checks = [{"id": c["id"], "status": c["status"], "value": c["value"]}
              for c in result["checks"] if c["active"]]
    return {"ticker": ticker, "asOf": result["asOf"], "status": result["status"],
            "met": sum(c["status"] == "met" for c in checks), "checks": checks}


def _scan_batch(restart=False):
    expected_day = latest_completed_session(utc_now()).date().isoformat()
    cached = _latest(expected_day)
    if not restart and cached and cached["complete"] and cached["expectedSession"] == expected_day:
        return {"scan": cached, "expectedSession": expected_day}
    with exclusive_cloud_operation("market-checklist"):
        now = utc_now()
        expected = latest_completed_session(now)
        current = _rpc("query", "latest")
        if current is None or current["expectedSession"] != expected.date().isoformat() or (restart and current["complete"]):
            tickers = sorted(set(get_sp500_tickers()))
            if not 400 <= len(tickers) <= 600:
                raise RuntimeError("The S&P 500 constituent list could not be verified.")
            current = {"expectedSession": expected.date().isoformat(), "startedAt": now.isoformat(),
                "updatedAt": now.isoformat(), "ruleVersion": RULE_VERSION, "universe": tickers,
                "processed": 0, "complete": False, "rows": [], "unavailable": []}
        if current["complete"]:
            return {"scan": current, "expectedSession": expected.date().isoformat()}
        # Convex's numeric values can arrive as floats, including integral counters.
        current["processed"] = int(current["processed"])
        batch = current["universe"][current["processed"]:current["processed"] + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="checklist") as pool:
            # Contexts carry the cloud lease into each worker's price writes.
            futures = [pool.submit(copy_context().run, _scan_stock, ticker, now, expected) for ticker in batch]
            # Downloads overlap; ordered commits preserve the resumable universe prefix.
            for ticker, future in zip(batch, futures):
                try:
                    current["rows"].append(future.result())
                except Exception:
                    current["unavailable"].append(ticker)
                current["processed"] += 1
                current["updatedAt"] = utc_now().isoformat()
                current["complete"] = current["processed"] == len(current["universe"])
            _rpc("mutation", "save", scan=current, lease=_lease.get())
            _remember_scan(current, expected.date().isoformat())
        return {"scan": current, "expectedSession": expected.date().isoformat()}


def worker():
    """Local dashboard worker also discovers stocks when nobody is searching."""
    while True:
        try:
            result = scan_batch()
            delay = 3600 if result["scan"]["complete"] else 2
        except Exception:
            delay = 60
        time.sleep(delay)


if __name__ == "__main__":
    worker()
