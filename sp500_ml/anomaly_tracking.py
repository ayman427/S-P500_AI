"""Prospective frozen-input observations; cloud-only, no historical backfill."""
import hashlib
import inspect
import json
import math
import os
import threading

import pandas as pd
import requests

from .anomaly_evaluation import TICKERS
from .cloud_data import read_csv
from .data import _drop_incomplete_session, _ticker_file
from .prediction_store import load_environment
from .search_prices import latest_completed_session
from .sessions import _calendar, execution_window, utc_now
from .spike_anomalies import METHOD, analyze_event
from .spike_rules import spike_signals
from .spikes import read_spikes

PROTOCOL = "price_volume_prospective_v1"
_completed = set()


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def protocol_spec():
    return {"method": METHOD, "tickers": TICKERS, "calendar": "XNYS", "horizonSessions": 20,
            "declineThreshold": -.1, "multiSignalMinimum": 2, "volumeThreshold": 2., "marketThreshold": .03,
            "accelerationThreshold": .05, "codeSha256": digest(inspect.getsource(analyze_event)+inspect.getsource(spike_signals))}


def rpc(kind, function, args):
    load_environment()
    from urllib.parse import urlparse
    url = os.environ.get("CONVEX_URL", "").rstrip("/")
    if urlparse(url).scheme != "https":
        raise RuntimeError("Prospective tracking requires a configured HTTPS Convex deployment.")
    try:
        response = requests.post(url+"/api/"+kind, json={"path":"anomalyTracking:"+function,
            "args":{**args,"secret":os.environ["PREDICTION_STORE_SECRET"]},"format":"json"}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise ValueError("Unconfirmed database operation")
        return payload["value"]
    except (requests.RequestException, ValueError, KeyError):
        raise RuntimeError("Prospective tracking database operation failed; no write was confirmed.") from None


def capture_record(ticker, event, history, market, protocol, now):
    close_at, next_open, _ = execution_window(event["date"])
    if protocol["frozenAt"] > close_at.timestamp()*1000:
        return None
    if not close_at <= now < next_open:
        return None
    day = pd.Timestamp(event["date"])
    calendar = _calendar(day.year)
    dates = calendar.sessions_in_range(day-pd.Timedelta(days=180),day)[-62:]
    stock = {r["date"]:r for r in history if r["date"] <= event["date"]}
    dates = dates[dates >= pd.Timestamp(min(stock))]
    snapshot = {"stock":[stock.get(d.date().isoformat(),{"date":d.date().isoformat(),"close":None,"volume":None}) for d in dates],
                "market":[{"date":d,"close":market.get(d)} for d in (event["previousDate"],event["date"])]}
    # Analyze exactly the captured window, excluding every later price.
    if any(r["close"] is None for r in snapshot["stock"]):
        return None
    a = analyze_event(event,snapshot["stock"],market)
    signals = [{k:s[k] for k in ("id","value","threshold","status")} for s in a["signals"]]
    if signals[0]["status"] != "elevated":
        return None
    additional = signals[1:]
    flagged = None if any(s["status"] == "unavailable" for s in additional) else sum(s["status"] == "elevated" for s in additional) >= 2
    due = calendar.sessions_in_range(day,day+pd.Timedelta(days=60))[20]
    text = packed(snapshot)
    return {"key":f"{PROTOCOL}:{ticker}:{event['date']}","protocol":PROTOCOL,"protocolHash":protocol["hash"],"ticker":ticker,
            "eventDate":event["date"],"closeAt":close_at.timestamp()*1000,"nextOpenAt":next_open.timestamp()*1000,
            "dueDate":due.date().isoformat(),"dueAt":(calendar.session_close(due)+pd.Timedelta(minutes=30)).timestamp()*1000,
            "eventClose":event["close"],"inputSnapshot":text,"inputSha256":digest(text),"signals":signals,"flagged":flagged}


def outcome_snapshot(record, prices, now):
    if now.timestamp()*1000 < record["dueAt"]:
        return None
    calendar = _calendar(pd.Timestamp(record["eventDate"]).year)
    dates = calendar.sessions_in_range(record["eventDate"],record["dueDate"])
    if len(dates) != 21:
        raise ValueError("Invalid prospective outcome horizon")
    prices = _drop_incomplete_session(prices, now=now)
    close = prices.Close.reindex(dates)
    if len(close) != 21 or not close.map(lambda x: pd.notna(x) and math.isfinite(float(x)) and x > 0).all():
        return None
    # Ratios use the contemporaneous adjusted event close, avoiding split-scale mismatches.
    return packed([{"date":d.date().isoformat(),"close":float(v)} for d,v in close.items()])


def settle_due(now):
    cursor, count, unavailable = None, 0, 0
    while True:
        page = rpc("query","due",{"cursor":cursor})
        for record in page["page"]:
            try:
                prices = read_csv(_ticker_file(record["ticker"]),parse_dates=["Date"],index_col="Date")
                text = outcome_snapshot(record,prices,now)
                if text is not None:
                    receipt = rpc("mutation","settle",{"key":record["key"],"outcomeSnapshot":text,"outcomeSha256":digest(text)})
                    if not receipt.get("id"):
                        raise RuntimeError("Settlement was not confirmed")
                    count += bool(receipt.get("settled"))
                else:
                    unavailable += 1
            except (ValueError, OSError, RuntimeError, requests.RequestException):
                unavailable += 1
                continue
        if page["isDone"]:
            return {"settled":count,"unavailable":unavailable}
        cursor = page["continueCursor"]


def run_once(now=None):
    now = utc_now() if now is None else pd.Timestamp(now)
    spec = packed(protocol_spec())
    protocol = rpc("mutation","register",{"key":PROTOCOL,"hash":digest(spec),"spec":spec})
    if protocol.get("hash") != digest(spec):
        raise RuntimeError("Frozen protocol mismatch")
    settlement = settle_due(now)
    summary = {"settled":settlement["settled"],"settlementPending":settlement["unavailable"],"captured":0,"statuses":{}}
    day = latest_completed_session(now).date().isoformat()
    close_at, next_open, _ = execution_window(day)
    market = None
    for ticker in TICKERS:
        key = (ticker,day)
        if key in _completed:
            continue
        try:
            if protocol["frozenAt"] > close_at.timestamp()*1000:
                status = "not_active"
            elif not close_at <= now < next_open:
                status = "missed_window"
            else:
                if market is None:
                    try:
                        prices = _drop_incomplete_session(read_csv(_ticker_file("SPY"),parse_dates=["Date"],index_col="Date"),now=now)
                        market = {d.date().isoformat():float(v) for d,v in prices.Close.items() if pd.notna(v) and math.isfinite(float(v)) and v>0}
                    except (ValueError, OSError, RuntimeError, requests.RequestException):
                        market = {}
                result = read_spikes(ticker,0)
                if result["asOf"] != day:
                    status = "stale"
                else:
                    event = next((e for e in result["events"] if e["date"] == day),None)
                    if event is None:
                        status = "no_spike"
                    else:
                        record = capture_record(ticker,event,result["history"],market,protocol,now)
                        if record is None:
                            status = "error"
                        else:
                            receipt = rpc("mutation","capture",{"record":record})
                            if not receipt.get("id"):
                                raise RuntimeError("Capture was not confirmed")
                            summary["captured"] += bool(receipt.get("created"))
                            status = "captured"
            receipt = rpc("mutation","checkpoint",{"protocol":PROTOCOL,"ticker":ticker,"session":day,"status":status})
            if not receipt.get("id"):
                raise RuntimeError("Coverage checkpoint was not confirmed")
            if status in {"captured","no_spike","not_active","missed_window"}:
                _completed.add(key)
        except (ValueError, OSError, RuntimeError, requests.RequestException):
            status = "error"
            try:
                rpc("mutation","checkpoint",{"protocol":PROTOCOL,"ticker":ticker,"session":day,"status":status})
            except RuntimeError:
                pass
        summary["statuses"][status] = summary["statuses"].get(status,0)+1
    return summary


def worker():
    pause = threading.Event()
    while True:
        try:
            result = run_once()
            if result["captured"] or result["settled"] or result["settlementPending"] or result["statuses"].get("error"):
                print("[rise-prospective] "+packed(result),flush=True)
        except Exception:
            print("[rise-prospective] Tracking unavailable; no prospective save confirmed. Retrying later.",flush=True)
        pause.wait(900)


if __name__ == "__main__":
    print(packed(run_once()))
