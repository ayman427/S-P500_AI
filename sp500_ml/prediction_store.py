"""Server-only Convex prediction history and delayed market reconciliation.

Run ``python -m sp500_ml.prediction_store`` as a persistent hourly worker.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import yfinance as yf

from .sessions import _calendar, execution_window, utc_now, validate_signal_time

log = logging.getLogger(__name__)
_file_environment = {}
_SERVER_SETTINGS = {"CONVEX_URL", "PREDICTION_STORE_SECRET", "CLERK_JWT_ISSUER_DOMAIN", "CLERK_AUTHORIZED_PARTIES", "FINNHUB_API_KEY", "API_NINJAS_API_KEY", "SEC_USER_AGENT", "GEMINI_API_KEY", "GEMINI_ANALYSIS_MODEL"}


def load_environment():
    """Reload file-owned settings while preserving explicit process overrides."""
    path = Path(__file__).resolve().parent.parent / ".env.local"
    settings = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name in _SERVER_SETTINGS:
                settings[name] = value.strip().strip('"').strip("'")
    for name in _SERVER_SETTINGS:
        current = os.environ.get(name)
        if current is None or (name in _file_environment and current == _file_environment[name]):
            if name in settings:
                os.environ[name] = settings[name]
                _file_environment[name] = settings[name]
            else:
                os.environ.pop(name, None)
                _file_environment.pop(name, None)


load_environment()


def configured():
    load_environment()
    return bool(os.environ.get("CONVEX_URL") and os.environ.get("PREDICTION_STORE_SECRET"))


def call(kind, name, arguments, token=None):
    load_environment()
    url = os.environ["CONVEX_URL"].rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
        raise ValueError("CONVEX_URL must use HTTPS")
    response = requests.post(f"{url}/api/{kind}", headers={"Authorization": f"Bearer {token}"} if token else {}, json={
        "path": f"predictions:{name}", "args": {**arguments, "secret": os.environ["PREDICTION_STORE_SECRET"]},
        "format": "json",
    }, timeout=20)
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "success":
        raise RuntimeError("Convex prediction operation failed; check the deployment function logs")
    return body["value"]


def build_prediction(record, model_type, model_hash, now=None):
    now = utc_now() if now is None else now
    entry = validate_signal_time(record["as_of"], now)
    horizon = {"open_to_close": 1, "open_to_close_5d": 5}[record["target_key"]]
    calendar = _calendar(entry.year)
    exit_session = calendar.session_offset(entry, horizon - 1)
    signal_close, entry_at, _ = execution_window(record["as_of"])
    trained = record.get("model_trained_through")
    if not trained or pd.Timestamp(trained) > pd.Timestamp(record["as_of"]):
        raise ValueError("Model training provenance is missing or later than the signal")
    trained_at = pd.Timestamp(record.get("model_trained_at_utc"))
    if pd.isna(trained_at) or trained_at.tzinfo is None or trained_at > now:
        raise ValueError("Model creation timestamp is missing or invalid")
    prediction = float(record["predicted_return"])
    if not math.isfinite(prediction):
        raise ValueError("Prediction must be finite")
    identity = [record["ticker"], model_type, model_hash, record["target_key"], record["as_of"]]
    prediction_record = dict(key=hashlib.sha256(json.dumps(identity).encode()).hexdigest(), ticker=record["ticker"],
                modelType=model_type, modelHash=model_hash, targetKey=record["target_key"],
                signalDate=record["as_of"], predictedReturn=prediction, trainedThrough=trained,
                signalClose=signal_close.timestamp() * 1000, entryDate=entry.date().isoformat(),
                exitDate=exit_session.date().isoformat(), entryAt=entry_at.timestamp() * 1000,
                exitAt=calendar.session_close(exit_session).timestamp() * 1000)
    if model_type == "hybrid":
        prediction_record["hybrid"] = record["hybrid"]
    return prediction_record


def apply_hybrid_feedback(record, fingerprint, token):
    """Adapt only from this owner's genuinely issued, settled component forecasts."""
    if not token:
        from .auth import AuthenticationError
        raise AuthenticationError()
    identity = [record["ticker"], "hybrid", fingerprint, record["target_key"], record["as_of"]]
    cutoff = execution_window(record["as_of"])[0].timestamp() * 1000
    result = call("query", "hybridFeedback", {"ticker": record["ticker"], "targetKey": record["target_key"],
        "modelHash": fingerprint, "key": hashlib.sha256(json.dumps(identity).encode()).hexdigest(), "signalClose": cutoff}, token=token)
    existing = result.get("existing")
    if existing:
        if not existing.get("hybrid") or existing.get("signalDate") != record["as_of"]:
            raise RuntimeError("Saved hybrid forecast provenance could not be verified.")
        return {**record, "predicted_return": existing["predictedReturn"], "hybrid": existing["hybrid"]}
    observations = sorted(result["observations"], key=lambda row: row["entryAt"])
    independent, last_exit = [], float("-inf")
    now_ms = utc_now().timestamp() * 1000
    for row in observations:
        h = row.get("hybrid", {})
        if (row.get("modelHash") != fingerprint or row.get("status") != "settled" or row.get("ticker") != record["ticker"]
            or row.get("targetKey") != record["target_key"] or row["exitAt"] > cutoff or row.get("settledAt", float("inf")) > now_ms
            or not all(math.isfinite(value) for value in (row["actualReturn"], h.get("ridgeReturn", float("nan")), h.get("boostingReturn", float("nan"))))):
            continue
        if row["entryAt"] >= last_exit:
            independent.append(row)
            last_exit = row["exitAt"]
    independent = independent[-60:]
    h = {**record["hybrid"]}
    if independent:
        ridge_error = sum(abs(row["actualReturn"] - row["hybrid"]["ridgeReturn"]) for row in independent) / len(independent)
        boosting_error = sum(abs(row["actualReturn"] - row["hybrid"]["boostingReturn"]) for row in independent) / len(independent)
        scale = max((ridge_error + boosting_error) / 2, 1e-9)
        ridge_score, boosting_score = math.exp(-ridge_error / scale), math.exp(-boosting_error / scale)
        learned_weight = ridge_score / (ridge_score + boosting_score)
        confidence = len(independent) / (len(independent) + 10)
        h["ridgeWeight"] = (1-confidence)*h["ridgeWeight"] + confidence*learned_weight
        h["feedbackSamples"] = len(independent)
        h["feedbackThrough"] = max(row["settledAt"] for row in independent)
    prediction = h["ridgeWeight"]*h["ridgeReturn"] + (1-h["ridgeWeight"])*h["boostingReturn"]
    return {**record, "predicted_return": prediction, "hybrid": h}


def track(ticker, model_type, target_key, token=None):
    if not token:
        from .auth import AuthenticationError
        raise AuthenticationError()
    if not configured():
        return {"status": "unconfigured", "message": "Live prediction history needs a Convex connection.", "rows": []}
    from .training import _load_latest_prediction
    message = "Prediction saved before the market opened."
    record = None
    try:
        record, fingerprint = _load_latest_prediction(ticker, model_type, target_key)
        if model_type == "hybrid":
            record = apply_hybrid_feedback(record, fingerprint, token)
        prediction = build_prediction(record, model_type, fingerprint)
    except ValueError as error:
        prediction = None
        message = f"Not recorded as a live prediction: {error}"
        if record is not None and "signal session closes" in str(error):
            message = (f"Not saved: this forecast uses prices through {record['as_of']}, "
                       "and its recording window has passed. Update market data, then search again "
                       "after the latest session closes and before the next session opens.")
    receipt = None
    if prediction is not None:
        receipt = call("mutation", "record", {"prediction": prediction}, token=token)
        if not isinstance(receipt, dict) or not receipt.get("id") or not isinstance(receipt.get("created"), bool):
            raise RuntimeError("Database did not provide a valid save confirmation")
    rows = call("query", "history", {"ticker": ticker, "modelType": model_type, "targetKey": target_key}, token=token)
    # A forecast already saved overnight remains confirmed after the market opens.
    # Do not attempt a new, late insertion just to redisplay its receipt.
    if receipt is None and record is not None:
        identity = [record["ticker"], model_type, fingerprint, record["target_key"], record["as_of"]]
        existing_key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        existing = next((row for row in rows if row.get("key") == existing_key
                         and row.get("predictedReturn") == record.get("predicted_return")), None)
        if existing is not None:
            prediction = {"key": existing_key}
            receipt = {"id": existing["_id"], "created": False}
    confirmed = bool(receipt and any(row.get("_id") == receipt["id"] and row.get("key") == prediction["key"] for row in rows))
    if receipt and not confirmed:
        raise RuntimeError("Prediction write could not be confirmed by reading it back")
    if confirmed:
        message = "Prediction saved and verified in Convex." if receipt["created"] else "This prediction is already saved in Convex. No duplicate was added."
    deployment = urlparse(os.environ["CONVEX_URL"]).hostname
    response = {"status": "connected", "deployment": deployment,
            "saved": confirmed, "recordId": receipt["id"] if confirmed else None,
            "created": receipt["created"] if confirmed else False,
            "message": f"Convex ({deployment}). {message}", "rows": rows}
    if model_type == "hybrid" and confirmed:
        response["forecast"] = record
    return response


def my_predictions(token, cursor=None):
    if not token:
        from .auth import AuthenticationError
        raise AuthenticationError()
    if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 8192):
        raise ValueError("Invalid prediction history cursor")
    result = call("query", "mine", {"paginationOpts": {"numItems": 20, "cursor": cursor}}, token=token)
    return {"rows": result["page"], "hasMore": not result["isDone"],
            "cursor": result["continueCursor"] if not result["isDone"] else None}


def reconcile_once():
    if not configured():
        raise RuntimeError("Set CONVEX_URL and PREDICTION_STORE_SECRET before starting the comparison worker")
    count = 0
    for row in call("query", "due", {}):
        try:
            start, end = row["entryDate"], row["exitDate"]
            prices = yf.Ticker(row["ticker"]).history(start=start,
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(), auto_adjust=True)
            if prices.empty:
                continue
            prices.index = pd.Index([stamp.date().isoformat() for stamp in prices.index])
            if prices.index.has_duplicates:
                raise ValueError("Duplicate market price dates")
            if start not in prices.index or end not in prices.index:
                continue  # Never substitute another session for missing data.
            opening, closing = float(prices.loc[start, "Open"]), float(prices.loc[end, "Close"])
            if not all(math.isfinite(x) and x > 0 for x in (opening, closing)):
                raise ValueError("Invalid market prices")
            call("mutation", "settle", {"id": row["_id"], "entryDate": start, "exitDate": end,
                                       "entryOpen": opening, "exitClose": closing})
            count += 1
        except Exception:
            log.exception("Could not reconcile prediction %s", row["_id"])
    return count


def worker():
    while True:
        try:
            reconcile_once()
        except Exception:
            log.exception("Prediction comparison sync failed; retrying in an hour")
        time.sleep(3600)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not configured():
        raise SystemExit("Set CONVEX_URL and PREDICTION_STORE_SECRET first")
    if args.once:
        print(f"Settled {reconcile_once()} predictions")
    else:
        worker()
