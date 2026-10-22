"""Completed-session price/volume observations, never a manipulation verdict."""
import math

import pandas as pd
import requests

from .cloud_data import read_csv
from .data import _drop_incomplete_session, _ticker_file
from .sessions import _calendar
from .spikes import read_spikes
from .rise_risk import price_outcome

METHOD = "price_volume_observations_v1"


def analyze_event(event, history, market):
    day = pd.Timestamp(event["date"])
    calendar = _calendar(day.year)
    dates = calendar.sessions_in_range(day - pd.Timedelta(days=180), day)[-62:]
    raw = pd.DataFrame(history).set_index("date")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.reindex(dates[dates >= raw.index.min()])
    close = pd.to_numeric(raw.close, errors="coerce")
    past = close.iloc[:-1]
    returns = past.pct_change(fill_method=None).iloc[1:]
    valid = len(returns) >= 20 and returns.notna().all() and (past > 0).all()
    mean, std = (float(returns.mean()), float(returns.std())) if valid else (None, None)
    threshold = max(.05, mean + 3 * std) if valid else None
    volume = pd.to_numeric(raw.get("volume", pd.Series(index=raw.index, dtype=float)), errors="coerce")
    baseline = volume.iloc[:-1].tail(20)
    ratio = float(volume.iloc[-1] / baseline.mean()) if len(baseline) == 20 and baseline.notna().all() and (baseline >= 0).all() and baseline.mean() > 0 and pd.notna(volume.iloc[-1]) and volume.iloc[-1] >= 0 else None
    acceleration = None
    if len(close) >= 11 and close.iloc[-11:].notna().all():
        latest = float(close.iloc[-1] / close.iloc[-6] - 1)
        earlier = float(close.iloc[-6] / close.iloc[-11] - 1)
        acceleration = latest - earlier
    previous = calendar.previous_session(day).date().isoformat()
    market_return = None
    if event["date"] in market and previous in market and market[previous] > 0:
        market_return = market[event["date"]] / market[previous] - 1
    excess = event["return"] - market_return if market_return is not None else None
    def signal(key, label, value, limit, unit):
        return {"id": key, "label": label, "value": value, "threshold": limit, "unit": unit,
                "status": "unavailable" if value is None or limit is None else "elevated" if value >= limit - 1e-12 else "within_rule"}
    signals = [signal("price", "Price vs normal", event["return"] if valid else None, threshold, "return"),
               signal("market", "Above SPY", excess, .03, "percentage_points"),
               signal("volume", "Trading volume", ratio, 2., "multiple"),
               signal("acceleration", "5-session acceleration", acceleration, .05, "percentage_points")]
    reasons = []
    if signals[0]["status"] == "elevated": reasons.append("Price rose beyond its prior trading range")
    if signals[1]["status"] == "elevated": reasons.append("outperformed SPY")
    if signals[2]["status"] == "elevated": reasons.append("volume was unusually high")
    if signals[3]["status"] == "elevated": reasons.append("the rise accelerated")
    closes = {row["date"]: float(row["close"]) for row in history if math.isfinite(float(row["close"])) and row["close"] > 0}
    observed, _, decline, _ = price_outcome(event, closes)
    gain = event["close"] - event["previousClose"]
    return {"method": METHOD, "asOf": event["date"], "signals": signals,
            "title": "Unusual rise detected" if signals[0]["status"] == "elevated" else "Price rise - incomplete baseline",
            "summary": "; ".join(reasons) + "." if reasons else "Not enough aligned data to characterize this rise.",
            "baselineFrom": past.index[0].date().isoformat() if len(past) else None,
            "baselineThrough": previous, "baselineReturns": len(returns) if valid else 0,
            "priorDailyVolatility": std, "marketReturn": market_return,
            "historicalOutcome": {"sessions": len(observed), "through": observed[-1][0] if observed else event["date"],
                "return5": observed[4][1] / event["close"] - 1 if len(observed) >= 5 else None,
                "return20": observed[19][1] / event["close"] - 1 if len(observed) >= 20 else None,
                "largestDecline": decline if observed else None,
                "gainReversed": max(0., (event["close"] - observed[-1][1]) / gain) if observed and gain > 0 else None}}


def read_anomalies(ticker, years=5):
    if years not in (0, 1, 3, 5, 10):
        raise ValueError("Unsupported history period.")
    result = read_spikes(ticker, years=0)
    history = result["history"]
    market = {}
    try:
        prices = _drop_incomplete_session(read_csv(_ticker_file("SPY"), parse_dates=["Date"], index_col="Date"))
        market = {day.date().isoformat(): float(value) for day, value in prices.Close.items() if pd.notna(value) and math.isfinite(float(value)) and value > 0}
    except (OSError, ValueError, RuntimeError, requests.RequestException):
        pass
    cutoff = (pd.Timestamp(result["asOf"]) - pd.DateOffset(years=years)).date().isoformat() if years else result["from"]
    events = []
    for event in result["events"]:
        if event["date"] < cutoff:
            continue
        calendar = _calendar(pd.Timestamp(event["date"]).year)
        if not calendar.is_session(event["date"]) or calendar.previous_session(event["date"]).date().isoformat() != event["previousDate"]:
            continue
        events.append({**event, "analysis": analyze_event(event, history, market)})
    visible = [row for row in history if row["date"] >= cutoff]
    return {**result, "from": visible[0]["date"], "history": visible, "events": events, "evaluation": evaluate_outcomes(events)}


def evaluate_outcomes(events):
    """Descriptive, non-overlapping outcome counts, not a fitted risk model."""
    result = {"eligible": 0, "flagged": 0, "flaggedReversals": 0, "baselineReversals": 0, "excluded": 0}
    cutoff = "9999-12-31"
    for event in sorted(events, key=lambda row: row["date"], reverse=True):
        a = event["analysis"]
        outcome = a["historicalOutcome"]
        if outcome["sessions"] != 20 or outcome["through"] >= cutoff or any(s["status"] == "unavailable" for s in a["signals"]):
            result["excluded"] += 1
            continue
        cutoff = event["date"]
        result["eligible"] += 1
        reversed_ = outcome["largestDecline"] <= -.1 + 1e-12
        result["baselineReversals"] += int(reversed_)
        flagged = sum(s["status"] == "elevated" for s in a["signals"] if s["id"] != "price") >= 2
        result["flagged"] += int(flagged)
        result["flaggedReversals"] += int(flagged and reversed_)
    return result
