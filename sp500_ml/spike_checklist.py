"""One versioned price/volume rule, evaluated against disjoint future windows."""
import numpy as np
import pandas as pd
import exchange_calendars as xcals

from .spike_rules import prepare_prices, spike_signals, DETECTOR_VERSION
from .search_prices import latest_completed_session

RULE_VERSION = "price_volume_v2"
HORIZON = 5
# No production warning until this candidate passes independent validation.
RULE_APPROVED = False


def _trial_summary(indices, target):
    selected, next_allowed = [], -1
    for index in indices:
        if index >= next_allowed:
            selected.append(index)
            next_allowed = index + HORIZON
    hits = sum(bool(target.iloc[index]) for index in selected)
    count = len(selected)
    return {"trials": count, "hits": hits, "falseAlerts": count - hits,
            "precision": hits / count if count else None}


def evaluate_checklist(prices, now=None, expected_session=None):
    prices = prepare_prices(prices)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    expected = latest_completed_session(now) if expected_session is None else pd.Timestamp(expected_session)
    # A future or unfinished bar must never contribute to a current watch snapshot.
    prices = prices.loc[prices.index <= expected]
    if len(prices) < 2:
        raise ValueError("Not enough completed price history for the checklist.")
    calendar = xcals.get_calendar("XNYS", start=prices.index[0] - pd.Timedelta(days=1), end=expected + pd.Timedelta(days=1))
    sessions = calendar.sessions_in_range(prices.index[0], prices.index[-1])
    prices = prices.reindex(sessions)
    if len(prices) < 2:
        raise ValueError("Not enough exchange-session history for the checklist.")
    close, volume = prices.Close, prices.Volume
    volume_base = volume.shift(3).rolling(20, min_periods=20).mean()
    volume_ratio = volume.rolling(3, min_periods=3).mean() / volume_base.where(volume_base > 0)
    volume_ratio = volume_ratio.where(np.isfinite(volume_ratio))
    momentum = close.pct_change(5, fill_method=None)
    breakout = close / close.shift(1).rolling(20, min_periods=20).max() - 1
    metrics = [("volume", "Volume buildup", volume_ratio, 1.5, "3-session average >= 1.5x preceding 20 sessions", "ratio"),
               ("momentum", "Positive momentum", momentum, .02, "5-session price gain >= 2%", "return"),
               ("breakout", "Closing-price breakout", breakout, 0., "Close above the preceding 20-session high", "return")]
    known = volume_ratio.notna() & momentum.notna() & breakout.notna() & (close.notna().rolling(26, min_periods=26).sum() == 26)
    triggered = known & (volume_ratio >= 1.5) & (momentum >= .02) & (breakout > 0)
    returns, thresholds, spike = spike_signals(close)
    target = pd.concat([spike.shift(-day) for day in range(1, HORIZON + 1)], axis=1).fillna(False).any(axis=1)
    split = int(len(close) * .7)
    eligible = known & thresholds.notna() & pd.Series(np.arange(len(close)) >= split, index=close.index)
    future_known = pd.concat([(returns.notna() & thresholds.notna()).shift(-day, fill_value=False) for day in range(1, HORIZON + 1)], axis=1).all(axis=1)
    eligible &= future_known
    eligible.iloc[max(0, len(close) - HORIZON):] = False
    eligible_indices = np.flatnonzero(eligible.to_numpy())
    alerts = _trial_summary(np.flatnonzero((eligible & triggered).to_numpy()), target)
    ordinary = _trial_summary(eligible_indices, target)
    midpoint = split + (len(close) - HORIZON - split) // 2
    folds = []
    for lower, upper in ((split, midpoint), (midpoint, len(close) - HORIZON)):
        # Purge fold boundaries so a first-fold outcome cannot extend into the next fold.
        positions = np.flatnonzero((eligible & triggered).to_numpy())
        folds.append(_trial_summary([i for i in positions if lower <= i and i + HORIZON < upper], target))
    baseline_rate = ordinary["precision"]
    precision = alerts["precision"]
    lift = precision / baseline_rate if precision is not None and baseline_rate else None
    count = alerts["trials"]
    if count:
        z = 1.96
        lower_bound = (precision + z*z/(2*count) - z*np.sqrt(precision*(1-precision)/count + z*z/(4*count*count))) / (1+z*z/count)
    else:
        lower_bound = None
    sufficient = len(eligible_indices) >= 252 and count >= 20 and all(fold["trials"] >= 10 for fold in folds)
    supported = bool(sufficient and alerts["hits"] >= 5 and lift is not None and lift >= 2
        and lower_bound > baseline_rate and all(fold["hits"] >= 3 and fold["precision"] >= 2*baseline_rate for fold in folds))
    checks = [{"id": key, "label": label, "rule": rule, "unit": unit, "source": "Adjusted daily prices", "active": True,
               "availableAt": close.index[-1].date().isoformat(),
               "value": float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else None,
               "status": "unknown" if pd.isna(series.iloc[-1]) else "met" if
                   (series.iloc[-1] > threshold if key == "breakout" else series.iloc[-1] >= threshold) else "not_met"}
              for key, label, series, threshold, rule, unit in metrics]
    for key, label in (("insiders", "Disclosed insider buying"), ("catalyst", "Upcoming earnings / announcements")):
        checks.append({"id": key, "label": label, "rule": "Awaiting dated evidence integration", "unit": "none", "source": "Not connected", "active": False,
                       "availableAt": None, "value": None, "status": "unknown"})
    from .sessions import _calendar
    calendar = _calendar(expected.year)
    recent_sessions = calendar.sessions_window(expected, -25)
    missing = [day.date().isoformat() for day in recent_sessions if day not in close.index or pd.isna(close.loc[day])]
    stale = close.index[-1] != expected
    complete = bool(known.iloc[-1]) and not missing
    status = "stale" if stale else "incomplete" if not complete else "caution" if bool(triggered.iloc[-1]) and supported and RULE_APPROVED else "watch" if bool(triggered.iloc[-1]) else "no_signal"
    return {"asOf": close.index[-1].date().isoformat(), "expectedSession": expected.date().isoformat(), "status": status, "checks": checks,
            "ruleVersion": RULE_VERSION, "detectorVersion": DETECTOR_VERSION, "horizonSessions": HORIZON, "usableForTraining": False,
            "missingSessions": missing, "ruleApproved": RULE_APPROVED,
            "validation": {"from": close.index[split].date().isoformat(), "sessions": len(eligible_indices), "alerts": count,
                           "hits": alerts["hits"], "falseAlerts": alerts["falseAlerts"], "precision": precision,
                           "baselineRate": baseline_rate, "baselineTrials": ordinary["trials"], "lift": lift, "precisionLowerBound": lower_bound,
                           "supported": supported, "status": "insufficient" if not sufficient else "promising" if supported else "failed",
                           "folds": folds}}
