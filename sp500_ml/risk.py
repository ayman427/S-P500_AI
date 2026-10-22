"""Descriptive historical risk, independent of model forecasts and trade rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cloud_data import read_csv
from .data import _drop_incomplete_session, _normalize_ticker, _ticker_file


def evaluate_risk(prices, years=5, threshold=0.20):
    if years not in (1, 3, 5, 10, 0):
        raise ValueError("History must be 1, 3, 5, 10 years or 0 for all available history.")
    if threshold not in (0.10, 0.20, 0.30):
        raise ValueError("Decline threshold must be 10%, 20%, or 30%.")
    closes = prices["Close"].astype(float).sort_index()
    closes = closes[~closes.index.duplicated(keep="last")]
    if len(closes) < 2 or not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("Risk analysis needs at least two valid positive closing prices.")
    available_start = closes.index[0]
    if years:
        closes = closes.loc[closes.index >= closes.index[-1] - pd.DateOffset(years=years)]
    if len(closes) < 2:
        raise ValueError("Not enough prices in the selected history period.")
    drawdowns = closes / closes.cummax() - 1
    returns = closes.pct_change().dropna()
    episodes = []
    peak_date, peak_price = closes.index[0], float(closes.iloc[0])
    active = None

    def finish(recovery_date=None):
        if active and active["decline"] <= -threshold + 1e-12:
            end = recovery_date if recovery_date is not None else closes.index[-1]
            episodes.append({**active,
                "peakDate": peak_date.date().isoformat(), "peakPrice": peak_price,
                "recoveryDate": recovery_date.date().isoformat() if recovery_date is not None else None,
                "daysUnderwater": int((end - peak_date).days),
                "riseFromWindowStart": peak_price / float(closes.iloc[0]) - 1,
                "reboundNeededAtTrough": peak_price / active["troughPrice"] - 1,
            })

    for day, value in closes.iloc[1:].items():
        value = float(value)
        if value >= peak_price:
            finish(day)
            peak_date, peak_price, active = day, value, None
            continue
        decline = value / peak_price - 1
        if active is None:
            active = {"startDate": day.date().isoformat(), "startPrice": value,
                      "troughDate": day.date().isoformat(), "troughPrice": value,
                      "decline": decline, "thresholdDate": None}
        if decline < active["decline"]:
            active.update(troughDate=day.date().isoformat(), troughPrice=value, decline=decline)
        if active["thresholdDate"] is None and decline <= -threshold + 1e-12:
            active["thresholdDate"] = day.date().isoformat()
    finish()
    events = sorted(episodes, key=lambda event: event["decline"])[:5]
    positions = set(np.linspace(0, len(closes) - 1, min(500, len(closes)), dtype=int))
    positions.update(int(closes.index.get_loc(day)) for day in
                     (closes.idxmin(), closes.idxmax(), drawdowns.idxmin()))
    for event in events:
        for key in ("peakDate", "startDate", "troughDate", "thresholdDate", "recoveryDate"):
            if event[key]:
                positions.add(int(closes.index.get_loc(pd.Timestamp(event[key]))))
    return {
        "years": years, "threshold": threshold,
        "availableFrom": available_start.date().isoformat(),
        "from": closes.index[0].date().isoformat(), "asOf": closes.index[-1].date().isoformat(),
        "sessions": len(closes), "startPrice": float(closes.iloc[0]), "latestPrice": float(closes.iloc[-1]),
        "totalReturn": float(closes.iloc[-1] / closes.iloc[0] - 1),
        "maxDrawdown": float(drawdowns.min()), "currentDrawdown": float(drawdowns.iloc[-1]),
        "annualizedVolatility": float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else None,
        "worstDailyReturn": float(returns.min()),
        "events": events, "qualifyingDeclines": len(episodes),
        "history": [{"date": closes.index[i].date().isoformat(), "close": float(closes.iloc[i]),
                     "drawdown": float(drawdowns.iloc[i])} for i in sorted(positions)],
    }


def read_risk(ticker, years=5, threshold=0.20):
    ticker = _normalize_ticker(ticker)
    path = _ticker_file(ticker)
    if not path.exists():
        raise ValueError(f"No saved price history for {ticker}.")
    prices = _drop_incomplete_session(read_csv(path, parse_dates=["Date"], index_col="Date"))
    result = {"ticker": ticker, **evaluate_risk(prices, years, threshold), "benchmark": None}
    try:
        market = prices if ticker == "SPY" else _drop_incomplete_session(
            read_csv(_ticker_file("SPY"), parse_dates=["Date"], index_col="Date"))
        dates = prices.sort_index().loc[result["from"]:result["asOf"]].index.drop_duplicates()
        # Equal sessions prevent a missing market day from skewing the comparison.
        benchmark = evaluate_risk(market.reindex(dates), years=0, threshold=threshold)
        result["benchmark"] = {"ticker": "SPY", **{key: benchmark[key] for key in
            ("from", "asOf", "sessions", "totalReturn", "maxDrawdown", "currentDrawdown", "annualizedVolatility", "history")}}
    except Exception:
        result["benchmarkError"] = "SPY comparison is unavailable for the selected period."
    return result
