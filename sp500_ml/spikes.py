"""Descriptive upward price anomalies, not evidence of manipulation."""
import numpy as np
import pandas as pd

from .cloud_data import read_csv
from .data import _drop_incomplete_session, _normalize_ticker, _ticker_file
from .spike_rules import prepare_prices, spike_signals


def prior_activity(prices, closes, index):
    """Use completed sessions strictly before the event, including disjoint baselines."""
    volumes = pd.to_numeric(prices.get("Volume", pd.Series(dtype=float)), errors="coerce")
    volumes = volumes[~volumes.index.duplicated(keep="last")].reindex(closes.index)
    rows = []
    for sessions in (3, 5, 10):
        start = index - sessions
        recent = volumes.iloc[max(0, start):index]
        baseline = volumes.iloc[max(0, start - 20):max(0, start)]
        valid = (start >= 0 and len(recent) == sessions and len(baseline) == 20
                 and np.isfinite(recent).all() and np.isfinite(baseline).all()
                 and (recent >= 0).all() and (baseline >= 0).all() and baseline.mean() > 0)
        returns = closes.pct_change().iloc[max(1, start):index]
        rows.append({"sessions": sessions,
                     "priceReturn": float(closes.iloc[index - 1] / closes.iloc[start - 1] - 1) if start >= 1 else None,
                     "meanAbsoluteReturn": float(returns.abs().mean()) if start >= 1 else None,
                     "volumeRatio": float(recent.mean() / baseline.mean()) if valid else None})
    return {"asOf": closes.index[index - 1].date().isoformat(), "windows": rows}


def evaluate_spikes(prices, years=5):
    if years not in (1, 3, 5, 10, 0):
        raise ValueError("History must be 1, 3, 5, 10 years or 0 for all available history.")
    prices = prepare_prices(prices)
    closes = prices.Close
    returns, limits, detected = spike_signals(closes)
    start = closes.index[-1] - pd.DateOffset(years=years) if years else closes.index[0]
    events = []
    for i in np.flatnonzero((detected & (closes.index >= start)).to_numpy()):
        events.append({"date": closes.index[i].date().isoformat(),
                       "previousDate": closes.index[i - 1].date().isoformat(),
                       "previousClose": float(closes.iloc[i - 1]), "close": float(closes.iloc[i]),
                       "return": float(returns.iloc[i]), "threshold": float(limits.iloc[i]),
                       "priorActivity": prior_activity(prices, closes, i)})
    window = closes.loc[closes.index >= start]
    volumes = pd.to_numeric(prices.get("Volume", pd.Series(dtype=float)), errors="coerce")
    volumes = volumes[~volumes.index.duplicated(keep="last")].reindex(window.index)
    return {"from": window.index[0].date().isoformat(), "asOf": window.index[-1].date().isoformat(),
            "baselineSessions": 60, "minimumBaselineSessions": 20,
            "assessedSessions": int(limits.loc[window.index].notna().sum()),
            "events": sorted(events, key=lambda row: row["return"], reverse=True),
            "history": [{"date": date.date().isoformat(), "close": float(close),
                         "volume": float(volume) if np.isfinite(volume) and volume >= 0 else None}
                        for (date, close), volume in zip(window.items(), volumes)]}


def read_spikes(ticker, years=5):
    ticker = _normalize_ticker(ticker)
    path = _ticker_file(ticker)
    if not path.exists():
        raise ValueError(f"No saved price history for {ticker}.")
    prices = _drop_incomplete_session(read_csv(path, parse_dates=["Date"], index_col="Date"))
    return {"ticker": ticker, **evaluate_spikes(prices, years)}
