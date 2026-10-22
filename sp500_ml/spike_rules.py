"""Shared input validation and the completed-session spike definition."""
import numpy as np
import pandas as pd

DETECTOR_VERSION = "daily_5pct_3sigma_v1"


def prepare_prices(prices):
    if "Close" not in prices:
        raise ValueError("Closing prices are required.")
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")].copy()
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is not None or prices.index.hasnans or not prices.index.equals(prices.index.normalize()):
        raise ValueError("Prices require valid exchange dates without a timezone.")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    if len(prices) < 2 or not np.isfinite(prices.Close).all() or (prices.Close <= 0).any():
        raise ValueError("Spike analysis needs at least two valid positive closing prices.")
    volume = pd.to_numeric(prices.get("Volume", pd.Series(index=prices.index, dtype=float)), errors="coerce")
    prices["Volume"] = volume.where(np.isfinite(volume) & (volume >= 0))
    return prices


def spike_signals(close):
    returns = close.pct_change(fill_method=None)
    baseline = returns.shift(1).rolling(60, min_periods=20)
    thresholds = (baseline.mean() + 3 * baseline.std()).clip(lower=.05)
    return returns, thresholds, returns >= thresholds
