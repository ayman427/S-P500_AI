"""Point-in-time feature engineering for stock and market data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PRICE_COLUMNS, _ticker_file
from .feature_names import FEATURES, FUNDAMENTAL_FEATURES, PRICE_FEATURES
from .fundamentals import _align_fundamental_events

__all__ = [
    "FEATURES",
    "PRICE_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "build_features",
    "_available_features",
    "_load_market_data",
]


def build_features(
    prices: pd.DataFrame,
    market: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build point-in-time stock and market features with a next-close label."""
    prices = prices.sort_index().copy()
    if not PRICE_COLUMNS.issubset(prices.columns):
        raise ValueError(f"Price data must contain: {', '.join(sorted(PRICE_COLUMNS))}")
    if "Close" not in market.columns:
        raise ValueError("Market data must contain a Close column.")

    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float)
    daily_return = np.log(close).diff()
    features = pd.DataFrame(index=prices.index)
    features["close"] = close
    features["return_1d"] = daily_return
    features["return_5d"] = np.log(close / close.shift(5))
    features["return_20d"] = np.log(close / close.shift(20))
    features["return_60d"] = np.log(close / close.shift(60))
    features["volatility_5d"] = daily_return.rolling(5).std()
    features["volatility_20d"] = daily_return.rolling(20).std()
    features["volatility_60d"] = daily_return.rolling(60).std()
    features["intraday_return"] = prices["Close"] / prices["Open"] - 1
    features["overnight_gap"] = prices["Open"] / close.shift(1) - 1
    features["daily_range"] = (prices["High"] - prices["Low"]) / close
    average_volume = volume.rolling(20).mean()
    features["volume_ratio_20d"] = (volume / average_volume - 1).replace(
        [np.inf, -np.inf], np.nan
    )

    market_close = market["Close"].astype(float).reindex(prices.index).ffill()
    market_return = np.log(market_close).diff()
    features["market_return_1d"] = market_return
    features["market_return_5d"] = np.log(market_close / market_close.shift(5))
    features["market_return_20d"] = np.log(market_close / market_close.shift(20))
    features["market_volatility_20d"] = market_return.rolling(20).std()
    if fundamentals is not None:
        aligned_fundamentals = _align_fundamental_events(fundamentals, prices.index)
        features[FUNDAMENTAL_FEATURES] = aligned_fundamentals[FUNDAMENTAL_FEATURES]
    else:
        features[FUNDAMENTAL_FEATURES] = np.nan
    features["next_date"] = pd.Series(prices.index, index=prices.index).shift(-1)
    features["next_open"] = prices["Open"].shift(-1)
    features["next_close"] = close.shift(-1)
    return features.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[feature for feature in FEATURES if feature not in FUNDAMENTAL_FEATURES]
    )


def _available_features(features: pd.DataFrame) -> list[str]:
    return [
        name
        for name in FEATURES
        if name in features and features[name].nunique(dropna=True) > 1
    ]


def _load_market_data() -> pd.DataFrame:
    market_path = _ticker_file("SPY")
    if not market_path.exists():
        raise RuntimeError("SPY market data is missing. Run `python -m sp500_ml update` first.")
    return pd.read_csv(market_path, parse_dates=["Date"], index_col="Date")







