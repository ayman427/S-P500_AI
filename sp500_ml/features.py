"""Point-in-time feature engineering for stock and market data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .cloud_data import read_csv

from .data import PRICE_COLUMNS, SECTOR_ETFS, VIX_TICKER, _ticker_file
from .feature_names import (
    FEATURES,
    FUNDAMENTAL_FEATURES,
    PRICE_FEATURES,
    SECTOR_ETF_SYMBOLS,
)
from .fundamentals import FUNDAMENTAL_SOURCE_COLUMNS, _align_fundamental_events

__all__ = [
    "FEATURES",
    "PRICE_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "build_features",
    "_available_features",
    "_load_market_data",
]

def _load_sector_data() -> pd.DataFrame:
    """Load sector ETF returns aligned to trading sessions."""
    from .paths import DATA_DIR
    sector_data = {}
    for etf in SECTOR_ETFS:
        path = DATA_DIR / f"{etf}.csv"
        if path.exists():
            df = read_csv(path, parse_dates=["Date"], index_col="Date")
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
            sector_data[etf] = df["Close"].astype(float)
    if not sector_data:
        return pd.DataFrame()
    combined = pd.concat(sector_data, axis=1)
    combined.columns = list(sector_data.keys())
    return combined.sort_index()


def _load_vix_data() -> pd.Series:
    """Load VIX level data."""
    from .paths import DATA_DIR
    path = DATA_DIR / f"{VIX_TICKER}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = read_csv(path, parse_dates=["Date"], index_col="Date")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    return df["Close"].astype(float).rename("VIX")


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
    features["return_5d_x_volatility_20d"] = (
        features["return_5d"] * features["volatility_20d"]
    )
    features["return_20d_x_market_return_5d"] = (
        features["return_20d"] * features["market_return_5d"]
    )

    # Sector returns are optional until their local histories have been synced.
    sector_data = _load_sector_data()
    sector_close = sector_data.reindex(prices.index).ffill()
    for etf in SECTOR_ETF_SYMBOLS:
        column = f"sector_{etf.lower()}_return_"
        if etf in sector_close:
            for horizon in (1, 5, 20):
                features[f"{column}{horizon}d"] = np.log(
                    sector_close[etf] / sector_close[etf].shift(horizon)
                )
        else:
            for horizon in (1, 5, 20):
                features[f"{column}{horizon}d"] = np.nan

    # VIX features
    vix_data = _load_vix_data()
    if not vix_data.empty:
        vix_aligned = vix_data.reindex(prices.index).ffill()
        features["vix_level"] = vix_aligned
        features["vix_change_1d"] = vix_aligned.diff()
        features["vix_change_5d"] = vix_aligned - vix_aligned.shift(5)
    else:
        features["vix_level"] = np.nan
        features["vix_change_1d"] = np.nan
        features["vix_change_5d"] = np.nan

    if fundamentals is not None:
        aligned_fundamentals = _align_fundamental_events(fundamentals, prices.index)
        if set(FUNDAMENTAL_SOURCE_COLUMNS).issubset(aligned_fundamentals.columns):
            growth_features = (
                ("fund_revenue_raw", "fund_revenue_yoy_log_change"),
                ("fund_net_income_raw", "fund_net_income_yoy_log_change"),
                ("fund_assets_raw", "fund_assets_yoy_log_change"),
            )
            for source, feature in growth_features:
                values = aligned_fundamentals[source].astype(float)
                signed_log = np.sign(values) * np.log1p(np.abs(values))
                features[feature] = signed_log - signed_log.shift(252)
            ratio_features = [
                "fund_net_margin",
                "fund_operating_margin",
                "fund_liabilities_to_assets",
                "fund_equity_to_assets",
            ]
            for feature in ratio_features:
                features[feature] = aligned_fundamentals.get(
                    feature, pd.Series(np.nan, index=prices.index)
                )
        else:
            for feature in FUNDAMENTAL_FEATURES:
                features[feature] = aligned_fundamentals.get(
                    feature, pd.Series(np.nan, index=prices.index)
                )
    else:
        features[FUNDAMENTAL_FEATURES] = np.nan

    features["next_date"] = pd.Series(prices.index, index=prices.index).shift(-1)
    features["next_5d_date"] = pd.Series(prices.index, index=prices.index).shift(-5)
    features["next_open"] = prices["Open"].shift(-1)
    features["next_close"] = close.shift(-1)
    features["next_5d_close"] = close.shift(-5)
    features["next_open_to_close_return"] = (
        features["next_close"] / features["next_open"] - 1
    )
    features["next_5d_open_to_close_return"] = (
        features["next_5d_close"] / features["next_open"] - 1
    )
    dates = [prices.index.max(), market.index.max()]
    dates.extend(sector_data[column].last_valid_index() for column in sector_data)
    if not vix_data.empty:
        dates.append(vix_data.last_valid_index())
    features.attrs["input_as_of"] = min(value for value in dates if value is not None)
    return features.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            feature
            for feature in PRICE_FEATURES
            if not feature.startswith(("sector_", "vix_"))
        ]
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
    return read_csv(market_path, parse_dates=["Date"], index_col="Date")







