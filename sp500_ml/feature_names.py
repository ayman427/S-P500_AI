"""Feature name lists shared by feature building and fundamentals alignment."""
from __future__ import annotations

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "intraday_return",
    "overnight_gap",
    "daily_range",
    "volume_ratio_20d",
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_volatility_20d",
    "fund_revenue_log",
    "fund_net_income_log",
    "fund_assets_log",
    "fund_net_margin",
    "fund_operating_margin",
    "fund_liabilities_to_assets",
    "fund_equity_to_assets",
]
PRICE_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "intraday_return",
    "overnight_gap",
    "daily_range",
    "volume_ratio_20d",
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_volatility_20d",
]
FUNDAMENTAL_FEATURES = [
    "fund_revenue_log",
    "fund_net_income_log",
    "fund_assets_log",
    "fund_net_margin",
    "fund_operating_margin",
    "fund_liabilities_to_assets",
    "fund_equity_to_assets",
]
assert PRICE_FEATURES + FUNDAMENTAL_FEATURES == FEATURES, (
    "PRICE_FEATURES + FUNDAMENTAL_FEATURES must equal FEATURES; update all three together."
)
