"""Feature name lists shared by feature building and fundamentals alignment."""
from __future__ import annotations

SECTOR_ETF_SYMBOLS = (
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"
)
SECTOR_FEATURES = [
    f"sector_{symbol.lower()}_return_{horizon}d"
    for symbol in SECTOR_ETF_SYMBOLS
    for horizon in (1, 5, 20)
]
VIX_FEATURES = ["vix_level", "vix_change_1d", "vix_change_5d"]

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
    "return_5d_x_volatility_20d",
    "return_20d_x_market_return_5d",
    *SECTOR_FEATURES,
    *VIX_FEATURES,
    "fund_revenue_yoy_log_change",
    "fund_net_income_yoy_log_change",
    "fund_assets_yoy_log_change",
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
    "return_5d_x_volatility_20d",
    "return_20d_x_market_return_5d",
    *SECTOR_FEATURES,
    *VIX_FEATURES,
]
FUNDAMENTAL_FEATURES = [
    "fund_revenue_yoy_log_change",
    "fund_net_income_yoy_log_change",
    "fund_assets_yoy_log_change",
    "fund_net_margin",
    "fund_operating_margin",
    "fund_liabilities_to_assets",
    "fund_equity_to_assets",
]
assert PRICE_FEATURES + FUNDAMENTAL_FEATURES == FEATURES, (
    "PRICE_FEATURES + FUNDAMENTAL_FEATURES must equal FEATURES; update all three together."
)
