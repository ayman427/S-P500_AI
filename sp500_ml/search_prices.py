"""Refresh the searched stock and its market inputs before inference."""
from __future__ import annotations

import threading
from contextlib import ExitStack

import numpy as np
import pandas as pd
from .cloud_data import read_csv

from . import data
from .sessions import _calendar, utc_now

_lock = threading.Lock()
_symbol_locks = {}
_checked = {}


def _summary(symbols, expected, cached):
    dates = [_checked[symbol][2] for symbol in symbols]
    common = min(dates)
    return {"asOf": dates[0].date().isoformat(), "inputAsOf": common.date().isoformat(),
            "latestCompletedSession": expected.date().isoformat(), "cached": cached}


def latest_completed_session(now):
    calendar = _calendar(now.year)
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    session = calendar.date_to_session(today, direction="previous")
    # Give the provider time to publish its final daily bar.
    if now < calendar.session_close(session) + pd.Timedelta(minutes=30):
        session = calendar.previous_session(session)
    return session


def refresh_search_prices(ticker, *, include_market_inputs=True):
    ticker = data._normalize_ticker(ticker)
    symbols = list(dict.fromkeys([ticker, "SPY", *data.SECTOR_ETFS, data.VIX_TICKER])) if include_market_inputs else [ticker]
    with _lock:
        symbol_locks = [_symbol_locks.setdefault(symbol, threading.Lock()) for symbol in sorted(symbols)]
    with ExitStack() as locks:
        # Consistent ordering protects shared market inputs without blocking unrelated stocks.
        for lock in symbol_locks:
            locks.enter_context(lock)
        now = utc_now()
        expected = latest_completed_session(now)
        needed = [symbol for symbol in symbols if symbol not in _checked
                  or _checked[symbol][0] != expected
                  or (now - _checked[symbol][1]).total_seconds() >= 60
                  or not data._ticker_file(symbol).exists()]
        if not needed:
            return _summary(symbols, expected, True)
        data.DATA_DIR.mkdir(parents=True, exist_ok=True)
        starts = {}
        stored = {}
        for symbol in needed:
            path = data._ticker_file(symbol)
            start = data.START_DATE
            if path.exists():
                old = read_csv(path, parse_dates=["Date"], index_col="Date")
                old.index = pd.to_datetime(old.index, utc=True).tz_convert(None).normalize()
                if data.PRICE_COLUMNS.issubset(old.columns) and not old.empty and not data._full_refresh_due(path, now):
                    stored[symbol] = old
                    start = (old.index.max() - pd.Timedelta(days=10)).date().isoformat()
            starts.setdefault(start, []).append(symbol)
        staged = {}
        full = set()
        for start, batch in starts.items():
            downloaded = data._download_batch(batch, start)
            for symbol in batch:
                fresh = data._extract_ticker(downloaded, symbol)
                if not fresh.empty:
                    fresh = fresh.loc[fresh.index <= expected]
                # A bulk response can omit a symbol or its final daily bar.
                # Retry that symbol once before rejecting the entire refresh.
                if fresh.empty or fresh.index.max() != expected:
                    fresh = data._extract_ticker(data._download_batch([symbol], start), symbol)
                    if not fresh.empty:
                        fresh = fresh.loc[fresh.index <= expected]
                if fresh.empty:
                    raise RuntimeError(f"Latest completed prices for {symbol} are unavailable ({expected.date()}). Please try again shortly.")
                if start != data.START_DATE:
                    fresh, drift = data._merge_price_history(stored[symbol], fresh)
                    if drift:
                        fresh = data._extract_ticker(data._download_batch([symbol], data.START_DATE), symbol)
                        if not fresh.empty:
                            fresh = fresh.loc[fresh.index <= expected]
                        if fresh.empty:
                            raise RuntimeError(f"Adjusted price refresh failed for {symbol}. Please try again.")
                        full.add(symbol)
                else:
                    full.add(symbol)
                values = fresh[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise RuntimeError(f"Invalid price history returned for {symbol}.")
                staged[symbol] = fresh
        # Do not publish partial downloads when any required market input failed.
        for symbol, frame in staged.items():
            data._atomic_write_prices(frame, data._ticker_file(symbol))
            if symbol in full:
                data._record_full_refresh(data._ticker_file(symbol), now)
            _checked[symbol] = (expected, now, frame.index.max())
        return _summary(symbols, expected, False)
