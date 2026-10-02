"""Price history download, local storage, and adjustment-drift detection."""
from __future__ import annotations

import re
import os
from io import StringIO
from pathlib import Path
from datetime import time as dt_time
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .paths import DATA_DIR

START_DATE = "2000-01-01"
BATCH_SIZE = 25
FULL_REFRESH_INTERVAL_DAYS = 180
PRICE_COLUMN_ORDER = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = set(PRICE_COLUMN_ORDER)


def get_sp500_tickers() -> list[str]:
    """Read the current S&P 500 constituents from Wikipedia."""
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(
        StringIO(response.text),
        attrs={"id": "constituents"},
    )
    return (
        tables[0]["Symbol"]
        .astype(str)
        .str.replace(".", "-", regex=False)
        .drop_duplicates()
        .tolist()
    )


def _ticker_file(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}.csv"


def _atomic_write_prices(prices: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        prices.to_csv(temporary_path, date_format="%Y-%m-%d")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _full_refresh_marker(path: Path) -> Path:
    return path.with_suffix(".full_refresh")


def _record_full_refresh(path: Path, now: pd.Timestamp) -> None:
    marker = _full_refresh_marker(path)
    temporary_path = marker.with_name(f".{marker.name}.{uuid4().hex}.tmp")
    current_time = pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")
    try:
        temporary_path.write_text(current_time.isoformat(), encoding="utf-8")
        os.replace(temporary_path, marker)
    finally:
        temporary_path.unlink(missing_ok=True)


def _drop_incomplete_session(
    prices: pd.DataFrame, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    if prices.empty:
        return prices
    current_time = pd.Timestamp.now(tz=ZoneInfo("America/New_York")) if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize(ZoneInfo("America/New_York"))
    else:
        current_time = current_time.tz_convert(ZoneInfo("America/New_York"))
    today = current_time.normalize().tz_localize(None)
    if current_time.time() < dt_time(16, 30):
        return prices.loc[prices.index < today]
    return prices.loc[prices.index <= today]


def _full_refresh_due(path: Path, now: pd.Timestamp) -> bool:
    current_time = pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")
    marker = _full_refresh_marker(path)
    if marker.exists():
        try:
            last_modified = pd.Timestamp(marker.read_text(encoding="utf-8"))
            if last_modified.tzinfo is None:
                last_modified = last_modified.tz_localize("UTC")
            else:
                last_modified = last_modified.tz_convert("UTC")
        except (OSError, ValueError):
            return True
    else:
        last_modified = pd.Timestamp.fromtimestamp(path.stat().st_mtime, tz="UTC")
    return (current_time - last_modified).days >= FULL_REFRESH_INTERVAL_DAYS


def _normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z0-9-]+", normalized):
        raise ValueError("Ticker must contain only letters, numbers, dots, or hyphens.")
    return normalized


def _extract_ticker(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(download.columns, pd.MultiIndex):
        if ticker in download.columns.get_level_values(0):
            download = download.xs(ticker, axis=1, level=0)
        elif ticker in download.columns.get_level_values(1):
            download = download.xs(ticker, axis=1, level=1)
        else:
            return pd.DataFrame()

    columns = {str(column).lower(): column for column in download.columns}
    if not PRICE_COLUMNS.issubset({column.title() for column in columns}):
        return pd.DataFrame()

    # Iterate a fixed list rather than the PRICE_COLUMNS set so column order
    # (and therefore file layout) is identical across runs/processes.
    prices = download[[columns[column.lower()] for column in PRICE_COLUMN_ORDER]].copy()
    prices.columns = [column.title() for column in PRICE_COLUMN_ORDER]
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None).normalize()
    prices.index.name = "Date"
    prices = prices.dropna(subset=["Close"])
    return prices[~prices.index.duplicated(keep="last")].sort_index()


def _download_batch(tickers: list[str], start: str) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        multi_level_index=True,
        progress=False,
        threads=True,
    )


def _merge_price_history(old: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Merge freshly downloaded rows with stored history and flag adjustment drift.

    A split or dividend changes Yahoo's auto-adjust factor for all earlier
    history, so an overlapping date whose adjusted close moved materially
    means the stored series is stale and needs a full re-download.
    """
    old = old[~old.index.duplicated(keep="last")].sort_index()
    fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
    overlap = old.index.intersection(fresh.index)
    drifted = False
    if not overlap.empty:
        relative_diff = (
            (old.loc[overlap, "Close"] - fresh.loc[overlap, "Close"]).abs()
            / old.loc[overlap, "Close"].replace(0, np.nan)
        )
        drifted = bool((relative_diff > 1e-4).any())
    combined = pd.concat([old, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined, drifted


def _price_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []


def list_tickers() -> None:
    tickers = [path.stem for path in _price_files() if path.stem != "SPY"]
    if not tickers:
        raise RuntimeError("No local price data. Run `python -m sp500_ml update` first.")
    print(" ".join(tickers))


def update_data() -> None:
    constituents = get_sp500_tickers()
    tickers = list(dict.fromkeys([*constituents, "SPY"]))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
    full_history = []
    incremental_batches: dict[str, list[str]] = {}
    for ticker in tickers:
        path = _ticker_file(ticker)
        if not path.exists():
            full_history.append(ticker)
            continue
        try:
            stored_columns = set(pd.read_csv(path, nrows=0).columns)
        except (OSError, pd.errors.ParserError):
            stored_columns = set()
        refresh_due = _full_refresh_due(path, now)
        if not PRICE_COLUMNS.issubset(stored_columns) or refresh_due:
            full_history.append(ticker)
        else:
            last_date = pd.to_datetime(
                pd.read_csv(path, usecols=["Date"])["Date"]
            ).max()
            start = (last_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            incremental_batches.setdefault(start, []).append(ticker)

    batches = [(full_history, START_DATE)]
    batches.extend((symbols, start) for start, symbols in incremental_batches.items())

    updated = 0
    drifted_tickers: set[str] = set()
    for symbols, start in batches:
        for offset in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[offset : offset + BATCH_SIZE]
            try:
                downloaded = _download_batch(batch, start)
            except Exception as error:
                print(f"Download failed for {', '.join(batch)}: {error}")
                continue

            for ticker in batch:
                fresh = _drop_incomplete_session(_extract_ticker(downloaded, ticker), now)
                if fresh.empty:
                    print(f"No new Yahoo Finance data returned for {ticker}.")
                    continue

                path = _ticker_file(ticker)
                if path.exists() and start != START_DATE:
                    old = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                    old.index = pd.to_datetime(old.index, utc=True).tz_convert(None).normalize()
                    old = _drop_incomplete_session(old, now)
                    fresh, drifted = _merge_price_history(old, fresh)
                    if drifted:
                        drifted_tickers.add(ticker)
                        continue
                _atomic_write_prices(fresh, path)
                if start == START_DATE:
                    _record_full_refresh(path, now)
                updated += 1

    if drifted_tickers:
        drifted_list = sorted(drifted_tickers)
        print(
            f"Detected adjusted-price drift (likely a split/dividend) for "
            f"{len(drifted_list)} ticker(s); re-downloading full history: "
            f"{', '.join(drifted_list)}"
        )
        for offset in range(0, len(drifted_list), BATCH_SIZE):
            batch = drifted_list[offset : offset + BATCH_SIZE]
            try:
                downloaded = _download_batch(batch, START_DATE)
            except Exception as error:
                print(f"Full re-download failed for {', '.join(batch)}: {error}")
                continue
            for ticker in batch:
                fresh = _drop_incomplete_session(_extract_ticker(downloaded, ticker), now)
                if fresh.empty:
                    print(f"No Yahoo Finance data returned for {ticker} during drift re-download.")
                    continue
                _atomic_write_prices(fresh, _ticker_file(ticker))
                _record_full_refresh(_ticker_file(ticker), now)
                updated += 1

    print(f"Updated {updated} price files for {len(constituents)} current constituents and SPY.")
