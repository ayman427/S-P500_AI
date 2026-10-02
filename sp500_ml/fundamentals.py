"""SEC point-in-time fundamentals: download, cache, and align to trading dates."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import _normalize_ticker
from .feature_names import FUNDAMENTAL_FEATURES
from .paths import FUNDAMENTAL_DIR, ROOT

FUNDAMENTAL_CONCEPTS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
}

FUNDAMENTALS_CACHE_TTL_HOURS = 24


def _fundamental_file(ticker: str) -> Path:
    return FUNDAMENTAL_DIR / f"{ticker}.csv"


def _fundamental_events_from_companyfacts(
    company_facts: dict, ticker: str | None = None
) -> pd.DataFrame:
    gaap_facts = company_facts.get("facts", {}).get("us-gaap", {})
    event_series = {}

    for feature, concept_names in FUNDAMENTAL_CONCEPTS.items():
        # Companies retag concepts over time (e.g. the 2018 revenue-recognition
        # standard change), so merge every matching tag instead of only the first.
        # Tuple order is tag priority (preferred first) for overlapping filings.
        matching_concepts = [
            (priority, gaap_facts[name])
            for priority, name in enumerate(concept_names)
            if name in gaap_facts
        ]
        if not matching_concepts:
            continue

        is_flow_item = feature not in {"assets", "liabilities", "equity"}
        quarterly_records = []
        annual_records = []
        for concept_priority, concept in matching_concepts:
            units = concept.get("units", {})
            unit = "USD" if "USD" in units else next(iter(units), None)
            if unit is None:
                continue
            for entry in units[unit]:
                if entry.get("form") not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
                    continue
                filed = pd.to_datetime(entry.get("filed"), errors="coerce")
                end = pd.to_datetime(entry.get("end"), errors="coerce")
                if pd.isna(filed) or pd.isna(end):
                    continue

                start = pd.to_datetime(entry.get("start"), errors="coerce")
                record = {
                    "available_date": filed.normalize() + pd.Timedelta(days=1),
                    "end": end,
                    "value": float(entry["val"]),
                    "concept_priority": concept_priority,
                }
                if not is_flow_item:
                    if not pd.isna(start):
                        continue
                    quarterly_records.append({**record, "duration_rank": 0})
                    continue

                if pd.isna(start):
                    continue
                duration_days = (end - start).days
                if 60 <= duration_days <= 120:
                    quarterly_records.append({**record, "duration_rank": 0})
                elif 330 <= duration_days <= 380:
                    annual_records.append(record)

        records = list(quarterly_records)
        if is_flow_item and annual_records and quarterly_records:
            # Q4 is only ever reported as part of the annual total in the
            # 10-K, so derive the discrete quarter (annual minus the three
            # reported quarters) instead of injecting the ~4x annual figure.
            quarterly_df = pd.DataFrame(quarterly_records)
            for annual in annual_records:
                fiscal_year_start = annual["end"] - pd.DateOffset(years=1) + pd.Timedelta(days=1)
                in_year = quarterly_df[
                    (quarterly_df["end"] > fiscal_year_start)
                    & (quarterly_df["end"] <= annual["end"])
                    & (quarterly_df["available_date"] <= annual["available_date"])
                ]
                covered_quarters = in_year.sort_values(
                    ["available_date", "end", "concept_priority"],
                    ascending=[True, True, False],
                    kind="mergesort",
                ).drop_duplicates("end", keep="last")
                if len(covered_quarters) != 3:
                    # Can't reliably derive Q4; skip rather than risk a level jump.
                    label = f"{ticker} " if ticker else ""
                    print(
                        f"SEC Q4 derivation skipped for {label}{feature}, fiscal year "
                        f"{pd.Timestamp(annual['end']).year}: found "
                        f"{len(covered_quarters)} of 3 required quarterly filings."
                    )
                    continue
                records.append(
                    {
                        "available_date": annual["available_date"],
                        "end": annual["end"],
                        "value": annual["value"] - covered_quarters["value"].sum(),
                        "duration_rank": 0,
                        "concept_priority": annual["concept_priority"],
                    }
                )

        if records:
            observations = pd.DataFrame(records).sort_values(
                ["available_date", "end", "duration_rank", "concept_priority"],
                ascending=[True, False, True, True],
                kind="mergesort",
            )
            observations = observations.drop_duplicates("available_date", keep="first")
            event_series[feature] = observations.set_index("available_date")["value"]

    if not event_series:
        return pd.DataFrame(columns=FUNDAMENTAL_FEATURES, dtype=float)

    fundamentals = pd.concat(event_series, axis=1, sort=True).sort_index().ffill()
    revenue = fundamentals.get("revenue", pd.Series(index=fundamentals.index, dtype=float))
    net_income = fundamentals.get("net_income", pd.Series(index=fundamentals.index, dtype=float))
    operating_income = fundamentals.get(
        "operating_income", pd.Series(index=fundamentals.index, dtype=float)
    )
    assets = fundamentals.get("assets", pd.Series(index=fundamentals.index, dtype=float))
    liabilities = fundamentals.get(
        "liabilities", pd.Series(index=fundamentals.index, dtype=float)
    )
    equity = fundamentals.get("equity", pd.Series(index=fundamentals.index, dtype=float))

    result = pd.DataFrame(index=fundamentals.index)
    result["fund_revenue_log"] = np.sign(revenue) * np.log1p(np.abs(revenue))
    result["fund_net_income_log"] = np.sign(net_income) * np.log1p(np.abs(net_income))
    result["fund_assets_log"] = np.sign(assets) * np.log1p(np.abs(assets))
    result["fund_net_margin"] = net_income / revenue.replace(0, np.nan)
    result["fund_operating_margin"] = operating_income / revenue.replace(0, np.nan)
    result["fund_liabilities_to_assets"] = liabilities / assets.replace(0, np.nan)
    result["fund_equity_to_assets"] = equity / assets.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _read_windows_sec_user_agent() -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "SEC_USER_AGENT")
        return str(value).strip()
    except (ImportError, OSError):
        return ""


def _get_sec_user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "").strip() or _read_windows_sec_user_agent()


def update_fundamentals(ticker: str) -> None:
    ticker = _normalize_ticker(ticker)
    user_agent = _get_sec_user_agent()
    if not user_agent:
        raise RuntimeError(
            "Set SEC_USER_AGENT once in your Windows user environment before requesting SEC data."
        )

    FUNDAMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    ticker_map_path = FUNDAMENTAL_DIR / "company_tickers.json"
    if ticker_map_path.exists():
        ticker_map = json.loads(ticker_map_path.read_text(encoding="utf-8"))
    else:
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30
        )
        response.raise_for_status()
        ticker_map = response.json()
        ticker_map_path.write_text(json.dumps(ticker_map), encoding="utf-8")

    company = next(
        (
            item
            for item in ticker_map.values()
            if str(item.get("ticker", "")).upper() == ticker
        ),
        None,
    )
    if company is None:
        raise RuntimeError(f"SEC does not list a company matching ticker {ticker}.")

    cik = int(company["cik_str"])
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    response = requests.get(facts_url, headers=headers, timeout=60)
    response.raise_for_status()
    events = _fundamental_events_from_companyfacts(response.json(), ticker)
    if events.empty:
        raise RuntimeError(f"No supported SEC company facts found for {ticker}.")
    path = _fundamental_file(ticker)
    events.to_csv(path, index_label="Date", date_format="%Y-%m-%d")
    print(f"Saved point-in-time fundamentals for {ticker} to {path.relative_to(ROOT)}")


def _load_fundamentals(ticker: str, dates: pd.Index) -> pd.DataFrame:
    fundamentals, warning = _load_training_fundamentals(ticker, dates)
    if warning:
        print(f"SEC fundamentals note: {warning}")
    if fundamentals is None:
        raise RuntimeError(
            f"The trained model for {ticker} requires SEC fundamentals, but they "
            "are unavailable; refusing to generate a prediction or paper signal."
        )
    return fundamentals


def _read_fundamentals(ticker: str, dates: pd.Index) -> pd.DataFrame:
    path = _fundamental_file(ticker)
    fundamentals = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    fundamentals.index = pd.to_datetime(fundamentals.index).normalize()
    return _align_fundamental_events(fundamentals, dates)


def _fundamentals_cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp.fromtimestamp(
        path.stat().st_mtime, tz="UTC"
    )
    return age.total_seconds() < FUNDAMENTALS_CACHE_TTL_HOURS * 3600


def _load_training_fundamentals(
    ticker: str, dates: pd.Index
) -> tuple[pd.DataFrame | None, str | None]:
    path = _fundamental_file(ticker)
    warning = None
    if not _fundamentals_cache_is_fresh(path):
        try:
            update_fundamentals(ticker)
        except (RuntimeError, requests.RequestException, ValueError) as error:
            if not path.exists():
                return None, f"SEC fundamentals unavailable; training without them: {error}"
            warning = f"SEC refresh failed; using cached fundamentals: {error}"

    try:
        return _read_fundamentals(ticker, dates), warning
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as error:
        return None, f"SEC fundamentals could not be loaded; training without them: {error}"


def _align_fundamental_events(events: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    trading_dates = pd.DatetimeIndex(dates).normalize()
    event_dates = pd.DatetimeIndex(events.index).normalize()
    dated_events = events.copy()
    dated_events.index = event_dates
    union = event_dates.union(trading_dates).sort_values()
    return dated_events.reindex(union).ffill().reindex(trading_dates)
