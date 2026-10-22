"""Paper-trading ledgers: settlement, summaries, and per-model isolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .cloud_data import read_csv

from .artifacts import atomic_csv, exclusive_file
from .sessions import execution_window, utc_now, validate_signal_time

from .data import _drop_incomplete_session, _normalize_ticker, _ticker_file
from .models import _model_artifact_paths
from .news import _fetch_recent_news, _save_news_cache
from .paths import PAPER_DIR, ROOT
from .training import _load_latest_prediction

# Retraining changes the model fingerprint and starts a new ledger, so a model
# younger than this is "frozen" and train warns before cutting its record short.
PAPER_MODEL_FREEZE_DAYS = 30


def _settle_paper_rows(ledger: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    settled = ledger.astype(object).copy()
    prices = prices.sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate price sessions cannot be used to settle paper trades.")
    for row_index in settled.index[settled["status"] == "pending"]:
        signal_date = pd.Timestamp(settled.at[row_index, "signal_date"])
        _, _, execution_date = execution_window(signal_date)
        if execution_date not in prices.index:
            if len(prices) and prices.index.max() > execution_date:
                raise ValueError(f"Missing execution-session prices for {execution_date.date()}; refusing to use a later session.")
            continue
        next_open = float(prices.loc[execution_date, "Open"])
        next_close = float(prices.loc[execution_date, "Close"])
        if not np.isfinite([next_open, next_close]).all() or min(next_open, next_close) <= 0:
            raise ValueError("Invalid execution-session prices.")
        cost_bps = float(settled.at[row_index, "transaction_cost_bps_per_side"])
        if not np.isfinite(cost_bps) or cost_bps < 0:
            raise ValueError("Invalid paper transaction cost.")
        round_trip_cost = 2 * cost_bps / 10_000
        enter_long = _entry_flags(settled.loc[[row_index]])[0]
        gross_return = next_close / next_open - 1
        net_return = gross_return - round_trip_cost

        settled.loc[row_index, "execution_date"] = execution_date.date().isoformat()
        settled.loc[row_index, "next_open"] = next_open
        settled.loc[row_index, "next_close"] = next_close
        settled.loc[row_index, "gross_open_to_close_return"] = gross_return
        settled.loc[row_index, "round_trip_cost_return"] = round_trip_cost
        settled.loc[row_index, "net_return_if_traded"] = net_return
        settled.loc[row_index, "strategy_daily_return"] = net_return if enter_long else 0.0
        settled.loc[row_index, "status"] = "settled"
    return settled


def _entry_flags(frame):
    flags = frame["enter_long"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Invalid paper entry decision.")
    return flags.eq("true").to_numpy()


def _paper_summary(ledger: pd.DataFrame) -> dict:
    if ledger["signal_date"].duplicated().any():
        raise ValueError("Duplicate signal dates in paper ledger.")
    if not ledger["status"].isin(["pending", "settled"]).all():
        raise ValueError("Unknown paper settlement status.")
    settled = ledger.loc[ledger["status"] == "settled"].sort_values("signal_date")
    flags = _entry_flags(settled)
    net_returns = pd.to_numeric(settled["net_return_if_traded"], errors="raise").to_numpy(dtype=float)
    recorded = pd.to_numeric(settled["strategy_daily_return"], errors="raise").to_numpy(dtype=float)
    daily_returns = np.where(flags, net_returns, 0.0)
    if not np.isfinite(net_returns).all() or not np.isfinite(recorded).all():
        raise ValueError("Non-finite settled paper returns.")
    if not np.allclose(recorded, daily_returns, rtol=1e-8, atol=1e-10):
        raise ValueError("Paper strategy return contradicts its entry decision and net trade return.")
    if {"next_open", "next_close", "transaction_cost_bps_per_side"}.issubset(settled.columns):
        opened = pd.to_numeric(settled["next_open"]).to_numpy(dtype=float)
        closed = pd.to_numeric(settled["next_close"]).to_numpy(dtype=float)
        costs = pd.to_numeric(settled["transaction_cost_bps_per_side"]).to_numpy(dtype=float)
        if not np.isfinite(np.concatenate([opened, closed, costs])).all() or (opened <= 0).any() or (closed <= 0).any() or (costs < 0).any():
            raise ValueError("Invalid prices or costs in settled paper ledger.")
        expected = closed / opened - 1 - 2 * costs / 10_000
        if not np.allclose(net_returns, expected, rtol=1e-8, atol=1e-10):
            raise ValueError("Paper net return contradicts execution prices and costs.")
    trade_returns = net_returns[flags]
    wins = int((trade_returns > 0).sum())
    losses = int((trade_returns < 0).sum())
    return {
        "signals_logged": int(len(ledger)),
        "settled_sessions": int(len(settled)),
        "pending_signals": int((ledger["status"] == "pending").sum()),
        "trades": int(flags.sum()),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trade_returns) if len(trade_returns) else None,
        "strategy_cumulative_return": float(np.prod(1 + daily_returns) - 1),
        "always_long_cumulative_return": float(np.prod(1 + net_returns) - 1),
        "no_trade_cumulative_return": 0.0,
    }


def _validate_ledger_identity(ledger, ticker, model_type, fingerprint):
    if ledger.empty:
        return
    required = {"ticker", "model_type", "model_sha256", "created_at_utc", "expected_execution_date"}
    if not required.issubset(ledger):
        raise ValueError("Legacy paper ledger lacks verifiable identity or timing; preserved as history only.")
    for key, expected in [("ticker", ticker), ("model_type", model_type), ("model_sha256", fingerprint)]:
        if not ledger[key].eq(expected).all():
            raise ValueError(f"Paper ledger {key} does not match the selected model.")
    for rule in ["entry_threshold", "transaction_cost_bps_per_side"]:
        if ledger[rule].nunique(dropna=False) != 1:
            raise ValueError("Paper ledger mixes different signal rules or costs.")
    for _, row in ledger.iterrows():
        values = np.asarray([row["predicted_return"], row["entry_threshold"], row["transaction_cost_bps_per_side"], row["signal_edge_after_costs"]], dtype=float)
        if not np.isfinite(values).all() or values[2] < 0:
            raise ValueError("Invalid paper signal or cost values.")
        expected_edge = values[0] - 2 * values[2] / 10_000
        if not np.isclose(values[3], expected_edge, rtol=1e-8, atol=1e-10) or bool(_entry_flags(pd.DataFrame([row]))[0]) != bool(expected_edge > values[1]):
            raise ValueError("Paper entry decision contradicts its recorded prediction, threshold, and costs.")
        execution = validate_signal_time(row["signal_date"], row["created_at_utc"])
        if str(row["expected_execution_date"]) != execution.date().isoformat():
            raise ValueError("Paper execution date contradicts the exchange calendar.")
        if row["status"] == "settled" and str(row["execution_date"]) != execution.date().isoformat():
            raise ValueError("Settled paper trade used the wrong execution session.")
    _paper_summary(ledger)


def _paper_ledger_path(
    ticker: str,
    model_fingerprint: str | None = None,
    model_type: str = "hist_gradient_boosting",
    schema_version: int = 1,
) -> Path:
    ticker = _normalize_ticker(ticker)
    default_path = PAPER_DIR / f"{ticker}.csv"
    if model_fingerprint is None:
        model_path, _, _ = _model_artifact_paths(ticker, model_type)
        if not model_path.exists():
            raise RuntimeError(f"No trained {model_type} model for {ticker}; paper results are unavailable.")
        model_fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()

    versioned_path = PAPER_DIR / f"{ticker}_{model_fingerprint[:10]}.csv"
    if schema_version >= 2:
        return PAPER_DIR / f"{ticker}_{model_type}_{model_fingerprint[:10]}_v{schema_version}.csv"
    if not default_path.exists():
        return versioned_path if versioned_path.exists() else default_path
    try:
        recorded = read_csv(default_path, usecols=["model_sha256"])[
            "model_sha256"
        ].dropna()
    except (OSError, ValueError, pd.errors.ParserError):
        recorded = pd.Series(dtype=str)
    if not recorded.empty and recorded.eq(model_fingerprint).all():
        return default_path
    return versioned_path


def _paper_model_freeze_status(
    ticker: str, model_type: str = "hist_gradient_boosting"
) -> dict | None:
    model_path, _, _ = _model_artifact_paths(ticker, model_type)
    if not model_path.exists():
        return None
    ledger_path = _paper_ledger_path(ticker, model_type=model_type, schema_version=3)
    if not ledger_path.exists():
        return None
    try:
        signals = read_csv(ledger_path, usecols=["signal_date"])
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if signals.empty:
        return None
    first_signal = pd.to_datetime(signals["signal_date"]).min().normalize()
    age_days = (pd.Timestamp.now().normalize() - first_signal).days
    return {
        "ledger": ledger_path.name,
        "age_days": age_days,
        "freeze_days": PAPER_MODEL_FREEZE_DAYS,
        "still_frozen": age_days < PAPER_MODEL_FREEZE_DAYS,
    }


def _paper_trade(
    ticker: str,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
    model_type: str = "hist_gradient_boosting",
) -> None:
    prediction, model_fingerprint = _load_latest_prediction(ticker, model_type)
    ticker = prediction["ticker"]
    headlines = _fetch_recent_news(ticker)
    _save_news_cache(ticker, headlines)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = _paper_ledger_path(
        ticker, model_fingerprint, model_type, schema_version=3
    )
    columns = [
        "ticker",
        "model_type",
        "signal_date",
        "created_at_utc",
        "expected_execution_date",
        "last_close",
        "predicted_return",
        "model_sha256",
        "entry_threshold",
        "transaction_cost_bps_per_side",
        "news_context_json",
        "status",
        "execution_date",
        "next_open",
        "next_close",
        "signal_edge_after_costs",
        "enter_long",
        "gross_open_to_close_return",
        "round_trip_cost_return",
        "net_return_if_traded",
        "strategy_daily_return",
    ]
    if ledger_path.exists():
        ledger = read_csv(ledger_path, dtype={"signal_date": str})
        _validate_ledger_identity(ledger, ticker, model_type, model_fingerprint)
        if "news_context_json" not in ledger.columns:
            ledger["news_context_json"] = ""
        if "model_type" not in ledger.columns:
            ledger["model_type"] = "hist_gradient_boosting"
        if not ledger.empty:
            if not ledger["model_sha256"].eq(model_fingerprint).all():
                raise RuntimeError(
                    f"Paper ledger {ledger_path.name} contains a different model version; refusing to mix results."
                )
            thresholds = pd.to_numeric(ledger["entry_threshold"]).unique()
            costs = pd.to_numeric(ledger["transaction_cost_bps_per_side"]).unique()
            if len(thresholds) != 1 or not np.isclose(thresholds[0], entry_threshold):
                raise RuntimeError("Entry threshold differs from the existing paper ledger.")
            if len(costs) != 1 or not np.isclose(costs[0], transaction_cost_bps):
                raise RuntimeError("Transaction cost differs from the existing paper ledger.")
        settlement_prices = _drop_incomplete_session(
            read_csv(
                _ticker_file(ticker), parse_dates=["Date"], index_col="Date"
            )
        )
        ledger = _settle_paper_rows(ledger, settlement_prices)
    else:
        ledger = pd.DataFrame(columns=columns)

    signal_date = prediction["as_of"]
    matching_signals = ledger.index[
        ledger.get("signal_date", pd.Series(dtype=str)).astype(str) == signal_date
    ]
    if len(matching_signals):
        print(f"Signal for {ticker} on {signal_date} is already logged; no duplicate added.")
    else:
        created_at = utc_now()
        execution = validate_signal_time(signal_date, created_at)
        if not np.isfinite([prediction["predicted_return"], entry_threshold, transaction_cost_bps]).all() or transaction_cost_bps < 0:
            raise ValueError("Invalid paper forecast, threshold, or transaction cost.")
        if prediction.get("model_trained_through") and pd.Timestamp(prediction["model_trained_through"]) > pd.Timestamp(signal_date):
            raise ValueError("Model was trained after its signal date.")
        if prediction.get("model_trained_at_utc") and pd.Timestamp(prediction["model_trained_at_utc"]) > created_at:
            raise ValueError("Model training timestamp is in the future.")
        round_trip_cost = 2 * transaction_cost_bps / 10_000
        signal_edge_after_costs = prediction["predicted_return"] - round_trip_cost
        ledger = pd.concat(
            [
                ledger,
                pd.DataFrame(
                    [
                        {
                            **prediction,
                            "signal_date": signal_date,
                            "created_at_utc": created_at.isoformat(),
                            "expected_execution_date": execution.date().isoformat(),
                            "model_type": model_type,
                            "model_sha256": model_fingerprint,
                            "entry_threshold": entry_threshold,
                            "transaction_cost_bps_per_side": transaction_cost_bps,
                            "signal_edge_after_costs": signal_edge_after_costs,
                            "enter_long": signal_edge_after_costs > entry_threshold,
                            "news_context_json": json.dumps(headlines, separators=(",", ":")),
                            "status": "pending",
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    ledger = ledger.reindex(columns=columns)
    _paper_summary(ledger)
    atomic_csv(ledger, ledger_path)
    print(json.dumps(_paper_summary(ledger), indent=2))
    print(f"Saved paper-trading ledger to {ledger_path.relative_to(ROOT)}")
    if headlines:
        print("Recent headlines and entity context (not used by the model):")
        for headline in headlines:
            print(
                f"{headline['published_at_utc']} | {headline['publisher']} | "
                f"{headline['title']} | {headline['url']}"
            )
            for relationship in headline["relationships"]:
                print(
                    f"  Candidate relationship: {relationship['subject_ticker']} "
                    f"{relationship['predicate']} {relationship['counterparty_candidate']}"
                )
                for context in relationship["counterparty_context"]:
                    print(f"  Context: {context['title']} | {context['snippet']} | {context['url']}")


def _paper_status(
    ticker: str, model_type: str = "hist_gradient_boosting"
) -> None:
    ticker = _normalize_ticker(ticker)
    ledger_path = _paper_ledger_path(
        ticker, model_type=model_type, schema_version=3
    )
    if not ledger_path.exists():
        raise RuntimeError(f"No paper trades for {ticker}; run `paper-trade --ticker {ticker}` first.")
    ledger = read_csv(ledger_path, dtype={"signal_date": str})
    model_path, _, _ = _model_artifact_paths(ticker, model_type)
    fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()
    _validate_ledger_identity(ledger, ticker, model_type, fingerprint)

    # Settle against local prices so status reflects sessions completed since the last signal.
    price_path = _ticker_file(ticker)
    pending_before = int((ledger["status"] == "pending").sum())
    if price_path.exists() and pending_before and "signal_edge_after_costs" in ledger.columns:
        prices = _drop_incomplete_session(
            read_csv(price_path, parse_dates=["Date"], index_col="Date")
        )
        settled = _settle_paper_rows(ledger, prices)
        newly_settled = pending_before - int((settled["status"] == "pending").sum())
        if newly_settled:
            _paper_summary(settled)
            atomic_csv(settled, ledger_path)
            print(f"Settled {newly_settled} pending signal(s) using local price data.")
        ledger = settled
    print(json.dumps(_paper_summary(ledger), indent=2))


def paper_trade(ticker, entry_threshold=0.0, transaction_cost_bps=10.0, model_type="hist_gradient_boosting"):
    ticker = _normalize_ticker(ticker)
    with exclusive_file(PAPER_DIR / f".{ticker}_{model_type}.lock"):
        return _paper_trade(ticker, entry_threshold, transaction_cost_bps, model_type)


def paper_status(ticker, model_type="hist_gradient_boosting"):
    ticker = _normalize_ticker(ticker)
    with exclusive_file(PAPER_DIR / f".{ticker}_{model_type}.lock"):
        return _paper_status(ticker, model_type)
