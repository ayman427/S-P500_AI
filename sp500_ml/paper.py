"""Paper-trading ledgers: settlement, summaries, and per-model isolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    for row_index in settled.index[settled["status"] == "pending"]:
        signal_date = pd.Timestamp(settled.at[row_index, "signal_date"])
        later_prices = prices.loc[prices.index > signal_date].dropna(
            subset=["Open", "Close"]
        )
        if later_prices.empty:
            continue

        execution_date = later_prices.index[0]
        next_open = float(later_prices.iloc[0]["Open"])
        next_close = float(later_prices.iloc[0]["Close"])
        cost_bps = float(settled.at[row_index, "transaction_cost_bps_per_side"])
        round_trip_cost = 2 * cost_bps / 10_000
        enter_long = str(settled.at[row_index, "enter_long"]).lower() == "true"
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


def _paper_summary(ledger: pd.DataFrame) -> dict:
    settled = ledger.loc[ledger["status"] == "settled"].sort_values("signal_date")
    entered = settled.loc[settled["enter_long"].astype(str).str.lower() == "true"]
    trade_returns = pd.to_numeric(entered["net_return_if_traded"])
    wins = int((trade_returns > 0).sum())
    losses = int((trade_returns < 0).sum())
    daily_returns = pd.to_numeric(settled["strategy_daily_return"]).fillna(0).to_numpy()
    always_long_returns = pd.to_numeric(settled["net_return_if_traded"]).fillna(0).to_numpy()
    return {
        "signals_logged": int(len(ledger)),
        "settled_sessions": int(len(settled)),
        "pending_signals": int((ledger["status"] == "pending").sum()),
        "trades": int(len(entered)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(entered) if len(entered) else None,
        "strategy_cumulative_return": float(np.prod(1 + daily_returns) - 1),
        "always_long_cumulative_return": float(np.prod(1 + always_long_returns) - 1),
        "no_trade_cumulative_return": 0.0,
    }


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
            return default_path
        model_fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()

    versioned_path = PAPER_DIR / f"{ticker}_{model_fingerprint[:10]}.csv"
    if schema_version >= 2:
        return PAPER_DIR / f"{ticker}_{model_type}_{model_fingerprint[:10]}_v{schema_version}.csv"
    if not default_path.exists():
        return versioned_path if versioned_path.exists() else default_path
    try:
        recorded = pd.read_csv(default_path, usecols=["model_sha256"])[
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
    ledger_path = _paper_ledger_path(ticker, model_type=model_type, schema_version=2)
    if not ledger_path.exists():
        return None
    try:
        signals = pd.read_csv(ledger_path, usecols=["signal_date"])
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


def paper_trade(
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
        ticker, model_fingerprint, model_type, schema_version=2
    )
    columns = [
        "ticker",
        "model_type",
        "signal_date",
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
        ledger = pd.read_csv(ledger_path, dtype={"signal_date": str})
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
            pd.read_csv(
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
    ledger.to_csv(ledger_path, index=False)
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


def paper_status(
    ticker: str, model_type: str = "hist_gradient_boosting"
) -> None:
    ticker = _normalize_ticker(ticker)
    ledger_path = _paper_ledger_path(
        ticker, model_type=model_type, schema_version=2
    )
    if not ledger_path.exists():
        ledger_path = _paper_ledger_path(ticker, model_type=model_type)
    if not ledger_path.exists():
        raise RuntimeError(f"No paper trades for {ticker}; run `paper-trade --ticker {ticker}` first.")
    ledger = pd.read_csv(ledger_path, dtype={"signal_date": str})

    # Settle against local prices so status reflects sessions completed since the last signal.
    price_path = _ticker_file(ticker)
    pending_before = int((ledger["status"] == "pending").sum())
    if price_path.exists() and pending_before and "signal_edge_after_costs" in ledger.columns:
        prices = _drop_incomplete_session(
            pd.read_csv(price_path, parse_dates=["Date"], index_col="Date")
        )
        settled = _settle_paper_rows(ledger, prices)
        newly_settled = pending_before - int((settled["status"] == "pending").sum())
        if newly_settled:
            settled.to_csv(ledger_path, index=False)
            print(f"Settled {newly_settled} pending signal(s) using local price data.")
        ledger = settled
    print(json.dumps(_paper_summary(ledger), indent=2))
