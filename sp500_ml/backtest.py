"""Walk-forward validation, holdout evaluation, and model comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .data import _drop_incomplete_session, _normalize_ticker, _price_files, _ticker_file
from .features import _available_features, _load_market_data, build_features
from .fundamentals import _fundamental_file, _load_training_fundamentals, _read_fundamentals
from .models import _holdout_report_path, _new_model
from .paths import MODEL_DIR, ROOT

FINAL_HOLDOUT_SESSIONS = 252


class DataSufficiencyError(RuntimeError):
    """A ticker lacks enough usable history; safe to skip in batch runs."""


# Anything outside this tuple is a real bug and must propagate from batch runs.
_SKIPPABLE_ERRORS = (
    DataSufficiencyError,
    OSError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)


def _wilson_interval(wins: int, trials: int, z_score: float = 1.959963984540054) -> list[float] | None:
    if trials == 0:
        return None
    rate = wins / trials
    denominator = 1 + z_score**2 / trials
    center = (rate + z_score**2 / (2 * trials)) / denominator
    margin = (
        z_score
        * np.sqrt(rate * (1 - rate) / trials + z_score**2 / (4 * trials**2))
        / denominator
    )
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def _trade_results(
    predicted_returns: np.ndarray,
    validation: pd.DataFrame,
    entry_threshold: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    predicted_returns = np.asarray(predicted_returns, dtype=float)
    valid = validation["next_open"].notna() & validation["next_close"].notna()
    validation = validation.loc[valid]
    predicted_returns = predicted_returns[valid.to_numpy()]
    next_open = validation["next_open"].to_numpy()
    cost = 2 * transaction_cost_bps / 10_000
    signal_edge_after_costs = predicted_returns - cost
    gross_returns = validation["next_close"].to_numpy() / next_open - 1
    net_returns = gross_returns - cost
    execution_dates = validation.get(
        "next_date", pd.Series(pd.NaT, index=validation.index)
    )
    return pd.DataFrame(
        {
            "signal_date": validation.index,
            "execution_date": execution_dates.to_numpy(),
            "predicted_return": predicted_returns,
            "next_open": next_open,
            "next_close": validation["next_close"].to_numpy(),
            "signal_edge_after_costs": signal_edge_after_costs,
            "enter_long": signal_edge_after_costs > entry_threshold,
            "gross_open_to_close_return": gross_returns,
            "round_trip_cost_return": cost,
            "net_return_if_traded": net_returns,
            "strategy_daily_return": np.where(
                signal_edge_after_costs > entry_threshold, net_returns, 0.0
            ),
        },
        index=validation.index,
    )


def _trade_metrics(
    predicted_returns: np.ndarray,
    validation: pd.DataFrame,
    entry_threshold: float,
    transaction_cost_bps: float,
) -> dict:
    trade_log = _trade_results(
        predicted_returns, validation, entry_threshold, transaction_cost_bps
    )
    signals = trade_log["enter_long"].to_numpy(dtype=bool)
    net_returns = trade_log["net_return_if_traded"].to_numpy()
    trade_returns = net_returns[signals]
    wins = int(np.sum(trade_returns > 0))
    losses = int(np.sum(trade_returns < 0))
    trades = int(len(trade_returns))
    daily_returns = trade_log["strategy_daily_return"].to_numpy()
    equity = np.cumprod(1 + daily_returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdowns = equity / peaks - 1 if len(equity) else np.array([])
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    total_losses = float(-trade_returns[trade_returns < 0].sum())
    total_wins = float(trade_returns[trade_returns > 0].sum())
    win_rate = wins / trades if trades else None
    confidence_interval = _wilson_interval(wins, trades)
    always_long_wins = int(np.sum(net_returns > 0))
    always_long_losses = int(np.sum(net_returns < 0))
    always_long_total = float(np.prod(1 + net_returns) - 1)

    return {
        "entry_threshold_pct": entry_threshold * 100,
        "transaction_cost_bps_per_side": transaction_cost_bps,
        "sessions": int(len(validation)),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "breakeven_trades": trades - wins - losses,
        "win_rate": win_rate,
        "win_rate_95pct_wilson_interval": confidence_interval,
        "win_rate_confidently_above_50pct": (
            bool(confidence_interval[0] > 0.5) if confidence_interval else None
        ),
        "win_rate_confidently_below_50pct": (
            bool(confidence_interval[1] < 0.5) if confidence_interval else None
        ),
        "more_wins_than_losses": wins > losses,
        "mean_net_trade_return": float(trade_returns.mean()) if trades else None,
        "median_net_trade_return": float(np.median(trade_returns)) if trades else None,
        "profit_factor": total_wins / total_losses if total_losses else None,
        "cumulative_return": float(np.prod(1 + daily_returns) - 1),
        "max_drawdown": float(drawdowns.min()) if len(drawdowns) else 0.0,
        "annualized_sharpe": (
            float(np.mean(daily_returns) / daily_std * np.sqrt(252)) if daily_std > 0 else None
        ),
        "baselines": {
            "no_trade_cumulative_return": 0.0,
            "always_long": {
                "trades": int(len(net_returns)),
                "wins": always_long_wins,
                "losses": always_long_losses,
                "win_rate": (
                    always_long_wins / len(net_returns) if len(net_returns) else None
                ),
                "cumulative_return": always_long_total,
            },
        },
    }


def _walk_forward_metrics(
    features: pd.DataFrame,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
    model_type: str = "hist_gradient_boosting",
    ridge_alpha: float = 10.0,
) -> dict:
    labeled = features.dropna(subset=["next_close"])
    dates = labeled.index.sort_values().unique()
    if len(dates) < 100:
        raise DataSufficiencyError("Not enough history for three walk-forward validation folds.")

    first_validation_index = int(len(dates) * 0.7)
    fold_starts = np.linspace(first_validation_index, len(dates) - 2, 4, dtype=int)
    predicted_closes = []
    actual_closes = []
    previous_closes = []
    validation_parts = []
    fold_metrics = []

    for fold_number in range(3):
        start_date = dates[fold_starts[fold_number]]
        end_date = dates[fold_starts[fold_number + 1]] if fold_number < 2 else None
        training = labeled.loc[labeled.index < start_date]
        validation = labeled.loc[labeled.index >= start_date]
        if end_date is not None:
            validation = validation.loc[validation.index < end_date]
        if training.empty or validation.empty:
            continue

        model_features = _available_features(training)
        model = _new_model(model_type, ridge_alpha)
        model.fit(
            training[model_features], training["next_close"] / training["close"] - 1
        )
        predicted_return = model.predict(validation[model_features])
        predicted_close = validation["close"].to_numpy() * (1 + predicted_return)
        actual_close = validation["next_close"].to_numpy()
        previous_close = validation["close"].to_numpy()
        trade_metrics = _trade_metrics(
            predicted_return, validation, entry_threshold, transaction_cost_bps
        )
        fold_metrics.append(
            {
                "validation_start": validation.index.min().date().isoformat(),
                "validation_rows": int(len(validation)),
                "mae": float(mean_absolute_error(actual_close, predicted_close)),
                "previous_close_mae": float(
                    mean_absolute_error(actual_close, previous_close)
                ),
                "trading": trade_metrics,
            }
        )
        predicted_closes.extend(predicted_close)
        actual_closes.extend(actual_close)
        previous_closes.extend(previous_close)
        validation_parts.append(validation)

    if not fold_metrics:
        raise DataSufficiencyError("Unable to construct non-empty walk-forward validation folds.")

    actual = np.asarray(actual_closes)
    predicted = np.asarray(predicted_closes)
    previous = np.asarray(previous_closes)
    actual_returns = actual / previous - 1
    predicted_returns = predicted / previous - 1
    validation_oos = pd.concat(validation_parts).sort_index()
    return {
        "model_type": model_type,
        "validation_folds": fold_metrics,
        "validation_rows": int(len(actual)),
        "mean_absolute_error": float(mean_absolute_error(actual, predicted)),
        "mean_absolute_return_error": float(
            mean_absolute_error(actual_returns, predicted_returns)
        ),
        "directional_accuracy": float(
            np.mean(np.sign(predicted - previous) == np.sign(actual - previous))
        ),
        "previous_close_baseline_mae": float(mean_absolute_error(actual, previous)),
        "previous_close_baseline_mean_absolute_return_error": float(
            np.mean(np.abs(actual_returns))
        ),
        "trading": _trade_metrics(
            predicted / previous - 1,
            validation_oos,
            entry_threshold,
            transaction_cost_bps,
        ),
    }


def _new_intraday_ridge_model():
    return _new_model("ridge", ridge_alpha=10.0)


def _walk_forward_intraday_comparison(
    features: pd.DataFrame,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
) -> dict:
    labeled = features.dropna(subset=["next_open", "next_close"])
    dates = labeled.index.sort_values().unique()
    if len(dates) < 100:
        raise DataSufficiencyError("Not enough development history for intraday model comparison.")

    first_validation_index = int(len(dates) * 0.7)
    fold_starts = np.linspace(first_validation_index, len(dates) - 2, 4, dtype=int)
    candidates = {
        "hist_gradient_boosting": _new_model,
        "ridge": _new_intraday_ridge_model,
    }
    collected = {
        name: {"predicted_returns": [], "actual_returns": [], "folds": []}
        for name in candidates
    }
    validation_parts = []

    for fold_number in range(3):
        start_date = dates[fold_starts[fold_number]]
        end_date = dates[fold_starts[fold_number + 1]] if fold_number < 2 else None
        training = labeled.loc[labeled.index < start_date]
        validation = labeled.loc[labeled.index >= start_date]
        if end_date is not None:
            validation = validation.loc[validation.index < end_date]
        if training.empty or validation.empty:
            continue

        model_features = _available_features(training)
        actual_returns = validation["next_close"] / validation["next_open"] - 1
        training_returns = training["next_close"] / training["next_open"] - 1
        validation_parts.append(validation)

        for name, model_factory in candidates.items():
            model = model_factory()
            model.fit(training[model_features], training_returns)
            predicted_returns = np.asarray(
                model.predict(validation[model_features])
            )
            fold_trading = _trade_metrics(
                predicted_returns,
                validation,
                entry_threshold,
                transaction_cost_bps,
            )
            collected[name]["predicted_returns"].extend(predicted_returns)
            collected[name]["actual_returns"].extend(actual_returns.to_numpy())
            collected[name]["folds"].append(
                {
                    "validation_start": validation.index.min().date().isoformat(),
                    "validation_rows": int(len(validation)),
                    "mean_absolute_return_error": float(
                        mean_absolute_error(actual_returns, predicted_returns)
                    ),
                    "directional_accuracy": float(
                        np.mean(np.sign(predicted_returns) == np.sign(actual_returns))
                    ),
                    "trading": fold_trading,
                }
            )

    if not validation_parts:
        raise DataSufficiencyError("Unable to construct intraday model comparison folds.")

    validation_oos = pd.concat(validation_parts).sort_index()
    actual_oos = validation_oos["next_close"] / validation_oos["next_open"] - 1
    comparisons = {}
    for name, values in collected.items():
        predicted_returns = np.asarray(values["predicted_returns"])
        comparisons[name] = {
            "target": "next-session open-to-close return",
            "validation_rows": int(len(actual_oos)),
            "mean_absolute_return_error": float(
                mean_absolute_error(actual_oos, predicted_returns)
            ),
            "directional_accuracy": float(
                np.mean(np.sign(predicted_returns) == np.sign(actual_oos))
            ),
            "validation_folds": values["folds"],
            "trading": _trade_metrics(
                predicted_returns,
                validation_oos,
                entry_threshold,
                transaction_cost_bps,
            ),
        }

    return comparisons


def _final_holdout_metrics(
    features: pd.DataFrame,
    holdout_sessions: int = FINAL_HOLDOUT_SESSIONS,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
    model_type: str = "hist_gradient_boosting",
    ridge_alpha: float = 10.0,
) -> tuple[dict, pd.DataFrame]:
    labeled = features.dropna(subset=["next_close"])
    if len(labeled) <= holdout_sessions + 100:
        raise DataSufficiencyError(
            f"Need more than {holdout_sessions + 100} labeled sessions for the final holdout."
        )

    holdout_dates = labeled.index[-holdout_sessions:]
    holdout_start = holdout_dates[0]
    development = labeled.loc[labeled.index < holdout_start]
    holdout = labeled.loc[labeled.index >= holdout_start]
    if development.empty or holdout.empty:
        raise RuntimeError("Unable to separate development history from final holdout.")

    model_features = _available_features(development)
    model = _new_model(model_type, ridge_alpha)
    model.fit(development[model_features], development["next_close"] / development["close"] - 1)
    predicted_return = model.predict(holdout[model_features])
    predicted_close = holdout["close"].to_numpy() * (1 + predicted_return)
    trade_log = _trade_results(
        predicted_return, holdout, entry_threshold, transaction_cost_bps
    )
    actual_close = holdout["next_close"].to_numpy()
    previous_close = holdout["close"].to_numpy()
    return (
        {
            "model_type": model_type,
            "period_start": holdout.index.min().date().isoformat(),
            "period_end": holdout.index.max().date().isoformat(),
            "training_through": development.index.max().date().isoformat(),
            "sessions": int(len(holdout)),
            "mae": float(mean_absolute_error(actual_close, predicted_close)),
            "mean_absolute_return_error": float(
                mean_absolute_error(actual_close / previous_close - 1, predicted_return)
            ),
            "previous_close_baseline_mean_absolute_return_error": float(
                np.mean(np.abs(actual_close / previous_close - 1))
            ),
            "previous_close_baseline_mae": float(
                mean_absolute_error(actual_close, previous_close)
            ),
            "directional_accuracy": float(
                np.mean(np.sign(predicted_close - previous_close) == np.sign(actual_close - previous_close))
            ),
            "trading": _trade_metrics(
                predicted_return, holdout, entry_threshold, transaction_cost_bps
            ),
        },
        trade_log,
    )


def compare_models(
    ticker: str,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
) -> None:
    ticker = _normalize_ticker(ticker)
    price_path = _ticker_file(ticker)
    if not price_path.exists():
        raise RuntimeError(f"No local data for {ticker}. Run `python -m sp500_ml update` first.")

    prices = pd.read_csv(price_path, parse_dates=["Date"], index_col="Date")
    fundamentals, fundamentals_warning = _load_training_fundamentals(
        ticker, prices.index
    )
    features = build_features(prices, _load_market_data(), fundamentals)
    labeled = features.dropna(subset=["next_close"])
    if len(labeled) <= FINAL_HOLDOUT_SESSIONS + 100:
        raise RuntimeError("Not enough history for a development comparison and final holdout.")

    development = features.loc[
        features.index < labeled.index[-FINAL_HOLDOUT_SESSIONS]
    ]
    report = {
        "ticker": ticker,
        "fundamentals_included": fundamentals is not None,
        "fundamentals_warning": fundamentals_warning,
        "development_period_start": development.index.min().date().isoformat(),
        "development_period_end": development.index.max().date().isoformat(),
        "final_holdout_used": False,
        "results": _walk_forward_intraday_comparison(
            development,
            entry_threshold,
            transaction_cost_bps,
        ),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / f"{ticker}_intraday_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved development comparison to {report_path.relative_to(ROOT)}")
    if fundamentals_warning:
        print(f"SEC fundamentals note: {fundamentals_warning}")


def _summarize_universe_comparisons(per_ticker: dict[str, dict]) -> dict:
    candidate_names = (
        "hist_gradient_boosting",
        "ridge",
    )
    summary = {}
    for name in candidate_names:
        candidate_results = [
            result["candidates"][name]
            for result in per_ticker.values()
            if name in result["candidates"]
        ]
        total_trades = sum(result["trades"] for result in candidate_results)
        total_wins = sum(result["wins"] for result in candidate_results)
        total_losses = sum(result["losses"] for result in candidate_results)
        error_values = [result["return_mae"] for result in candidate_results]
        direction_values = [result["directional_accuracy"] for result in candidate_results]
        return_values = [result["net_return"] for result in candidate_results]
        summary[name] = {
            "tickers_compared": len(candidate_results),
            "mean_ticker_return_mae": (
                float(np.mean(error_values)) if error_values else None
            ),
            "median_ticker_return_mae": (
                float(np.median(error_values)) if error_values else None
            ),
            "mean_ticker_directional_accuracy": (
                float(np.mean(direction_values)) if direction_values else None
            ),
            "median_ticker_net_return": (
                float(np.median(return_values)) if return_values else None
            ),
            "profitable_tickers": int(sum(value > 0 for value in return_values)),
            "total_trades": int(total_trades),
            "wins": int(total_wins),
            "losses": int(total_losses),
            "pooled_win_rate": (
                total_wins / total_trades if total_trades else None
            ),
        }
    return summary


def compare_all_models(
    limit: int | None = None,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
) -> None:
    tickers = [path.stem for path in _price_files() if path.stem != "SPY"]
    if limit is not None:
        if limit < 1:
            raise ValueError("Ticker limit must be positive.")
        tickers = tickers[:limit]
    if not tickers:
        raise RuntimeError("No local ticker prices found. Run `python -m sp500_ml update` first.")

    market = _load_market_data()
    per_ticker = {}
    skipped = {}
    for position, ticker in enumerate(tickers, start=1):
        print(f"[{position}/{len(tickers)}] Comparing {ticker}...")
        try:
            prices = pd.read_csv(
                _ticker_file(ticker), parse_dates=["Date"], index_col="Date"
            )
            features = build_features(prices, market)
            labeled = features.dropna(subset=["next_open", "next_close"])
            if len(labeled) <= FINAL_HOLDOUT_SESSIONS + 100:
                raise DataSufficiencyError("Not enough history before the final holdout.")

            development = features.loc[
                features.index < labeled.index[-FINAL_HOLDOUT_SESSIONS]
            ]
            results = _walk_forward_intraday_comparison(
                development,
                entry_threshold,
                transaction_cost_bps,
            )
            per_ticker[ticker] = {
                "development_start": development.index.min().date().isoformat(),
                "development_end": development.index.max().date().isoformat(),
                "labeled_sessions": int(len(labeled)),
                "candidates": {
                    name: {
                        "return_mae": result["mean_absolute_return_error"],
                        "directional_accuracy": result["directional_accuracy"],
                        "trades": result["trading"]["trades"],
                        "wins": result["trading"]["wins"],
                        "losses": result["trading"]["losses"],
                        "win_rate": result["trading"]["win_rate"],
                        "net_return": result["trading"]["cumulative_return"],
                    }
                    for name, result in results.items()
                },
            }
        except _SKIPPABLE_ERRORS as error:
            skipped[ticker] = f"{type(error).__name__}: {error}"

    if not per_ticker:
        raise RuntimeError("No tickers had sufficient history for a comparison.")
    report = {
        "universe": "locally downloaded S&P 500 tickers",
        "requested_tickers": len(tickers),
        "compared_tickers": len(per_ticker),
        "skipped_tickers": skipped,
        "features": "price and SPY market context; SEC fundamentals excluded for uniformity",
        "final_holdout_used": False,
        "entry_threshold": entry_threshold,
        "transaction_cost_bps_per_side": transaction_cost_bps,
        "aggregate": _summarize_universe_comparisons(per_ticker),
        "per_ticker": per_ticker,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / "universe_intraday_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(
        f"Compared {len(per_ticker)} of {len(tickers)} tickers; "
        f"saved report to {report_path.relative_to(ROOT)}"
    )


def _metric_snapshot(metrics: dict, mae_key: str) -> dict:
    trading = metrics.get("trading", {})
    return {
        "mae": metrics.get(mae_key),
        "directional_accuracy": metrics.get("directional_accuracy"),
        "trades": trading.get("trades"),
        "win_rate": trading.get("win_rate"),
        "cumulative_return": trading.get("cumulative_return"),
    }


def _metric_deltas(old: dict, new: dict) -> dict:
    return {
        key: (
            new[key] - old[key]
            if isinstance(old.get(key), (int, float))
            and isinstance(new.get(key), (int, float))
            else None
        )
        for key in old
    }


def _load_saved_holdout(metrics_path: Path, old_metrics: dict) -> dict:
    inline = old_metrics.get("final_holdout")
    if isinstance(inline, dict):
        return inline
    holdout_path = _holdout_report_path(metrics_path)
    if holdout_path.exists():
        return json.loads(holdout_path.read_text(encoding="utf-8"))
    raise DataSufficiencyError(f"No final holdout report found for {metrics_path.name}.")


def reassess_saved_models() -> dict:
    """Recompute saved model reports using signal-time return decisions."""
    market = _load_market_data()
    results = {}
    skipped = {}
    metric_paths = sorted(MODEL_DIR.glob("*_metrics.json"))
    for metrics_path in metric_paths:
        try:
            old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            stem = metrics_path.name.removesuffix("_metrics.json")
            model_type = old_metrics.get("model_type") or (
                "ridge" if stem.endswith("_ridge") else "hist_gradient_boosting"
            )
            ticker = old_metrics.get("ticker") or stem.removesuffix("_ridge")
            prices = _drop_incomplete_session(
                pd.read_csv(
                    _ticker_file(ticker), parse_dates=["Date"], index_col="Date"
                )
            )
            fundamentals = None
            if old_metrics.get("fundamentals_included"):
                if not _fundamental_file(ticker).exists():
                    raise RuntimeError("The original SEC fundamentals cache is unavailable.")
                fundamentals = _read_fundamentals(ticker, prices.index)
            features = build_features(prices, market, fundamentals)
            old_holdout = _load_saved_holdout(metrics_path, old_metrics)
            old_holdout_end = old_holdout.get("period_end")
            if old_holdout_end:
                features = features.loc[
                    features.index <= pd.Timestamp(old_holdout_end)
                ]
            labeled = features.dropna(subset=["next_close"])
            development = features.loc[
                features.index < labeled.index[-FINAL_HOLDOUT_SESSIONS]
            ]
            ridge_alpha = old_metrics.get("ridge_alpha") or 10.0
            old_trading = old_metrics["development_walk_forward"]["trading"]
            entry_threshold = old_trading.get("entry_threshold_pct", 0.0) / 100
            transaction_cost_bps = old_trading.get(
                "transaction_cost_bps_per_side", 10.0
            )
            new_development = _walk_forward_metrics(
                development,
                entry_threshold=entry_threshold,
                transaction_cost_bps=transaction_cost_bps,
                model_type=model_type,
                ridge_alpha=ridge_alpha,
            )
            new_holdout, _ = _final_holdout_metrics(
                features,
                entry_threshold=entry_threshold,
                transaction_cost_bps=transaction_cost_bps,
                model_type=model_type,
                ridge_alpha=ridge_alpha,
            )
            old_development = old_metrics["development_walk_forward"]
            comparisons = {}
            for period, old_values, new_values in (
                ("development", old_development, new_development),
                ("final_holdout", old_holdout, new_holdout),
            ):
                old_snapshot = _metric_snapshot(old_values, "mean_absolute_return_error")
                new_snapshot = _metric_snapshot(new_values, "mean_absolute_return_error")
                comparisons[period] = {
                    "old": old_snapshot,
                    "reassessed": new_snapshot,
                    "delta": _metric_deltas(old_snapshot, new_snapshot),
                }
            results[f"{ticker}:{model_type}"] = comparisons
        except Exception as error:
            skipped[metrics_path.name] = f"{type(error).__name__}: {error}"

    report = {
        "signal_decision": "predicted return less round-trip costs, determined at signal close; no next-open data used for entry",
        "old_metrics_note": "Prior strategy/trading figures used next-open data to decide entry and are optimistic upper bounds, not like-for-like estimates.",
        "reports_found": len(metric_paths),
        "reports_reassessed": len(results),
        "skipped_reports": skipped,
        "results": results,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / "methodology_reassessment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Saved methodology reassessment to {report_path.relative_to(ROOT)}")
    return report


def _metrics_for_console(metrics: dict, holdout: dict, reveal_holdout: bool) -> dict:
    """Mask the final holdout by default so tuning can't quietly peek at it."""
    if reveal_holdout:
        return {**metrics, "final_holdout": holdout}
    return metrics
