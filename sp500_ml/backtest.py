"""Walk-forward validation, holdout evaluation, and model comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .cloud_data import read_csv
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from .data import _drop_incomplete_session, _normalize_ticker, _price_files, _ticker_file
from .features import _available_features, _load_market_data, build_features
from .fundamentals import _load_training_fundamentals
from .models import (
    MODEL_TYPES,
    MODEL_TARGET,
    MODEL_TARGETS,
    MODEL_TARGET_COLUMN,
    MODEL_TARGET_COLUMNS,
    _holdout_report_path,
    _new_model,
)
from .paths import MODEL_DIR, ROOT

FINAL_HOLDOUT_SESSIONS = 252
# Purged CV gap (sessions between train and validation to prevent leakage)
PURGE_GAP_SESSIONS = 5


class DataSufficiencyError(RuntimeError):
    """A ticker lacks enough usable history; safe to skip in batch runs."""


# Anything outside this tuple is a real bug and must propagate from batch runs.
_SKIPPABLE_ERRORS = (
    DataSufficiencyError,
    OSError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Filter features with insufficient variance for HistGradientBoosting."""
    def __init__(self, min_unique=2):
        self.min_unique = min_unique
        self.valid_features_ = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.valid_features_ = X.columns[X.nunique(dropna=True) >= self.min_unique].tolist()
        else:
            # numpy array
            self.valid_features_ = [i for i in range(X.shape[1])
                                   if len(np.unique(X[:, i][~np.isnan(X[:, i])])) >= self.min_unique]
        return self

    def transform(self, X):
        if self.valid_features_ is None:
            return X
        if hasattr(X, 'columns'):
            return X[self.valid_features_]
        return X[:, self.valid_features_]


def _fit_tuned_model(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    ridge_alpha: float = 10.0,
    purge_gap: int = PURGE_GAP_SESSIONS,
):
    if model_type == "hybrid":
        model = _new_model(model_type, ridge_alpha).set_params(purge_gap=purge_gap).fit(X, y)
        return model, {"ridge_alpha": ridge_alpha, **model.calibration_}
    parameter_grid = (
        {"ridge__alpha": sorted({0.1, 1.0, 10.0, 100.0, ridge_alpha})}
        if model_type == "ridge"
        else {
            "max_iter": [150, 300],
            "max_leaf_nodes": [7, 15],
        }
    )

    # Use pipeline with variance filter for HistGradientBoosting
    if model_type == "hist_gradient_boosting":
        pipeline = Pipeline([
            ("variance_filter", VarianceFilter(min_unique=2)),
            ("model", _new_model(model_type, ridge_alpha)),
        ])
        # Adjust parameter grid for pipeline
        param_grid = {f"model__{k}": v for k, v in parameter_grid.items()}
    else:
        pipeline = _new_model(model_type, ridge_alpha)
        param_grid = parameter_grid

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=2, gap=purge_gap),
        n_jobs=1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


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
    target_key: str = "open_to_close",
) -> pd.DataFrame:
    predicted_returns = np.asarray(predicted_returns, dtype=float)
    target_column = MODEL_TARGET_COLUMNS[target_key]
    if target_column not in validation and target_key == "open_to_close":
        validation = validation.copy()
        validation[target_column] = validation["next_close"] / validation["next_open"] - 1
    valid = validation[target_column].notna() & validation["next_open"].notna()
    validation = validation.loc[valid]
    predicted_returns = predicted_returns[valid.to_numpy()]
    horizon = 5 if target_key == "open_to_close_5d" else 1
    if horizon > 1:
        positions = np.arange(0, len(validation), horizon)
        validation = validation.iloc[positions]
        predicted_returns = predicted_returns[positions]
    next_open = validation["next_open"].to_numpy()
    exit_close = validation[
        "next_5d_close" if horizon > 1 else "next_close"
    ].to_numpy()
    cost = 2 * transaction_cost_bps / 10_000
    signal_edge_after_costs = predicted_returns - cost
    gross_returns = validation[target_column].to_numpy()
    net_returns = gross_returns - cost
    execution_dates = validation.get(
        "next_date", pd.Series(pd.NaT, index=validation.index)
    )
    return pd.DataFrame(
        {
            "signal_date": validation.index,
            "execution_date": execution_dates.to_numpy(),
            "exit_date": validation.get(
                "next_5d_date" if horizon > 1 else "next_date",
                pd.Series(pd.NaT, index=validation.index),
            ).to_numpy(),
            "predicted_return": predicted_returns,
            "next_open": next_open,
            "next_close": exit_close,
            "signal_edge_after_costs": signal_edge_after_costs,
            "enter_long": signal_edge_after_costs > entry_threshold,
            "gross_open_to_close_return": gross_returns,
            "gross_target_return": gross_returns,
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
    target_key: str = "open_to_close",
) -> dict:
    trade_log = _trade_results(
        predicted_returns, validation, entry_threshold, transaction_cost_bps, target_key
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
            float(
                np.mean(daily_returns)
                / daily_std
                * np.sqrt(252 / (5 if target_key == "open_to_close_5d" else 1))
            )
            if daily_std > 0
            else None
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
    target_key: str = "open_to_close",
    n_splits: int = 5,
    purge_gap: int = PURGE_GAP_SESSIONS,
) -> dict:
    """Purged walk-forward cross-validation with configurable gap."""
    if target_key not in MODEL_TARGET_COLUMNS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGET_COLUMNS.keys())}")
    target_col = MODEL_TARGET_COLUMNS[target_key]
    target_name = MODEL_TARGETS[target_key]
    target_end_column = "next_date" if target_key == "open_to_close" else "next_5d_date"

    labeled = features.dropna(subset=[target_col])
    dates = labeled.index.sort_values().unique()
    if len(dates) < 100:
        raise DataSufficiencyError("Not enough history for purged walk-forward validation.")

    test_size = max(1, len(dates) // (n_splits * 3))
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=purge_gap,
        test_size=test_size,
    )
    labeled_dates = labeled.index.sort_values().unique()

    predicted_returns = []
    actual_returns = []
    validation_parts = []
    fold_metrics = []

    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(labeled_dates)):
        train_dates = labeled_dates[train_idx]
        val_dates = labeled_dates[val_idx]

        if len(train_dates) == 0 or len(val_dates) == 0:
            continue

        training = labeled.loc[train_dates]
        validation = labeled.loc[val_dates]

        if training.empty or validation.empty:
            continue

        model_features = _available_features(training)
        model, best_params = _fit_tuned_model(
            model_type,
            training[model_features],
            training[target_col],
            ridge_alpha,
            purge_gap,
        )
        predicted_return = model.predict(validation[model_features])
        actual_return = validation[target_col].to_numpy()

        # For trading metrics, we need next_open and next_close
        predicted_close = validation["next_open"].to_numpy() * (1 + predicted_return)
        actual_close = validation["next_close"].to_numpy()
        trade_metrics = _trade_metrics(
            predicted_return,
            validation,
            entry_threshold,
            transaction_cost_bps,
            target_key,
        )
        fold_metrics.append(
            {
                "fold": fold_number,
                "validation_start": validation.index.min().date().isoformat(),
                "validation_end": validation.index.max().date().isoformat(),
                "training_target_end": training[target_end_column]
                .max()
                .date()
                .isoformat(),
                "validation_rows": int(len(validation)),
                "train_rows": int(len(training)),
                "purge_gap_sessions": purge_gap,
                "best_params": best_params,
                "mean_absolute_return_error": float(
                    mean_absolute_error(actual_return, predicted_return)
                ),
                "zero_return_baseline_mean_absolute_return_error": float(
                    np.mean(np.abs(actual_return))
                ),
                "open_price_mae": (
                    float(mean_absolute_error(actual_close, predicted_close))
                    if target_key == "open_to_close"
                    else None
                ),
                "trading": trade_metrics,
            }
        )
        predicted_returns.extend(predicted_return)
        actual_returns.extend(actual_return)
        validation_parts.append(validation)

    if not fold_metrics:
        raise DataSufficiencyError("Unable to construct non-empty purged walk-forward validation folds.")

    actual = np.asarray(actual_returns)
    predicted = np.asarray(predicted_returns)
    validation_oos = pd.concat(validation_parts).sort_index()
    return {
        "model_type": model_type,
        "target_key": target_key,
        "target": target_name,
        "n_splits": n_splits,
        "purge_gap_sessions": purge_gap,
        "validation_folds": fold_metrics,
        "validation_rows": int(len(actual)),
        "mean_absolute_return_error": float(
            mean_absolute_error(actual, predicted)
        ),
        "zero_return_baseline_mean_absolute_return_error": float(
            np.mean(np.abs(actual))
        ),
        "directional_accuracy": float(
            np.mean(np.sign(predicted) == np.sign(actual))
        ),
        "trading": _trade_metrics(
            predicted,
            validation_oos,
            entry_threshold,
            transaction_cost_bps,
            target_key,
        ),
    }


def _new_intraday_ridge_model():
    return _new_model("ridge", ridge_alpha=10.0)


def _walk_forward_intraday_comparison(
    features: pd.DataFrame,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
    target_key: str = "open_to_close",
) -> dict:
    if target_key not in MODEL_TARGET_COLUMNS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGET_COLUMNS.keys())}")
    target_col = MODEL_TARGET_COLUMNS[target_key]
    target_name = MODEL_TARGETS[target_key]

    labeled = features.dropna(subset=["next_open", "next_close", target_col])
    dates = labeled.index.sort_values().unique()
    if len(dates) < 100:
        raise DataSufficiencyError("Not enough development history for intraday model comparison.")

    test_size = max(1, len(dates) // 15)
    splitter = TimeSeriesSplit(n_splits=5, gap=PURGE_GAP_SESSIONS, test_size=test_size)
    labeled_dates = labeled.index.sort_values().unique()
    candidates = MODEL_TYPES
    collected = {
        name: {"predicted_returns": [], "actual_returns": [], "folds": []}
        for name in candidates
    }
    validation_parts = []

    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(labeled_dates)):
        training = labeled.loc[labeled_dates[train_idx]]
        validation = labeled.loc[labeled_dates[val_idx]]
        if training.empty or validation.empty:
            continue

        model_features = _available_features(training)
        actual_returns = validation[target_col]
        training_returns = training[target_col]
        validation_parts.append(validation)

        for name in candidates:
            model, best_params = _fit_tuned_model(
                name,
                training[model_features],
                training_returns,
                purge_gap=PURGE_GAP_SESSIONS,
            )
            predicted_returns = np.asarray(
                model.predict(validation[model_features])
            )
            fold_trading = _trade_metrics(
                predicted_returns,
                validation,
                entry_threshold,
                transaction_cost_bps,
                target_key,
            )
            collected[name]["predicted_returns"].extend(predicted_returns)
            collected[name]["actual_returns"].extend(actual_returns.to_numpy())
            collected[name]["folds"].append(
                {
                    "validation_start": validation.index.min().date().isoformat(),
                    "validation_rows": int(len(validation)),
                    "purge_gap_sessions": PURGE_GAP_SESSIONS,
                    "best_params": best_params,
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
    actual_oos = validation_oos[target_col]
    comparisons = {}
    for name, values in collected.items():
        predicted_returns = np.asarray(values["predicted_returns"])
        comparisons[name] = {
            "target_key": target_key,
            "target": target_name,
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
                target_key,
            ),
        }

    comparisons["zero_return_baseline"] = {
        "target_key": target_key,
        "target": target_name,
        "validation_rows": int(len(actual_oos)),
        "mean_absolute_return_error": float(np.mean(np.abs(actual_oos))),
        "directional_accuracy": None,
        "validation_folds": [],
        "trading": _trade_metrics(
            np.zeros(len(actual_oos)), validation_oos,
            0.0, transaction_cost_bps, target_key,
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
    target_key: str = "open_to_close",
) -> tuple[dict, pd.DataFrame]:
    if target_key not in MODEL_TARGET_COLUMNS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGET_COLUMNS.keys())}")
    target_col = MODEL_TARGET_COLUMNS[target_key]
    target_name = MODEL_TARGETS[target_key]
    labeled = features.dropna(subset=[target_col])
    if len(labeled) <= holdout_sessions + 100:
        raise DataSufficiencyError(
            f"Need more than {holdout_sessions + 100} labeled sessions for the final holdout."
        )

    holdout_dates = labeled.index[-holdout_sessions:]
    holdout_start = holdout_dates[0]
    target_end_column = "next_date" if target_key == "open_to_close" else "next_5d_date"
    development = labeled.loc[labeled[target_end_column] < holdout_start]
    holdout = labeled.loc[labeled.index >= holdout_start]
    if development.empty or holdout.empty:
        raise RuntimeError("Unable to separate development history from final holdout.")

    model_features = _available_features(development)
    model, best_params = _fit_tuned_model(
        model_type,
        development[model_features],
        development[target_col],
        ridge_alpha,
    )
    predicted_return = model.predict(holdout[model_features])
    predicted_close = holdout["next_open"].to_numpy() * (1 + predicted_return)
    trade_log = _trade_results(
        predicted_return,
        holdout,
        entry_threshold,
        transaction_cost_bps,
        target_key,
    )
    actual_close = holdout["next_close"].to_numpy()
    actual_return = holdout[target_col].to_numpy()
    return (
        {
            "model_type": model_type,
            "target_key": target_key,
            "target": target_name,
            "best_params": best_params,
            "period_start": holdout.index.min().date().isoformat(),
            "period_end": holdout.index.max().date().isoformat(),
            "training_through": development.index.max().date().isoformat(),
            "sessions": int(len(holdout)),
            "mae": (
                float(mean_absolute_error(actual_close, predicted_close))
                if target_key == "open_to_close"
                else None
            ),
            "mean_absolute_return_error": float(
                mean_absolute_error(actual_return, predicted_return)
            ),
            "zero_return_baseline_mean_absolute_return_error": float(
                np.mean(np.abs(actual_return))
            ),
            "open_price_baseline_mae": (
                float(mean_absolute_error(actual_close, holdout["next_open"].to_numpy()))
                if target_key == "open_to_close"
                else None
            ),
            "directional_accuracy": float(
                np.mean(np.sign(predicted_return) == np.sign(actual_return))
            ),
            "trading": _trade_metrics(
                predicted_return,
                holdout,
                entry_threshold,
                transaction_cost_bps,
                target_key,
            ),
        },
        trade_log,
    )


def compare_models(
    ticker: str,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
    target_key: str = "open_to_close",
) -> None:
    ticker = _normalize_ticker(ticker)
    if target_key not in MODEL_TARGETS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGETS.keys())}")
    price_path = _ticker_file(ticker)
    if not price_path.exists():
        raise RuntimeError(f"No local data for {ticker}. Run `python -m sp500_ml update` first.")

    prices = read_csv(price_path, parse_dates=["Date"], index_col="Date")
    fundamentals, fundamentals_warning = _load_training_fundamentals(
        ticker, prices.index
    )
    features = build_features(prices, _load_market_data(), fundamentals)
    target_col = MODEL_TARGET_COLUMNS[target_key]
    labeled = features.dropna(subset=[target_col])
    if len(labeled) <= FINAL_HOLDOUT_SESSIONS + 100:
        raise RuntimeError("Not enough history for a development comparison and final holdout.")

    holdout_start = labeled.index[-FINAL_HOLDOUT_SESSIONS]
    target_end_column = "next_date" if target_key == "open_to_close" else "next_5d_date"
    development = features.loc[features[target_end_column] < holdout_start]
    report = {
        "ticker": ticker,
        "fundamentals_included": fundamentals is not None,
        "fundamentals_warning": fundamentals_warning,
        "target_key": target_key,
        "target": MODEL_TARGETS[target_key],
        "development_period_start": development.index.min().date().isoformat(),
        "development_period_end": development.index.max().date().isoformat(),
        "final_holdout_used": False,
        "results": _walk_forward_intraday_comparison(
            development,
            entry_threshold,
            transaction_cost_bps,
            target_key=target_key,
        ),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_name = (
        f"{ticker}_intraday_comparison.json"
        if target_key == "open_to_close"
        else f"{ticker}_{target_key}_intraday_comparison.json"
    )
    report_path = MODEL_DIR / report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved development comparison to {report_path.relative_to(ROOT)}")
    if fundamentals_warning:
        print(f"SEC fundamentals note: {fundamentals_warning}")


def _summarize_universe_comparisons(per_ticker: dict[str, dict]) -> dict:
    candidate_names = (*MODEL_TYPES, "zero_return_baseline")
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
        direction_values = [result["directional_accuracy"] for result in candidate_results if result["directional_accuracy"] is not None]
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
            "tickers_beating_zero_return_mae": sum(
                result.get("beats_zero_return_mae", False) for result in candidate_results
            ),
            "tickers_beating_always_long": sum(
                result.get("beats_always_long", False) for result in candidate_results
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
    tickers: list[str] | None = None,
) -> None:
    available = [path.stem for path in _price_files() if path.stem != "SPY"]
    if tickers is not None:
        tickers = list(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers))
        missing = sorted(set(tickers) - set(available))
        if missing:
            raise ValueError(f"No local stock history for: {', '.join(missing)}")
    else:
        tickers = available
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
            prices = read_csv(
                _ticker_file(ticker), parse_dates=["Date"], index_col="Date"
            )
            prices = _drop_incomplete_session(prices)
            features = build_features(prices, market)
            labeled = features.dropna(subset=["next_open", "next_close"])
            if len(labeled) <= FINAL_HOLDOUT_SESSIONS + 100:
                raise DataSufficiencyError("Not enough history before the final holdout.")

            holdout_start = labeled.index[-FINAL_HOLDOUT_SESSIONS]
            development = features.loc[features["next_date"] < holdout_start]
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
                        "zero_return_baseline_mae": results["zero_return_baseline"]["mean_absolute_return_error"],
                        "beats_zero_return_mae": result["mean_absolute_return_error"] < results["zero_return_baseline"]["mean_absolute_return_error"],
                        "always_long_net_return": result["trading"]["baselines"]["always_long"]["cumulative_return"],
                        "beats_always_long": result["trading"]["cumulative_return"] > result["trading"]["baselines"]["always_long"]["cumulative_return"],
                    }
                    for name, result in results.items()
                },
            }
        except _SKIPPABLE_ERRORS as error:
            skipped[ticker] = f"{type(error).__name__}: {error}"

    if not per_ticker:
        raise RuntimeError("No tickers had sufficient history for a comparison.")
    report = {
        "protocol_version": 1,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "target_key": "open_to_close",
        "n_splits": 5,
        "purge_gap_sessions": PURGE_GAP_SESSIONS,
        "reserved_holdout_sessions": FINAL_HOLDOUT_SESSIONS,
        "selected_tickers": tickers,
        "universe": "locally downloaded S&P 500 tickers",
        "requested_tickers": len(tickers),
        "compared_tickers": len(per_ticker),
        "skipped_tickers": skipped,
        "features": "price, SPY, and locally available sector/VIX context; SEC fundamentals excluded for uniformity",
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
        "zero_return_baseline_mae": metrics.get(
            "zero_return_baseline_mean_absolute_return_error"
        ),
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
                read_csv(
                    _ticker_file(ticker), parse_dates=["Date"], index_col="Date"
                )
            )
            fundamentals = None
            if old_metrics.get("fundamentals_included"):
                fundamentals, fundamentals_warning = _load_training_fundamentals(
                    ticker, prices.index
                )
                if fundamentals is None:
                    raise DataSufficiencyError(
                        "The original SEC fundamentals are unavailable; refusing to "
                        "reassess this model without its trained feature set."
                    )
            features = build_features(prices, market, fundamentals)
            old_holdout = _load_saved_holdout(metrics_path, old_metrics)
            old_holdout_end = old_holdout.get("period_end")
            if old_holdout_end:
                features = features.loc[
                    features.index <= pd.Timestamp(old_holdout_end)
                ]
            labeled = features.dropna(subset=[MODEL_TARGET_COLUMN])
            holdout_start = labeled.index[-FINAL_HOLDOUT_SESSIONS]
            development = features.loc[features["next_date"] < holdout_start]
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
            results[f"{ticker}:{model_type}:{stem}"] = {
                "old_target": old_metrics.get("target", "next-session close-to-close return"),
                "reassessed_target": MODEL_TARGET,
                **comparisons,
            }
        except Exception as error:
            skipped[metrics_path.name] = f"{type(error).__name__}: {error}"

    report = {
        "signal_decision": f"{MODEL_TARGET} prediction less round-trip costs, determined at signal close; no next-open value is used to decide entry",
        "old_metrics_note": "Previous metrics trained on close-to-close returns and used next-open information to choose entries. They are not like-for-like with the new open-to-close target and may be optimistic.",
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


def ablate_fundamental_features() -> dict:
    """Compare price-only and price-plus-growth fundamentals on development folds."""
    market = _load_market_data()
    results = {}
    skipped = {}
    metric_paths = sorted(MODEL_DIR.glob("*_metrics.json"))
    for metrics_path in metric_paths:
        try:
            old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not old_metrics.get("fundamentals_included"):
                continue
            stem = metrics_path.name.removesuffix("_metrics.json")
            model_type = old_metrics.get("model_type") or (
                "ridge" if stem.endswith("_ridge") else "hist_gradient_boosting"
            )
            ticker = old_metrics.get("ticker") or stem.removesuffix("_ridge")
            prices = _drop_incomplete_session(
                read_csv(
                    _ticker_file(ticker), parse_dates=["Date"], index_col="Date"
                )
            )
            fundamentals, warning = _load_training_fundamentals(
                ticker, prices.index
            )
            if fundamentals is None:
                raise DataSufficiencyError(
                    "Cannot compare fundamental features without a current SEC cache."
                )
            features_with_fundamentals = build_features(prices, market, fundamentals)
            features_price_only = build_features(prices, market, None)
            old_holdout = _load_saved_holdout(metrics_path, old_metrics)
            holdout_start = old_holdout.get("period_start")
            if holdout_start:
                cutoff = pd.Timestamp(holdout_start)
                features_with_fundamentals = features_with_fundamentals.loc[
                    features_with_fundamentals["next_date"] < cutoff
                ]
                features_price_only = features_price_only.loc[
                    features_price_only["next_date"] < cutoff
                ]
            old_trading = old_metrics["development_walk_forward"]["trading"]
            entry_threshold = old_trading.get("entry_threshold_pct", 0.0) / 100
            transaction_cost_bps = old_trading.get(
                "transaction_cost_bps_per_side", 10.0
            )
            ridge_alpha = old_metrics.get("ridge_alpha") or 10.0
            with_fundamentals = _walk_forward_metrics(
                features_with_fundamentals,
                entry_threshold=entry_threshold,
                transaction_cost_bps=transaction_cost_bps,
                model_type=model_type,
                ridge_alpha=ridge_alpha,
            )
            price_only = _walk_forward_metrics(
                features_price_only,
                entry_threshold=entry_threshold,
                transaction_cost_bps=transaction_cost_bps,
                model_type=model_type,
                ridge_alpha=ridge_alpha,
            )
            results[f"{ticker}:{model_type}"] = {
                "target": MODEL_TARGET,
                "development_through": features_with_fundamentals.index.max().date().isoformat(),
                "fundamentals_warning": warning,
                "price_only": _metric_snapshot(price_only, "mean_absolute_return_error"),
                "growth_and_ratios": _metric_snapshot(
                    with_fundamentals, "mean_absolute_return_error"
                ),
                "delta_fundamentals_minus_price_only": _metric_deltas(
                    _metric_snapshot(price_only, "mean_absolute_return_error"),
                    _metric_snapshot(with_fundamentals, "mean_absolute_return_error"),
                ),
            }
        except _SKIPPABLE_ERRORS as error:
            skipped[metrics_path.name] = f"{type(error).__name__}: {error}"

    report = {
        "target": MODEL_TARGET,
        "final_holdout_used": False,
        "feature_sets": {
            "price_only": "stock price and SPY market features",
            "growth_and_ratios": "price features plus 252-session signed-log changes and SEC margins/leverage ratios",
        },
        "reports_found": len(metric_paths),
        "models_compared": len(results),
        "skipped_models": skipped,
        "results": results,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / "fundamental_feature_ablation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Saved feature ablation to {report_path.relative_to(ROOT)}")
    return report


def _metrics_for_console(metrics: dict, holdout: dict, reveal_holdout: bool) -> dict:
    """Mask the final holdout by default so tuning can't quietly peek at it."""
    if reveal_holdout:
        return {**metrics, "final_holdout": holdout}
    return metrics
