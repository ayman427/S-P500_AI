"""Model training, artifact persistence, and next-close forecasting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from .backtest import (
    FINAL_HOLDOUT_SESSIONS,
    _final_holdout_metrics,
    _metrics_for_console,
    _walk_forward_metrics,
)
from .data import _drop_incomplete_session, _normalize_ticker, _ticker_file
from .feature_names import FUNDAMENTAL_FEATURES
from .features import (
    _available_features,
    _load_market_data,
    build_features,
)
from .fundamentals import _load_fundamentals, _load_training_fundamentals
from .models import (
    MODEL_TYPES,
    _holdout_report_path,
    _model_artifact_paths,
    _new_model,
    _prediction_output_path,
)
from .paths import BACKTEST_DIR, MODEL_DIR, ROOT

STALE_FORECAST_SESSIONS = 3


def train_model(
    ticker: str,
    entry_threshold: float = 0.0,
    transaction_cost_bps: float = 10.0,
    model_type: str = "hist_gradient_boosting",
    ridge_alpha: float = 10.0,
    reveal_holdout: bool = False,
) -> None:
    ticker = _normalize_ticker(ticker)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model must be one of: {', '.join(MODEL_TYPES)}")
    if ridge_alpha <= 0:
        raise ValueError("Ridge alpha must be greater than zero.")
    price_path = _ticker_file(ticker)
    if not price_path.exists():
        raise RuntimeError(f"No local data for {ticker}. Run `python -m sp500_ml update` first.")

    from .paper import _paper_model_freeze_status  # paper imports training

    freeze = _paper_model_freeze_status(ticker, model_type)
    if freeze and freeze["still_frozen"]:
        print(
            f"Warning: the paper ledger {freeze['ledger']} is {freeze['age_days']} day(s) old "
            f"(freeze window {freeze['freeze_days']}). Retraining starts a new ledger for the "
            "new model; the old one stops receiving signals."
        )

    prices = _drop_incomplete_session(
        pd.read_csv(price_path, parse_dates=["Date"], index_col="Date")
    )
    market = _load_market_data()
    fundamentals, fundamentals_warning = _load_training_fundamentals(
        ticker, prices.index
    )
    features = build_features(prices, market, fundamentals)
    labeled = features.dropna(subset=["next_close"])
    if labeled.empty:
        raise RuntimeError(f"Not enough history to train a model for {ticker}.")

    development_features = features.loc[
        features.index < labeled.index[-FINAL_HOLDOUT_SESSIONS]
    ]
    metrics = {
        "ticker": ticker,
        "fundamentals_included": fundamentals is not None,
        "fundamentals_warning": fundamentals_warning,
        "development_walk_forward": _walk_forward_metrics(
            development_features,
            entry_threshold,
            transaction_cost_bps,
            model_type,
            ridge_alpha,
        ),
    }
    metrics["model_type"] = model_type
    metrics["ridge_alpha"] = ridge_alpha if model_type == "ridge" else None
    holdout_metrics, holdout_trades = _final_holdout_metrics(
        features,
        FINAL_HOLDOUT_SESSIONS,
        entry_threshold,
        transaction_cost_bps,
        model_type,
        ridge_alpha,
    )
    model_features = _available_features(labeled)
    model = _new_model(model_type, ridge_alpha)
    model.fit(
        labeled[model_features], labeled["next_close"] / labeled["close"] - 1
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    model_path, metrics_path, trades_path = _model_artifact_paths(ticker, model_type)
    holdout_path = _holdout_report_path(metrics_path)
    metrics["final_holdout_report"] = holdout_path.name
    joblib.dump(
        {
            "model": model,
            "features": model_features,
            "ticker": ticker,
            "model_type": model_type,
            "ridge_alpha": ridge_alpha if model_type == "ridge" else None,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    holdout_path.write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")
    holdout_trades.to_csv(trades_path, index=False, date_format="%Y-%m-%d")
    print(json.dumps(_metrics_for_console(metrics, holdout_metrics, reveal_holdout), indent=2))
    print(f"Saved model to {model_path.relative_to(ROOT)}")
    print(f"Saved sealed final holdout report to {holdout_path.relative_to(ROOT)}")
    print(f"Saved final holdout trades to {trades_path.relative_to(ROOT)}")
    if fundamentals_warning:
        print(f"SEC fundamentals note: {fundamentals_warning}")


def _forecast_latest_close(
    ticker: str, model_type: str = "hist_gradient_boosting"
) -> tuple[dict, str, Path]:
    """Load the trained model and forecast the next close from the latest completed session."""
    ticker = _normalize_ticker(ticker)
    model_path, _, _ = _model_artifact_paths(ticker, model_type)
    price_path = _ticker_file(ticker)
    if not model_path.exists():
        raise RuntimeError(f"No trained model for {ticker}. Run `python -m sp500_ml train --ticker {ticker}` first.")
    if not price_path.exists():
        raise RuntimeError(f"No local data for {ticker}. Run `python -m sp500_ml update` first.")

    bundle = joblib.load(model_path)
    if bundle["ticker"] != ticker or bundle.get("model_type", "hist_gradient_boosting") != model_type:
        raise RuntimeError(f"The saved model does not match {ticker} using {model_type}.")
    prices = _drop_incomplete_session(
        pd.read_csv(price_path, parse_dates=["Date"], index_col="Date")
    )
    uses_fundamentals = any(name in FUNDAMENTAL_FEATURES for name in bundle["features"])
    fundamentals = _load_fundamentals(ticker, prices.index) if uses_fundamentals else None
    features = build_features(prices, _load_market_data(), fundamentals)
    if features.empty:
        raise RuntimeError(f"Not enough history to predict a closing price for {ticker}.")

    latest = features.iloc[[-1]][bundle["features"]]
    last_close = float(features["close"].iloc[-1])
    predicted_return = float(bundle["model"].predict(latest)[0])
    record = {
        "ticker": ticker,
        "as_of": latest.index[0].date().isoformat(),
        "last_close": last_close,
        "predicted_return": predicted_return,
        "predicted_next_close": last_close * (1 + predicted_return),
    }
    fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return record, fingerprint, model_path


def _load_latest_prediction(
    ticker: str, model_type: str = "hist_gradient_boosting"
) -> tuple[dict, str]:
    record, fingerprint, _ = _forecast_latest_close(ticker, model_type)
    return record, fingerprint


def predict(
    ticker: str, model_type: str = "hist_gradient_boosting"
) -> None:
    record, _, _ = _forecast_latest_close(ticker, model_type)
    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).normalize().tz_localize(None)
    sessions_old = max(len(pd.bdate_range(record["as_of"], today)) - 1, 0)
    if sessions_old >= STALE_FORECAST_SESSIONS:
        print(
            f"Warning: forecast is as of {record['as_of']}, {sessions_old} sessions old. "
            "Run `python -m sp500_ml update` before trusting it."
        )
    result = pd.DataFrame([record])
    prediction_path = _prediction_output_path(model_type)
    result.to_csv(prediction_path, index=False)
    print(result.to_string(index=False, formatters={
        "last_close": "{:.2f}".format,
        "predicted_next_close": "{:.2f}".format,
    }))
    print(f"Saved predictions to {prediction_path.relative_to(ROOT)}")
