"""Model training, artifact persistence, and next-session return forecasting."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from .cloud_data import read_csv
from .cloud_data import write_csv

from .artifacts import publish_generation
from .backtest import (
    FINAL_HOLDOUT_SESSIONS,
    _fit_tuned_model,
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
    _artifact_prefix,
    ModelRetrainingRequired,
    MODEL_TYPES,
    MODEL_TARGETS,
    MODEL_TARGET_COLUMNS,
    _model_artifact_paths,
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
    target_key: str = "open_to_close",
) -> None:
    ticker = _normalize_ticker(ticker)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model must be one of: {', '.join(MODEL_TYPES)}")
    if target_key not in MODEL_TARGETS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGETS.keys())}")
    if ridge_alpha <= 0:
        raise ValueError("Ridge alpha must be greater than zero.")
    price_path = _ticker_file(ticker)
    if not price_path.exists():
        raise RuntimeError(f"No local data for {ticker}. Run `python -m sp500_ml update` first.")

    from .paper import _paper_model_freeze_status  # paper imports training

    try:
        freeze = _paper_model_freeze_status(ticker, model_type) if target_key == "open_to_close" else None
    except RuntimeError as error:
        print(f"Previous artifacts could not be verified; publishing a new training run: {error}")
        freeze = None
    if freeze and freeze["still_frozen"]:
        print(
            f"Warning: the paper ledger {freeze['ledger']} is {freeze['age_days']} day(s) old "
            f"(freeze window {freeze['freeze_days']}). Retraining starts a new ledger for the "
            "new model; the old one stops receiving signals."
        )

    prices = _drop_incomplete_session(
        read_csv(price_path, parse_dates=["Date"], index_col="Date")
    )
    market = _load_market_data()
    fundamentals, fundamentals_warning = _load_training_fundamentals(
        ticker, prices.index
    )
    features = build_features(prices, market, fundamentals)
    target_col = MODEL_TARGET_COLUMNS[target_key]
    target_name = MODEL_TARGETS[target_key]
    labeled = features.dropna(subset=[target_col])
    if len(labeled) <= FINAL_HOLDOUT_SESSIONS + 100:
        raise RuntimeError(
            f"Not enough completed history to train {ticker}. Need more than "
            f"{FINAL_HOLDOUT_SESSIONS + 100} labeled sessions; found {len(labeled)}."
        )

    holdout_start = labeled.index[-FINAL_HOLDOUT_SESSIONS]
    target_end_column = "next_date" if target_key == "open_to_close" else "next_5d_date"
    development_features = features.loc[
        features[target_end_column] < holdout_start
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
            target_key=target_key,
        ),
    }
    metrics["model_type"] = model_type
    metrics["target_key"] = target_key
    metrics["target"] = target_name
    metrics["ridge_alpha"] = ridge_alpha if model_type in {"ridge", "hybrid"} else None
    holdout_metrics, holdout_trades = _final_holdout_metrics(
        features,
        FINAL_HOLDOUT_SESSIONS,
        entry_threshold,
        transaction_cost_bps,
        model_type,
        ridge_alpha,
        target_key=target_key,
    )
    model_features = _available_features(labeled)
    model, best_params = _fit_tuned_model(
        model_type,
        labeled[model_features],
        labeled[target_col],
        ridge_alpha,
    )
    metrics["final_model_best_params"] = best_params
    if model_type == "hybrid":
        metrics["experimental"] = True
        metrics["learning_policy"] = {"mode": "owner_settled_prediction_errors", "max_feedback_outcomes": 60, "rule": "settled_error_v1"}
        metrics["hybrid_calibration"] = model.calibration_
        metrics["evaluation_note"] = "Historical metrics evaluate the initial calibrated blend. Settled-error learning requires prospective evaluation."

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _artifact_prefix(ticker, model_type, target_key)
    metrics["final_holdout_report"] = f"{prefix}_holdout.json"
    bundle = {
        "model": model,
        "features": model_features,
        "ticker": ticker,
        "model_type": model_type,
        "target_key": target_key,
        "target": target_name,
        "best_params": best_params,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "trained_through": labeled.index.max().date().isoformat(),
        "training_rows": len(labeled),
        "entry_threshold": entry_threshold,
        "transaction_cost_bps_per_side": transaction_cost_bps,
        "ridge_alpha": ridge_alpha if model_type in {"ridge", "hybrid"} else None,
    }
    if model_type == "hybrid":
        bundle["experimental"] = True
        bundle["learning_policy"] = metrics["learning_policy"]
        bundle["hybrid_calibration"] = model.calibration_
    model_path, metrics_path, holdout_path, trades_path = publish_generation(
        MODEL_DIR, prefix, bundle, metrics, holdout_metrics, holdout_trades
    )
    print(json.dumps(_metrics_for_console(metrics, holdout_metrics, reveal_holdout), indent=2))
    print(f"Saved model to {model_path.relative_to(ROOT)}")
    print(f"Saved sealed final holdout report to {holdout_path.relative_to(ROOT)}")
    print(f"Saved final holdout trades to {trades_path.relative_to(ROOT)}")
    if fundamentals_warning:
        print(f"SEC fundamentals note: {fundamentals_warning}")


def _forecast_latest_close(
    ticker: str, model_type: str = "hist_gradient_boosting", target_key: str = "open_to_close"
) -> tuple[dict, str, Path]:
    """Forecast next-session open-to-close return from the latest completed session."""
    ticker = _normalize_ticker(ticker)
    if target_key not in MODEL_TARGETS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGETS.keys())}")
    model_path, _, _ = _model_artifact_paths(ticker, model_type, target_key)
    price_path = _ticker_file(ticker)
    if not model_path.exists():
        raise RuntimeError(f"No trained model for {ticker}. Run `python -m sp500_ml train --ticker {ticker}` first.")
    if not price_path.exists():
        raise RuntimeError(f"No local data for {ticker}. Run `python -m sp500_ml update` first.")

    model_bytes = model_path.read_bytes()
    bundle = joblib.load(BytesIO(model_bytes))
    if bundle["ticker"] != ticker or bundle.get("model_type", "hist_gradient_boosting") != model_type:
        raise RuntimeError(f"The saved model does not match {ticker} using {model_type}.")
    if bundle.get("target_key") != target_key:
        raise ModelRetrainingRequired(
            f"The saved {model_type} model for {ticker} uses target '{bundle.get('target_key')}', "
            f"but requested target is '{target_key}'. Retrain it with the selected target."
        )
    prices = _drop_incomplete_session(
        read_csv(price_path, parse_dates=["Date"], index_col="Date")
    )
    uses_fundamentals = any(name in FUNDAMENTAL_FEATURES for name in bundle["features"])
    fundamentals = _load_fundamentals(ticker, prices.index) if uses_fundamentals else None
    features = build_features(prices, _load_market_data(), fundamentals)
    if features.empty:
        raise RuntimeError(f"Not enough history to predict a closing price for {ticker}.")

    # Never forward-fill a delayed market input into a newer signal session.
    input_as_of = features.attrs.get("input_as_of", features.index.max())
    features = features.loc[features.index <= input_as_of]
    if features.empty:
        raise RuntimeError(f"No common completed market session is available for {ticker}.")

    if bundle.get("trained_through") and pd.Timestamp(bundle["trained_through"]) > features.index[-1]:
        raise RuntimeError("Model training extends past the available signal date; refusing a historical prediction.")
    if not set(bundle["features"]).issubset(features.columns):
        raise ModelRetrainingRequired("The saved model uses an older feature schema. Retrain it for the selected target.")
    latest = features.iloc[[-1]][bundle["features"]]
    last_close = float(features["close"].iloc[-1])
    predicted_return = float(bundle["model"].predict(latest)[0])
    record = {
        "ticker": ticker,
        "as_of": latest.index[0].date().isoformat(),
        "last_close": last_close,
        "predicted_return": predicted_return,
        "target_key": target_key,
        "target": MODEL_TARGETS[target_key],
        "model_trained_at_utc": bundle.get("trained_at_utc"),
        "model_trained_through": bundle.get("trained_through"),
    }
    if model_type == "hybrid":
        ridge, boosting = bundle["model"].predict_components(latest)
        record["hybrid"] = {"ridgeReturn": float(ridge[0]), "boostingReturn": float(boosting[0]),
                            "ridgeWeight": bundle["model"].ridge_weight_, "feedbackSamples": 0,
                            "feedbackThrough": None, "learningRule": "settled_error_v1"}
    fingerprint = hashlib.sha256(model_bytes).hexdigest()
    return record, fingerprint, model_path


def _load_latest_prediction(
    ticker: str, model_type: str = "hist_gradient_boosting", target_key: str = "open_to_close"
) -> tuple[dict, str]:
    record, fingerprint, _ = _forecast_latest_close(ticker, model_type, target_key)
    return record, fingerprint


def predict(
    ticker: str, model_type: str = "hist_gradient_boosting", target_key: str = "open_to_close"
) -> None:
    record, _, _ = _forecast_latest_close(ticker, model_type, target_key)
    today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).normalize().tz_localize(None)
    sessions_old = max(len(pd.bdate_range(record["as_of"], today)) - 1, 0)
    if sessions_old >= STALE_FORECAST_SESSIONS:
        print(
            f"Warning: forecast is as of {record['as_of']}, {sessions_old} sessions old. "
            "Run `python -m sp500_ml update` before trusting it."
        )
    result = pd.DataFrame([record])
    prediction_path = _prediction_output_path(model_type, target_key)
    write_csv(result, prediction_path, index=False)
    print(result.to_string(index=False, formatters={
        "last_close": "{:.2f}".format,
        "predicted_return": "{:+.2%}".format,
    }))
    print(f"Saved predictions to {prediction_path.relative_to(ROOT)}")
