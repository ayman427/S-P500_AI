"""Estimator factory and per-model artifact/prediction file paths."""
from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import _normalize_ticker
from .paths import BACKTEST_DIR, MODEL_DIR, PREDICTIONS_PATH

MODEL_TYPES = ("hist_gradient_boosting", "ridge")


def _new_model(model_type: str = "hist_gradient_boosting", ridge_alpha: float = 10.0):
    if model_type == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=ridge_alpha),
        )
    if model_type != "hist_gradient_boosting":
        raise ValueError(f"Unsupported model type: {model_type}")
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=150,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )


def _model_artifact_paths(ticker: str, model_type: str) -> tuple[Path, Path, Path]:
    ticker = _normalize_ticker(ticker)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model must be one of: {', '.join(MODEL_TYPES)}")
    prefix = ticker if model_type == "hist_gradient_boosting" else f"{ticker}_{model_type}"
    return (
        MODEL_DIR / f"{prefix}_next_close.joblib",
        MODEL_DIR / f"{prefix}_metrics.json",
        BACKTEST_DIR / f"{prefix}_final_holdout_trades.csv",
    )


def _holdout_report_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(
        metrics_path.name.removesuffix("_metrics.json") + "_holdout.json"
    )


def _prediction_output_path(model_type: str) -> Path:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model must be one of: {', '.join(MODEL_TYPES)}")
    if model_type == "hist_gradient_boosting":
        return PREDICTIONS_PATH
    return PREDICTIONS_PATH.with_name(f"predictions_{model_type}.csv")
