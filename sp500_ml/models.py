"""Estimator factory and per-model artifact/prediction file paths."""
from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import resolve_generation
from .data import _normalize_ticker
from .paths import BACKTEST_DIR, MODEL_DIR, PREDICTIONS_PATH

class ModelRetrainingRequired(RuntimeError):
    """A saved estimator predates the requested target or feature schema."""


MODEL_TYPES = ("hist_gradient_boosting", "ridge", "hybrid")
# Support multiple targets
MODEL_TARGETS = {
    "open_to_close": "next-session open-to-close return",
    "open_to_close_5d": "5-session forward open-to-close return",
}
MODEL_TARGET_COLUMNS = {
    "open_to_close": "next_open_to_close_return",
    "open_to_close_5d": "next_5d_open_to_close_return",
}
# Default target for backward compatibility
MODEL_TARGET = MODEL_TARGETS["open_to_close"]
MODEL_TARGET_COLUMN = MODEL_TARGET_COLUMNS["open_to_close"]


def _new_model(model_type: str = "hist_gradient_boosting", ridge_alpha: float = 10.0):
    if model_type == "hybrid":
        from .hybrid import HybridRegressor
        return HybridRegressor(ridge_alpha=ridge_alpha)
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


def _artifact_prefix(ticker: str, model_type: str, target_key: str = "open_to_close") -> str:
    ticker = _normalize_ticker(ticker)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model must be one of: {', '.join(MODEL_TYPES)}")
    if target_key not in MODEL_TARGETS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGETS.keys())}")
    model_prefix = ticker if model_type == "hist_gradient_boosting" else f"{ticker}_{model_type}"
    target_suffix = "_open_to_close" if target_key == "open_to_close" else "_open_to_close_5d"
    return f"{model_prefix}{target_suffix}"


def _model_artifact_paths(ticker: str, model_type: str, target_key: str = "open_to_close") -> tuple[Path, Path, Path]:
    prefix = _artifact_prefix(ticker, model_type, target_key)
    generation = resolve_generation(MODEL_DIR, prefix)
    if generation is not None:
        return generation
    return (
        MODEL_DIR / f"{prefix}_next_close.joblib",
        MODEL_DIR / f"{prefix}_metrics.json",
        BACKTEST_DIR / f"{prefix}_final_holdout_trades.csv",
    )


def _holdout_report_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(
        metrics_path.name.removesuffix("_metrics.json") + "_holdout.json"
    )


def _prediction_output_path(model_type: str, target_key: str = "open_to_close") -> Path:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model must be one of: {', '.join(MODEL_TYPES)}")
    if target_key not in MODEL_TARGETS:
        raise ValueError(f"Target must be one of: {', '.join(MODEL_TARGETS.keys())}")
    if target_key == "open_to_close":
        if model_type == "hist_gradient_boosting":
            return PREDICTIONS_PATH
        return PREDICTIONS_PATH.with_name(f"predictions_{model_type}.csv")
    return PREDICTIONS_PATH.with_name(f"predictions_{model_type}_{target_key}.csv")
