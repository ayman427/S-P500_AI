"""S&P 500 daily-price download, feature engineering, modeling, and paper trading.

Split into focused submodules for maintainability:
    data          price download/storage, adjustment-drift detection
    fundamentals  SEC point-in-time fundamentals
    feature_names shared feature-name lists
    features      point-in-time feature engineering
    models        estimator factory and per-model artifact paths
    backtest      walk-forward validation, holdout, model comparisons
    training      train/predict orchestration
    news          headline fetch and informational (non-model) entity extraction
    paper         paper-trading ledgers
    cli           `python -m sp500_ml <command> ...` entry point

Public names are re-exported here for backward compatibility with callers
that do ``import sp500_ml`` / ``from sp500_ml import ...``.
"""
from __future__ import annotations

import pandas as pd

from .backtest import (
    FINAL_HOLDOUT_SESSIONS,
    _final_holdout_metrics,
    _metrics_for_console,
    _new_intraday_ridge_model,
    _summarize_universe_comparisons,
    _trade_metrics,
    _trade_results,
    _walk_forward_intraday_comparison,
    _walk_forward_metrics,
    _wilson_interval,
    compare_all_models,
    compare_models,
    reassess_saved_models,
)
from .data import (
    BATCH_SIZE,
    PRICE_COLUMN_ORDER,
    PRICE_COLUMNS,
    START_DATE,
    _atomic_write_prices,
    _download_batch,
    _drop_incomplete_session,
    _extract_ticker,
    _full_refresh_due,
    _merge_price_history,
    _normalize_ticker,
    _price_files,
    _ticker_file,
    get_sp500_tickers,
    list_tickers,
    update_data,
)
from .feature_names import FEATURES, FUNDAMENTAL_FEATURES, PRICE_FEATURES
from .features import _available_features, _load_market_data, build_features
from .fundamentals import (
    FUNDAMENTAL_CONCEPTS,
    _align_fundamental_events,
    _fundamental_events_from_companyfacts,
    _fundamental_file,
    _get_sec_user_agent,
    _load_fundamentals,
    _load_training_fundamentals,
    _read_fundamentals,
    _read_windows_sec_user_agent,
    update_fundamentals,
)
from .models import MODEL_TYPES, _model_artifact_paths, _new_model, _prediction_output_path
from .news import (
    _enrich_news_relationships,
    _extract_headline_relationships,
    _fetch_recent_news,
    _parse_wikipedia_entity_context,
    _parse_yahoo_news,
    _save_news_cache,
    _search_entity_context,
    fetch_news,
)
from .paper import _paper_ledger_path, _paper_summary, _settle_paper_rows, paper_status, paper_trade
from .paths import (
    BACKTEST_DIR,
    DATA_DIR,
    FUNDAMENTAL_DIR,
    MODEL_DIR,
    NEWS_DIR,
    PAPER_DIR,
    PREDICTIONS_PATH,
    ROOT,
)
from .training import _forecast_latest_close, _load_latest_prediction, predict, train_model

__all__ = [
    "ROOT",
    "DATA_DIR",
    "FUNDAMENTAL_DIR",
    "NEWS_DIR",
    "MODEL_DIR",
    "BACKTEST_DIR",
    "PAPER_DIR",
    "PREDICTIONS_PATH",
    "START_DATE",
    "BATCH_SIZE",
    "FINAL_HOLDOUT_SESSIONS",
    "MODEL_TYPES",
    "FEATURES",
    "PRICE_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "PRICE_COLUMN_ORDER",
    "PRICE_COLUMNS",
    "FUNDAMENTAL_CONCEPTS",
    "get_sp500_tickers",
    "list_tickers",
    "update_data",
    "update_fundamentals",
    "fetch_news",
    "build_features",
    "compare_models",
    "compare_all_models",
    "reassess_saved_models",
    "train_model",
    "predict",
    "paper_trade",
    "paper_status",
]
