"""Filesystem locations shared across the sp500_ml package."""
from __future__ import annotations

from pathlib import Path
from .cloud_data import RemotePath

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = RemotePath("prices")
FUNDAMENTAL_DIR = RemotePath("fundamentals")
NEWS_DIR = RemotePath("news")
MODEL_DIR = RemotePath("models")
BACKTEST_DIR = RemotePath("backtests")
PAPER_DIR = RemotePath("paper_trades")
PREDICTIONS_PATH = RemotePath("prediction_exports/predictions.csv")
