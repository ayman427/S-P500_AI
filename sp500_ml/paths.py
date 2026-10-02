"""Filesystem locations shared across the sp500_ml package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "prices"
FUNDAMENTAL_DIR = ROOT / "data" / "fundamentals"
NEWS_DIR = ROOT / "data" / "news"
MODEL_DIR = ROOT / "models"
BACKTEST_DIR = ROOT / "backtests"
PAPER_DIR = ROOT / "paper_trades"
PREDICTIONS_PATH = ROOT / "predictions.csv"
