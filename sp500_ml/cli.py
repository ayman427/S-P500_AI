"""Command-line entry point: `python -m sp500_ml <command> ...`."""
from __future__ import annotations

import argparse

from .backtest import compare_all_models, compare_models, reassess_saved_models
from .data import list_tickers, update_data
from .fundamentals import update_fundamentals
from .models import MODEL_TYPES
from .news import fetch_news
from .paper import paper_status, paper_trade
from .training import predict, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and model S&P 500 daily prices.")
    parser.add_argument(
        "command",
        choices=[
            "update",
            "fundamentals",
            "news",
            "compare",
            "compare-all",
            "reassess",
            "list",
            "train",
            "predict",
            "run",
            "paper-trade",
            "paper-status",
        ],
    )
    parser.add_argument("--ticker", help="S&P 500 ticker to train or predict, e.g. AAPL")
    parser.add_argument(
        "--model",
        choices=MODEL_TYPES,
        default="hist_gradient_boosting",
        help="Estimator used for next-close training, prediction, and paper trades.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge regularization strength (must be positive).",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.0,
        help="Minimum predicted return at signal close after costs; 0.005 means 0.5%%.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="Estimated one-way transaction cost in basis points (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional ticker limit for compare-all; default compares every local ticker.",
    )
    parser.add_argument(
        "--reveal-holdout",
        action="store_true",
        help=(
            "Print the final holdout metrics after training (hidden by default "
            "so repeated tuning runs can't quietly overfit to it)."
        ),
    )
    args = parser.parse_args()
    if args.transaction_cost_bps < 0:
        parser.error("--transaction-cost-bps must be non-negative")
    if args.ridge_alpha <= 0:
        parser.error("--ridge-alpha must be greater than zero")

    if args.command == "update":
        update_data()
    elif args.command == "list":
        list_tickers()
    elif args.command == "compare-all":
        compare_all_models(
            limit=args.limit or None,
            entry_threshold=args.entry_threshold,
            transaction_cost_bps=args.transaction_cost_bps,
        )
    elif args.command == "reassess":
        reassess_saved_models()
    else:
        if not args.ticker:
            parser.error(f"--ticker is required for the {args.command} command")
        if args.command == "fundamentals":
            update_fundamentals(args.ticker)
        elif args.command == "news":
            fetch_news(args.ticker)
        elif args.command == "compare":
            compare_models(args.ticker, args.entry_threshold, args.transaction_cost_bps)
        elif args.command == "paper-trade":
            paper_trade(
                args.ticker,
                args.entry_threshold,
                args.transaction_cost_bps,
                args.model,
            )
        elif args.command == "paper-status":
            paper_status(args.ticker, args.model)
        elif args.command == "run":
            update_data()
            train_model(
                args.ticker,
                args.entry_threshold,
                args.transaction_cost_bps,
                args.model,
                args.ridge_alpha,
                args.reveal_holdout,
            )
            predict(args.ticker, args.model)
        else:
            if args.command == "train":
                train_model(
                    args.ticker,
                    args.entry_threshold,
                    args.transaction_cost_bps,
                    args.model,
                    args.ridge_alpha,
                    args.reveal_holdout,
                )
            elif args.command == "predict":
                predict(args.ticker, args.model)
