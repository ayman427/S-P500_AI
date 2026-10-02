from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import yfinance as yf

import sp500_ml


ROOT = Path(__file__).resolve().parent
PORT = 8765


def _project_python() -> str:
    candidates = (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_dashboard_state(
    ticker: str, model_type: str = "hist_gradient_boosting"
) -> dict:
    ticker = sp500_ml._normalize_ticker(ticker)
    price_path = sp500_ml._ticker_file(ticker)
    if not price_path.exists():
        raise FileNotFoundError(f"No downloaded price history for {ticker}.")

    prices = pd.read_csv(price_path, parse_dates=["Date"], index_col="Date").sort_index()
    closes = prices["Close"].astype(float)
    latest = prices.iloc[-1]
    previous_close = float(closes.iloc[-2]) if len(closes) > 1 else None
    latest_close = float(latest["Close"])
    daily_change = (
        latest_close / previous_close - 1 if previous_close not in (None, 0) else None
    )
    history = [
        {"date": date.date().isoformat(), "close": float(row["Close"])}
        for date, row in prices.tail(120).iterrows()
    ]

    model_path, metrics_path, _ = sp500_ml._model_artifact_paths(
        ticker, model_type
    )
    forecast = None
    forecast_error = None
    if model_path.exists():
        try:
            forecast, _ = sp500_ml._load_latest_prediction(ticker, model_type)
        except Exception as error:  # Surface missing local features without breaking the dashboard.
            forecast_error = str(error)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    comparison_path = sp500_ml.MODEL_DIR / f"{ticker}_intraday_comparison.json"
    if comparison_path.exists():
        comparison_report = json.loads(comparison_path.read_text(encoding="utf-8"))
        intraday_comparison = comparison_report.get("results")
    else:
        intraday_comparison = (metrics or {}).get("development_intraday_model_comparison")

    paper_path = sp500_ml._paper_ledger_path(
        ticker, model_type=model_type, schema_version=2
    )
    if not paper_path.exists():
        paper_path = sp500_ml._paper_ledger_path(ticker, model_type=model_type)
    paper_summary = None
    paper_rows = []
    headlines = []
    news_path = sp500_ml.NEWS_DIR / f"{ticker}.json"
    news_fetched_at = None
    if news_path.exists():
        try:
            news_cache = json.loads(news_path.read_text(encoding="utf-8"))
            headlines = news_cache.get("headlines", [])
            news_fetched_at = news_cache.get("fetched_at_utc")
        except (OSError, json.JSONDecodeError):
            headlines = []
    if paper_path.exists():
        ledger = pd.read_csv(paper_path)
        paper_summary = sp500_ml._paper_summary(ledger)
        for _, row in ledger.tail(12).iloc[::-1].iterrows():
            paper_rows.append({key: _json_value(value) for key, value in row.items() if key != "news_context_json"})
        if not headlines:
            for raw_context in ledger.get("news_context_json", pd.Series(dtype=str)).dropna().iloc[::-1]:
                try:
                    headlines = json.loads(raw_context)
                except (TypeError, json.JSONDecodeError):
                    continue
                if headlines:
                    break

    tickers = [path.stem for path in sp500_ml._price_files() if path.stem != "SPY"]
    final_holdout = (metrics or {}).get("final_holdout")
    if not isinstance(final_holdout, dict):
        holdout_path = sp500_ml.models._holdout_report_path(metrics_path)
        final_holdout = (
            json.loads(holdout_path.read_text(encoding="utf-8"))
            if holdout_path.exists()
            else {}
        )
    return {
        "ticker": ticker,
        "modelType": model_type,
        "ridgeAlpha": (metrics or {}).get("ridge_alpha", 10.0),
        "tickers": tickers,
        "quote": {
            "date": prices.index[-1].date().isoformat(),
            "close": latest_close,
            "previousClose": previous_close,
            "dailyChange": daily_change,
            "volume": int(latest["Volume"]),
        },
        "history": history,
        "forecast": forecast,
        "forecastError": forecast_error,
        "modelReady": model_path.exists(),
        "metrics": metrics,
        "intradayComparison": intraday_comparison,
        "holdout": final_holdout,
        "paper": paper_summary,
        "paperRows": paper_rows,
        "headlines": headlines,
        "newsFetchedAt": news_fetched_at,
    }


def _read_live_quote(ticker: str) -> dict:
    """Fetch a live quote from Yahoo Finance; never written to the stored price history."""
    ticker = sp500_ml._normalize_ticker(ticker)
    info = yf.Ticker(ticker).fast_info

    def field(name: str):
        try:
            return info[name]
        except (KeyError, TypeError):
            return None

    price = field("last_price")
    previous_close = field("previous_close")
    if price is None or not np.isfinite(price):
        raise RuntimeError(f"No live quote available for {ticker}.")
    previous_close = (
        float(previous_close)
        if previous_close is not None and np.isfinite(previous_close)
        else None
    )
    volume = field("last_volume")
    return {
        "ticker": ticker,
        "price": float(price),
        "previousClose": previous_close,
        "dailyChange": (
            float(price) / previous_close - 1 if previous_close else None
        ),
        "volume": int(volume) if volume is not None and np.isfinite(volume) else None,
        "fetchedAt": pd.Timestamp.now(tz="UTC").isoformat(),
    }


class DashboardServer(ThreadingHTTPServer):
    # Windows lets a second server share the port when reuse is on, silently splitting traffic.
    allow_reuse_address = False
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # A browser aborting a polled request is routine, not an error worth a traceback.
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SP500Dashboard/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/quote":
            ticker = parse_qs(parsed.query).get("ticker", ["AMD"])[0]
            try:
                self._send_json(200, _read_live_quote(ticker))
            except Exception as error:
                self._send_json(502, {"error": str(error)})
            return
        if parsed.path != "/api/state":
            self._send_json(404, {"error": "Not found"})
            return
        query = parse_qs(parsed.query)
        ticker = query.get("ticker", ["AMD"])[0]
        model_type = query.get("model_type", ["hist_gradient_boosting"])[0]
        try:
            if model_type not in sp500_ml.MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_type}")
            self._send_json(200, _read_dashboard_state(ticker, model_type))
        except Exception as error:
            self._send_json(400, {"error": str(error)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        action = parsed.path.removeprefix("/api/action/")
        if action not in {"update", "train", "paper-trade", "news", "compare"}:
            self._send_json(404, {"error": "Unknown action"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            ticker = sp500_ml._normalize_ticker(body.get("ticker", "AMD"))
            model_type = body.get("model_type", "hist_gradient_boosting")
            ridge_alpha = float(body.get("ridge_alpha", 10.0))
            if model_type not in sp500_ml.MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_type}")
            if ridge_alpha <= 0:
                raise ValueError("Ridge alpha must be greater than zero.")
            command = [_project_python(), "-m", "sp500_ml"]
            if action == "update":
                command.append("update")
            else:
                command.extend([action, "--ticker", ticker])
                if action in {"train", "paper-trade"}:
                    command.extend(["--model", model_type])
                if action == "train":
                    command.extend(["--ridge-alpha", str(ridge_alpha)])
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            output = result.stdout.strip()
            if result.returncode:
                self._send_json(400, {"error": result.stderr.strip() or output})
                return
            self._send_json(200, {"ok": True, "output": output})
        except subprocess.TimeoutExpired:
            self._send_json(504, {"error": "Action exceeded the 15-minute limit."})
        except Exception as error:
            self._send_json(400, {"error": str(error)})

    def log_message(self, format: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} {format % args}")


def main() -> None:
    server = DashboardServer(("127.0.0.1", PORT), DashboardHandler)
    print(f"Dashboard API listening on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()