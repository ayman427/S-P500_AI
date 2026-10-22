from __future__ import annotations

import json
import hashlib
from io import BytesIO
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sp500_ml.cloud_data import read_csv
import yfinance as yf
import joblib

import sp500_ml
from sp500_ml.paper import _validate_ledger_identity
from sp500_ml.models import ModelRetrainingRequired
from sp500_ml.auth import authenticate, AuthenticationError


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


def _model_inspection_payload(
    bundle: dict,
    feature_row: pd.DataFrame,
    predicted_return: float,
) -> dict:
    feature_names = bundle["features"]
    contributions: dict[str, float] = {}
    intercept = None
    model = bundle["model"]
    model_type = bundle.get("model_type", "hist_gradient_boosting")
    model_input = feature_row[feature_names]

    if model_type == "ridge" and hasattr(model, "named_steps"):
        imputed = model.named_steps["simpleimputer"].transform(model_input)
        scaled = model.named_steps["standardscaler"].transform(imputed)[0]
        estimator = model.named_steps["ridge"]
        coefficients = np.asarray(estimator.coef_).reshape(-1)
        contributions = {
            name: float(value)
            for name, value in zip(feature_names, scaled * coefficients)
        }
        intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])

    entry_threshold = bundle.get("entry_threshold", 0.0)
    transaction_cost_bps = bundle.get("transaction_cost_bps_per_side", 10.0)
    round_trip_cost = 2 * transaction_cost_bps / 10_000
    signal_edge_after_costs = predicted_return - round_trip_cost
    return {
        "modelType": model_type,
        "modelLabel": (
            "Ridge regression" if model_type == "ridge" else "Hybrid (experimental)" if model_type == "hybrid" else "HistGradientBoostingRegressor"
        ),
        "target": bundle.get("target", sp500_ml.MODEL_TARGET),
        "asOf": feature_row.index[-1].date().isoformat(),
        "lastClose": float(feature_row["close"].iloc[-1]) if "close" in feature_row else None,
        "predictedReturn": float(predicted_return),
        "ridgeAlpha": bundle.get("best_params", {}).get(
            "ridge__alpha", bundle.get("ridge_alpha")
        ),
        "bestParams": bundle.get("best_params", {}),
        "hybridCalibration": bundle.get("hybrid_calibration") if model_type == "hybrid" else None,
        "trainingRows": bundle.get("training_rows"),
        "trainedThrough": bundle.get("trained_through"),
        "featureCount": len(feature_names),
        "features": [
            {
                "name": name,
                "value": _json_value(model_input[name].iloc[0]),
                "contribution": contributions.get(name),
            }
            for name in feature_names
        ],
        "explanationType": (
            "additive_standardized_coefficients"
            if contributions
            else "feature_values_only"
        ),
        "intercept": intercept,
        "contributionSum": (
            float(intercept + sum(contributions.values()))
            if contributions
            else None
        ),
        "entryThreshold": entry_threshold,
        "transactionCostBpsPerSide": transaction_cost_bps,
        "signalEdgeAfterCosts": signal_edge_after_costs,
        "enterLong": signal_edge_after_costs > entry_threshold,
    }


def _read_model_inspection(ticker: str, model_type: str, predicted_return: float, target_key: str = "open_to_close") -> dict:
    ticker = sp500_ml._normalize_ticker(ticker)
    model_path, _, _ = sp500_ml._model_artifact_paths(ticker, model_type, target_key)
    bundle = joblib.load(BytesIO(model_path.read_bytes()))
    price_path = sp500_ml._ticker_file(ticker)
    prices = sp500_ml._drop_incomplete_session(
        read_csv(price_path, parse_dates=["Date"], index_col="Date").sort_index()
    )
    uses_fundamentals = any(
        name in sp500_ml.FUNDAMENTAL_FEATURES for name in bundle["features"]
    )
    fundamentals = (
        sp500_ml._load_fundamentals(ticker, prices.index)
        if uses_fundamentals
        else None
    )
    features = sp500_ml.build_features(
        prices, sp500_ml._load_market_data(), fundamentals
    )
    if features.empty:
        raise RuntimeError(f"Not enough completed sessions to inspect {ticker}.")
    latest = features.iloc[[-1]][["close", *bundle["features"]]]
    return _model_inspection_payload(bundle, latest, predicted_return)


def _read_stock_directory() -> dict:
    """Resolve stocks using price inventory and company metadata stored in Convex."""
    names = {}
    sources = [
        sp500_ml.FUNDAMENTAL_DIR / "company_tickers.json",
        sp500_ml.DATA_DIR.parent / "companies.json",
    ]
    for path in sources:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "company_tickers.json":
                for item in payload.values():
                    symbol = str(item.get("ticker", "")).upper().replace(".", "-")
                    if symbol and item.get("title"):
                        names[symbol] = str(item["title"])
            else:
                names.update({str(symbol).upper().replace(".", "-"): str(name) for symbol, name in payload.items()})
        except (OSError, ValueError, TypeError, AttributeError):
            continue  # Symbol search still works if optional metadata is unavailable.
    return {"stocks": [
        {"ticker": path.stem, "name": names.get(path.stem, path.stem)}
        for path in sp500_ml._price_files()
    ]}


def _read_dashboard_state(
    ticker: str, model_type: str = "hist_gradient_boosting", target_key: str = "open_to_close",
    reveal_holdout: bool = False, include_paper: bool = True, token: str | None = None,
) -> dict:
    ticker = sp500_ml._normalize_ticker(ticker)
    price_path = sp500_ml._ticker_file(ticker)
    if not price_path.exists():
        raise FileNotFoundError(f"No downloaded price history for {ticker}.")

    prices = read_csv(price_path, parse_dates=["Date"], index_col="Date").sort_index()
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
        ticker, model_type, target_key
    )
    forecast = None
    forecast_error = None
    model_inspection = None
    model_inspection_error = None
    forecast_needs_retraining = False
    if model_path.exists():
        try:
            forecast, fingerprint = sp500_ml._load_latest_prediction(ticker, model_type, target_key)
            if model_type == "hybrid" and token:
                from sp500_ml.prediction_store import apply_hybrid_feedback
                forecast = apply_hybrid_feedback(forecast, fingerprint, token)
        except ModelRetrainingRequired as error:
            forecast = None
            forecast_error = str(error)
            forecast_needs_retraining = True
        except Exception as error:
            forecast = None
            forecast_error = str(error)
        if forecast is not None:
            try:
                model_inspection = _read_model_inspection(
                    ticker, model_type, forecast["predicted_return"], target_key
                )
                if model_type == "hybrid":
                    model_inspection["hybridLearning"] = forecast.get("hybrid")
            except Exception as error:
                model_inspection_error = str(error)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    comparison_name = (
        f"{ticker}_intraday_comparison.json"
        if target_key == "open_to_close"
        else f"{ticker}_{target_key}_intraday_comparison.json"
    )
    comparison_path = sp500_ml.MODEL_DIR / comparison_name
    if comparison_path.exists():
        comparison_report = json.loads(comparison_path.read_text(encoding="utf-8"))
        intraday_comparison = comparison_report.get("results")
    else:
        intraday_comparison = (metrics or {}).get("development_intraday_model_comparison")

    paper_path = (
        sp500_ml._paper_ledger_path(ticker, model_type=model_type, schema_version=3)
        if include_paper and model_path.exists() and target_key == "open_to_close" else None
    )
    paper_error = None
    paper_notice = None
    if target_key != "open_to_close":
        paper_notice = "Verified paper trading supports the one-session target only."
    elif paper_path is None or not paper_path.exists():
        paper_notice = "No verified paper signals for this model yet. Earlier ledgers are preserved as historical files and are not included in these totals."
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
    ledger = None
    if paper_path is not None and paper_path.exists():
        try:
            ledger = read_csv(paper_path)
            fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()
            _validate_ledger_identity(ledger, ticker, model_type, fingerprint)
            paper_summary = sp500_ml._paper_summary(ledger)
        except (OSError, ValueError, KeyError) as error:
            paper_error = str(error)
            ledger = None
    if ledger is not None:
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
    current_target_model = (
        (metrics or {}).get("target_key", "open_to_close") == target_key
        and (metrics or {}).get("target") == sp500_ml.MODEL_TARGETS[target_key]
    )
    final_holdout = (metrics or {}).get("final_holdout") if current_target_model and reveal_holdout else None
    if current_target_model and reveal_holdout and not isinstance(final_holdout, dict):
        holdout_path = sp500_ml.models._holdout_report_path(metrics_path)
        final_holdout = (
            json.loads(holdout_path.read_text(encoding="utf-8"))
            if holdout_path.exists()
            else {}
        )
    if metrics and (
        metrics.get("ticker", ticker) != ticker
        or metrics.get("model_type", model_type) != model_type
        or metrics.get("target_key", target_key) != target_key
        or metrics.get("target", sp500_ml.MODEL_TARGETS[target_key]) != sp500_ml.MODEL_TARGETS[target_key]
    ):
        raise RuntimeError("Saved metrics do not match the selected ticker, model, and target; retrain this selection.")
    if sp500_ml._model_artifact_paths(ticker, model_type, target_key)[0] != model_path:
        raise RuntimeError("The selected model changed while loading results; refresh the dashboard.")
    return {
        "ticker": ticker,
        "modelType": model_type,
        "artifactIntegrity": "verified" if (metrics or {}).get("run_id") else "legacy_unverified",
        "paperError": paper_error,
        "paperNotice": paper_notice,
        "targetKey": target_key,
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
        "forecastNeedsRetraining": forecast_needs_retraining,
        "modelInspection": model_inspection,
        "modelInspectionError": model_inspection_error,
        "modelReady": bool(model_path.exists() and current_target_model and forecast is not None),
        "metrics": {key: value for key, value in metrics.items() if key != "final_holdout"} if metrics else None,
        "intradayComparison": intraday_comparison,
        "holdout": final_holdout,
        "paper": paper_summary if target_key == "open_to_close" else None,
        "paperRows": paper_rows if target_key == "open_to_close" else [],
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

    def _require_user(self):
        try:
            self.auth_user = authenticate(self.headers.get("Authorization"))
            return True
        except AuthenticationError as error:
            self._send_json(error.status, {"error": str(error)})
            return False

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._require_user():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/auth/me":
            self._send_json(200, {"userId": self.auth_user.user_id})
            return
        if parsed.path == "/api/market-watch":
            try:
                from sp500_ml.market_watch import read_scan
                self._send_json(200, read_scan())
            except Exception:
                self._send_json(502, {"error": "The S&P 500 scan is temporarily unavailable."})
            return
        if parsed.path == "/api/stocks":
            try:
                self._send_json(200, _read_stock_directory())
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if parsed.path == "/api/predictions":
            from sp500_ml.prediction_store import my_predictions
            try:
                cursor = parse_qs(parsed.query).get("cursor", [None])[0]
                self._send_json(200, my_predictions(self.auth_user.token, cursor))
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            except Exception:
                self._send_json(502, {"error": "Your prediction history is temporarily unavailable. Please try again."})
            return
        if parsed.path == "/api/quote":
            ticker = parse_qs(parsed.query).get("ticker", ["AMD"])[0]
            try:
                self._send_json(200, _read_live_quote(ticker))
            except Exception as error:
                self._send_json(502, {"error": str(error)})
            return
        if parsed.path in {"/api/spikes", "/api/risk", "/api/risk/news", "/api/risk/analysis"}:
            query = parse_qs(parsed.query)
            try:
                ticker = query.get("ticker", ["AMD"])[0]
                if parsed.path == "/api/spikes":
                    from sp500_ml.spike_anomalies import read_anomalies
                    payload = read_anomalies(ticker, int(query.get("years", ["5"])[0]))
                elif parsed.path == "/api/risk":
                    from sp500_ml.risk import read_risk
                    payload = read_risk(ticker, int(query.get("years", ["5"])[0]), float(query.get("threshold", ["0.2"])[0]))
                elif parsed.path == "/api/risk/news":
                    from sp500_ml.risk_news import read_event_news
                    payload = read_event_news(ticker, query.get("date", [""])[0], int(query.get("window", ["0"])[0]))
                else:
                    from sp500_ml.news_analysis import analyze_event_news
                    payload = analyze_event_news(ticker, query.get("date", [""])[0], int(query.get("window", ["0"])[0]))
                self._send_json(200, payload)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            except Exception:
                self._send_json(502, {"error": "Historical risk data is temporarily unavailable. Please try again."})
            return
        if parsed.path != "/api/state":
            self._send_json(404, {"error": "Not found"})
            return
        query = parse_qs(parsed.query)
        ticker = query.get("ticker", ["AMD"])[0]
        model_type = query.get("model_type", ["hist_gradient_boosting"])[0]
        target_key = query.get("target_key", ["open_to_close"])[0]
        try:
            if model_type not in sp500_ml.MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_type}")
            if target_key not in sp500_ml.MODEL_TARGETS:
                raise ValueError(f"Unsupported target: {target_key}")
            options = {"token": self.auth_user.token} if model_type == "hybrid" else {}
            self._send_json(200, _read_dashboard_state(ticker, model_type, target_key, False, include_paper=False, **options))
        except Exception as error:
            self._send_json(400, {"error": str(error)})

    def do_POST(self) -> None:
        if not self._require_user():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/market-watch/scan":
            try:
                from sp500_ml.market_watch import scan_batch
                self._send_json(200, scan_batch(restart=parse_qs(parsed.query).get("restart") == ["true"]))
            except Exception:
                self._send_json(502, {"error": "The S&P 500 scan could not advance. Please retry."})
            return
        if parsed.path == "/api/spikes/watch":
            try:
                from sp500_ml.spike_watch import run_watch
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict) or not isinstance(body.get("ticker"), str):
                    raise ValueError("A stock ticker is required.")
                self._send_json(200, run_watch(body["ticker"]))
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            except Exception:
                self._send_json(502, {"error": "Current watch could not be refreshed and saved. No current warning was confirmed."})
            return
        if parsed.path in {"/api/spikes/research", "/api/spikes/gemini"}:
            from sp500_ml.gemini_research import analyze_saved_evidence, GeminiResearchError
            try:
                from sp500_ml.spike_research import read_investigation
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict) or not isinstance(body.get("ticker"), str):
                    raise ValueError("A stock ticker and detected spike date are required.")
                research = analyze_saved_evidence if parsed.path.endswith("/gemini") else read_investigation
                self._send_json(200, research(body["ticker"], body.get("date")))
            except GeminiResearchError as error:
                self._send_json(error.status, {"error": str(error)})
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            except Exception:
                self._send_json(502, {"error": "Spike research is unavailable; no new save was confirmed."})
            return
        if parsed.path == "/api/prices/refresh":
            try:
                from sp500_ml.search_prices import refresh_search_prices
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                self._send_json(200, refresh_search_prices(body.get("ticker", "AMD")))
            except Exception as error:
                self._send_json(502, {"error": f"Could not refresh market data. {error}"})
            return
        if parsed.path == "/api/predictions":
            try:
                from sp500_ml.prediction_store import track
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                ticker = sp500_ml._normalize_ticker(body.get("ticker", "AMD"))
                model = body.get("model_type", "hist_gradient_boosting")
                target = body.get("target_key", "open_to_close")
                if model not in sp500_ml.MODEL_TYPES or target not in sp500_ml.MODEL_TARGETS:
                    raise ValueError("Unsupported model or period")
                self._send_json(200, track(ticker, model, target, token=self.auth_user.token))
            except Exception:
                self._send_json(502, {"error": "Live prediction storage is unavailable. Forecasts remain visible; no save was confirmed."})
            return
        action = parsed.path.removeprefix("/api/action/")
        if action in {"update", "paper-trade", "compare"}:
            self._send_json(403, {"error": "This maintenance action is available only through the trusted server CLI."})
            return
        if action not in {"update", "train", "paper-trade", "news", "compare"}:
            self._send_json(404, {"error": "Unknown action"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            ticker = sp500_ml._normalize_ticker(body.get("ticker", "AMD"))
            model_type = body.get("model_type", "hist_gradient_boosting")
            target_key = body.get("target_key", "open_to_close")
            ridge_alpha = float(body.get("ridge_alpha", 10.0))
            if model_type not in sp500_ml.MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_type}")
            if target_key not in sp500_ml.MODEL_TARGETS:
                raise ValueError(f"Unsupported target: {target_key}")
            if action == "paper-trade" and target_key != "open_to_close":
                raise ValueError("paper-trade currently supports only the one-session target.")
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
                if action in {"train", "compare"}:
                    command.extend(["--target", target_key])
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
    import threading
    from sp500_ml.prediction_store import configured, worker
    try:
        server = DashboardServer(("127.0.0.1", PORT), DashboardHandler)
    except OSError as error:
        if getattr(error, "winerror", None) == 10048 or error.errno == 98:
            print(f"Port {PORT} is already in use. If this is the dashboard API, keep using the running server. Stop its process before starting another instance.")
            return
        raise
    if configured():
        threading.Thread(target=worker, daemon=True, name="prediction-comparison").start()
        from sp500_ml.market_watch import worker as market_worker
        threading.Thread(target=market_worker, daemon=True, name="market-checklist").start()
        from sp500_ml.anomaly_tracking import worker as anomaly_worker
        threading.Thread(target=anomaly_worker, daemon=True, name="rise-prospective").start()
    print(f"Dashboard API listening on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
