"""ASGI entrypoint for the Vercel Python service."""
from __future__ import annotations

import subprocess

from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel

import dashboard_server
import sp500_ml
from sp500_ml.auth import authenticate, AuthenticationError

def require_user(request: Request):
    try:
        request.state.auth_user = authenticate(request.headers.get("Authorization"))
    except AuthenticationError as error:
        raise HTTPException(status_code=error.status, detail=str(error), headers={"WWW-Authenticate": "Bearer"}) from error


app = FastAPI(title="S&P 500 Research API", dependencies=[Depends(require_user)], docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/api/auth/me")
def current_user(request: Request):
    return {"userId": request.state.auth_user.user_id}


@app.get("/api/market-watch")
def market_watch():
    from sp500_ml.market_watch import read_scan
    try:
        return read_scan()
    except Exception as error:
        raise HTTPException(status_code=502, detail="The S&P 500 scan is temporarily unavailable.") from error


@app.post("/api/market-watch/scan")
def advance_market_watch(restart: bool = False):
    from sp500_ml.market_watch import scan_batch
    try:
        return scan_batch(restart=restart)
    except Exception as error:
        raise HTTPException(status_code=502, detail="The S&P 500 scan could not advance. Please retry.") from error


class ActionRequest(BaseModel):
    ticker: str = "AMD"
    model_type: str = "hist_gradient_boosting"
    target_key: str = "open_to_close"
    ridge_alpha: float = 10.0


@app.get("/api/stocks")
def read_stocks() -> dict:
    return dashboard_server._read_stock_directory()


@app.post("/api/predictions")
def track_prediction(body: ActionRequest, request: Request) -> dict:
    from sp500_ml.prediction_store import track
    if body.model_type not in sp500_ml.MODEL_TYPES or body.target_key not in sp500_ml.MODEL_TARGETS:
        raise HTTPException(status_code=400, detail="Unsupported model or period")
    try:
        return track(sp500_ml._normalize_ticker(body.ticker), body.model_type, body.target_key, token=request.state.auth_user.token)
    except Exception as error:
        raise HTTPException(status_code=502, detail="Live prediction storage unavailable; no save was confirmed.") from error


@app.get("/api/predictions")
def my_predictions(request: Request, cursor: str | None = None) -> dict:
    from sp500_ml.prediction_store import my_predictions as history
    try:
        return history(request.state.auth_user.token, cursor)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Your prediction history is temporarily unavailable. Please try again.") from error


@app.post("/api/prices/refresh")
def refresh_prices(body: ActionRequest) -> dict:
    from sp500_ml.search_prices import refresh_search_prices
    try:
        return refresh_search_prices(body.ticker)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Could not refresh market data. {error}") from error


@app.get("/api/state")
def read_state(
    request: Request,
    ticker: str = "AMD",
    model_type: str = "hist_gradient_boosting",
    target_key: str = "open_to_close",
    reveal_holdout: bool = False,
) -> dict:
    if model_type not in sp500_ml.MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
    if target_key not in sp500_ml.MODEL_TARGETS:
        raise HTTPException(status_code=400, detail=f"Unsupported target: {target_key}")
    try:
        options = {"token": request.state.auth_user.token} if model_type == "hybrid" else {}
        return dashboard_server._read_dashboard_state(ticker, model_type, target_key, False, include_paper=False, **options)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/quote")
def read_quote(ticker: str = "AMD") -> dict:
    try:
        return dashboard_server._read_live_quote(ticker)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


class SpikeResearchRequest(BaseModel):
    ticker: str
    date: str


@app.post("/api/spikes/gemini")
def gemini_research(body: SpikeResearchRequest):
    from fastapi.responses import JSONResponse
    from sp500_ml.gemini_research import analyze_saved_evidence, GeminiResearchError
    try:
        return JSONResponse(analyze_saved_evidence(body.ticker, body.date), headers={"Cache-Control": "no-store"})
    except GeminiResearchError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Gemini research is temporarily unavailable.") from error


class SpikeWatchRequest(BaseModel):
    ticker: str


@app.post("/api/spikes/watch")
def run_spike_watch(body: SpikeWatchRequest) -> dict:
    from sp500_ml.spike_watch import run_watch
    try:
        return run_watch(body.ticker)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Current watch could not be refreshed and saved. No current warning was confirmed.") from error


@app.post("/api/spikes/research")
def research_spike(body: SpikeResearchRequest) -> dict:
    from sp500_ml.spike_research import read_investigation as research
    try:
        return research(body.ticker, body.date)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Spike research is unavailable; no new save was confirmed.") from error


@app.get("/api/spikes")
def read_spikes(ticker: str, years: int = 5) -> dict:
    from sp500_ml.spike_anomalies import read_anomalies as evaluate
    try:
        return evaluate(ticker, years)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Price spike data is temporarily unavailable. Please try again.") from error


@app.get("/api/risk")
def read_risk(ticker: str, years: int = 5, threshold: float = 0.20) -> dict:
    from sp500_ml.risk import read_risk as evaluate
    try:
        return evaluate(ticker, years, threshold)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Historical risk data is temporarily unavailable. Please try again.") from error


@app.get("/api/risk/news")
def read_risk_news(ticker: str, date: str, window: int = 0) -> dict:
    from sp500_ml.risk_news import read_event_news
    try:
        return read_event_news(ticker, date, window)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Historical news is temporarily unavailable. Please try again.") from error


@app.get("/api/risk/analysis")
def read_risk_analysis(ticker: str, date: str, window: int = 0) -> dict:
    from sp500_ml.news_analysis import analyze_event_news
    try:
        return analyze_event_news(ticker, date, window)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="News analysis is temporarily unavailable. Please try again.") from error


@app.post("/api/action/{action}")
def run_action(action: str, body: ActionRequest) -> dict:
    if action in {"update", "paper-trade", "compare"}:
        raise HTTPException(status_code=403, detail="This maintenance action is available only through the trusted server CLI.")
    allowed_actions = {"update", "train", "paper-trade", "news", "compare"}
    if action not in allowed_actions:
        raise HTTPException(status_code=404, detail="Unknown action")
    try:
        ticker = sp500_ml._normalize_ticker(body.ticker)
        if body.model_type not in sp500_ml.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {body.model_type}")
        if body.target_key not in sp500_ml.MODEL_TARGETS:
            raise ValueError(f"Unsupported target: {body.target_key}")
        if action == "paper-trade" and body.target_key != "open_to_close":
            raise ValueError("paper-trade currently supports only the one-session target.")
        if body.ridge_alpha <= 0:
            raise ValueError("Ridge alpha must be greater than zero.")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    command = [dashboard_server._project_python(), "-m", "sp500_ml"]
    if action == "update":
        command.append(action)
    else:
        command.extend([action, "--ticker", ticker])
        if action in {"train", "paper-trade"}:
            command.extend(["--model", body.model_type])
        if action == "train":
            command.extend(["--ridge-alpha", str(body.ridge_alpha)])
        if action in {"train", "compare"}:
            command.extend(["--target", body.target_key])
    try:
        result = subprocess.run(
            command,
            cwd=dashboard_server.ROOT,
            capture_output=True,
            text=True,
            timeout=55,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise HTTPException(
            status_code=504,
            detail="Action exceeded the Vercel request time limit; run long training/update jobs locally.",
        ) from error
    if result.returncode:
        raise HTTPException(status_code=400, detail=result.stderr.strip() or result.stdout.strip())
    return {"ok": True, "output": result.stdout.strip()}
