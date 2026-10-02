"""ASGI entrypoint for the Vercel Python service."""
from __future__ import annotations

import subprocess

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dashboard_server
import sp500_ml

app = FastAPI(title="S&P 500 Research API")


class ActionRequest(BaseModel):
    ticker: str = "AMD"
    model_type: str = "hist_gradient_boosting"
    ridge_alpha: float = 10.0


@app.get("/api/state")
def read_state(ticker: str = "AMD", model_type: str = "hist_gradient_boosting") -> dict:
    if model_type not in sp500_ml.MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
    try:
        return dashboard_server._read_dashboard_state(ticker, model_type)
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


@app.post("/api/action/{action}")
def run_action(action: str, body: ActionRequest) -> dict:
    allowed_actions = {"update", "train", "paper-trade", "news", "compare"}
    if action not in allowed_actions:
        raise HTTPException(status_code=404, detail="Unknown action")
    try:
        ticker = sp500_ml._normalize_ticker(body.ticker)
        if body.model_type not in sp500_ml.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {body.model_type}")
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
