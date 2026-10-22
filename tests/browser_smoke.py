"""Signed-out Clerk browser smoke; temporary files are cleaned automatically."""
import json
from pathlib import Path
import subprocess
import tempfile
import time

import requests
from websockets.sync.client import connect

with tempfile.TemporaryDirectory(prefix="sp500-auth-smoke-") as directory:
    profile = Path(directory)
    browser = subprocess.Popen([
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--remote-debugging-port=0", "--user-data-dir=" + str(profile), "http://127.0.0.1:5173",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    connection = None
    try:
        deadline = time.monotonic() + 30
        while not (profile / "DevToolsActivePort").exists():
            if time.monotonic() > deadline: raise TimeoutError("Browser did not start")
            time.sleep(.1)
        port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
        page = next(p for p in requests.get(f"http://127.0.0.1:{port}/json", timeout=5).json() if p["type"] == "page")
        connection = connect(page["webSocketDebuggerUrl"], proxy=None, max_size=2000000, legacy=True)
        identifier = 0
        def evaluate(expression):
            global identifier
            identifier += 1
            connection.send(json.dumps({"id": identifier, "method": "Runtime.evaluate", "params": {"expression": expression, "returnByValue": True, "awaitPromise": True}}))
            while True:
                message = json.loads(connection.recv(timeout=30))
                if message.get("id") == identifier:
                    if message.get("result", {}).get("exceptionDetails"): raise RuntimeError(message["result"]["exceptionDetails"])
                    return message.get("result", {}).get("result", {}).get("value")
        deadline = time.monotonic() + 60
        while not evaluate("!!document.querySelector('input[name=identifier]')"):
            if time.monotonic() > deadline:
                raise TimeoutError("Clerk sign-in form did not load: " + str(evaluate("document.body.innerText")))
            time.sleep(.25)
        assert not evaluate("!!document.querySelector('[aria-label=\"Stock search\"]')")
        assert requests.get("http://127.0.0.1:8765/api/stocks", timeout=10).status_code == 401
        assert requests.post("http://127.0.0.1:8765/api/predictions", json={"ticker": "AMD"}, timeout=10).status_code == 401
        assert evaluate("!performance.getEntriesByType('resource').some(r => r.name.includes('/api/stocks') || r.name.includes('/api/state') || r.name.includes('/api/predictions'))")
        print("Clerk sign-in is visible; stock UI is hidden; APIs reject anonymous reads/writes; no data requests before sign-in.")
    finally:
        if connection:
            try: connection.send(json.dumps({"id": 99999, "method": "Browser.close"}))
            except Exception: pass
            connection.close()
        try: browser.wait(timeout=5)
        except subprocess.TimeoutExpired:
            browser.terminate()
            browser.wait(timeout=5)
