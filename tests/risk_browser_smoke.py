"""Render the real risk component with isolated fixtures; no account or live servers.

The authenticated-fetch hook is replaced only in this temporary test bundle.
Production authentication and the user's running servers are never changed.
"""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse

import requests
from websockets.sync.client import connect

from sp500_ml.risk import evaluate_risk
from sp500_ml.spikes import evaluate_spikes
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def run(tabs=False):
    with tempfile.TemporaryDirectory(prefix="sp500-risk-ui-") as directory:
        temporary = Path(directory)
        bundle = temporary / "dist"
        front = ROOT / "frontend"
        (temporary / "index.html").write_text('<html><div id="root"></div><script type="module" src="/main.tsx"></script></html>')
        (temporary / "main.tsx").write_text('''import React from "react";
import {createRoot} from "react-dom/client";
import RiskPanel from "risk-panel";
import "app-style"; import "base-style";
createRoot(document.getElementById("root")).render(<main className="stock-page"><div className="stock-card"><RiskPanel ticker="AAPL" /></div></main>);
''' if not tabs else '''import React from "react";
import {createRoot} from "react-dom/client";
import App from "risk-panel";
import "base-style";
createRoot(document.getElementById("root")).render(<App userId="isolated-test-user" />);
''')
        script = f'''
import {{build}} from {json.dumps((front / "node_modules/vite/dist/node/index.js").as_uri())};
import react from {json.dumps((front / "node_modules/@vitejs/plugin-react/dist/index.js").as_uri())};
await build({{
  configFile: false, root: {json.dumps(str(temporary))}, logLevel: "error",
  plugins: [{{name: "isolated-risk-fetch", enforce: "pre",
    resolveId(id) {{ if(id === "./useAuthenticatedFetch") return "\\0risk-test-fetch"; if(id === "@clerk/react") return "\\0risk-test-clerk"; }},
    load(id) {{ if(id === "\\0risk-test-clerk") return 'import {{createElement, useState}} from "react"; export function UserButton() {{ const [open,setOpen] = useState(false); return createElement("div", null, createElement("button", {{"aria-label":"Open user profile",onClick:()=>setOpen(!open)}}, "Profile"), open ? createElement("span", null, "Account menu") : null); }}'; if(id === "\\0risk-test-fetch") return 'import {{useCallback}} from "react"; export function useAuthenticatedFetch() {{ return useCallback((url, options) => fetch(url, {{...options, headers: {{...options?.headers, "X-Isolated-Auth": "yes"}}}}), []); }}'; }}
  }}, react()],
  resolve: {{alias: {{
    "risk-panel": {json.dumps(str(front / ("src/App.tsx" if tabs else "src/RiskPanel.tsx")))},
    "app-style": {json.dumps(str(front / "src/App.css"))},
    "base-style": {json.dumps(str(front / "src/index.css"))},
    "react-dom": {json.dumps(str(front / "node_modules/react-dom"))},
    "react": {json.dumps(str(front / "node_modules/react"))}
  }}}}, build: {{outDir: {json.dumps(str(bundle))}}}
}});
'''
        script_path = temporary / "build.mjs"
        script_path.write_text(script)
        subprocess.run(["node", str(script_path)], cwd=ROOT, check=True, timeout=90)
        risk = {"ticker": "AAPL", **evaluate_risk(pd.DataFrame({"Close": [200, 450, 900, 850, 720, 450, 600, 900, 700, 650]}, index=pd.date_range("2020-01-01", periods=10)), years=0)}
        risk["benchmark"] = {"ticker": "SPY", **evaluate_risk(pd.DataFrame({"Close": [100, 105, 110, 108, 106, 104, 110, 120, 119, 121]}, index=pd.date_range("2020-01-01", periods=10)), years=0)}
        spike_returns = [value for i in range(12) for value in ([0.] * 65 + [.1 + i * .005])]
        spike_prices = pd.DataFrame({"Close": 100 * (1 + pd.Series([0., *spike_returns])).cumprod().to_numpy()}, index=pd.bdate_range("2020-01-01", periods=len(spike_returns) + 1))
        spike_prices["Volume"] = [100000 + i * 1000 for i in range(len(spike_prices))]
        spikes = {"ticker": "AAPL", **evaluate_spikes(spike_prices, years=0)}
        for event in spikes["events"]:
            event["analysis"] = {"method": "price_volume_observations_v1", "asOf": event["date"], "title": "Unusual rise detected", "summary": "Price rose beyond its prior range; volume was unusually high.",
                "signals": [{"id": key, "label": label, "value": value, "threshold": threshold, "unit": unit, "status": "elevated"} for key,label,value,threshold,unit in
                    [("price","Price vs normal",.155,.05,"return"),("market","Above SPY",.1,.03,"percentage_points"),("volume","Trading volume",3.,2.,"multiple"),("acceleration","5-session acceleration",.1,.05,"percentage_points")]],
                "baselineFrom": "2019-01-01", "baselineThrough": event["previousDate"], "baselineReturns": 60, "priorDailyVolatility": .01, "marketReturn": .01,
                "historicalOutcome": {"sessions": 20, "through": (pd.Timestamp(event["date"])+pd.Timedelta(days=30)).date().isoformat(), "return5": -.05, "return20": -.1, "largestDecline": -.15, "gainReversed": 1.}}
        spikes["evaluation"] = {"eligible": 5, "flagged": 3, "flaggedReversals": 1, "baselineReversals": 2, "excluded": 7}
        prediction_rows = [
            {"_id": "aapl", "_creationTime": 1578500000000, "ticker": "AAPL", "modelType": "hist_gradient_boosting", "targetKey": "open_to_close", "signalDate": "2020-01-08", "predictedReturn": .005, "status": "pending", "exitDate": "2020-01-09", "modelHash": "hash-aapl"},
            {"_id": "msft", "_creationTime": 1578400000000, "ticker": "MSFT", "modelType": "ridge", "targetKey": "open_to_close_5d", "signalDate": "2020-01-07", "predictedReturn": .02, "actualReturn": .01, "status": "settled", "exitDate": "2020-01-14", "modelHash": "hash-msft"},
        ]
        request_log = []
        request_bodies = []
        trained_selections = set()
        research_attempts = {}

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *_): pass

            def do_GET(self):
                if self.path.startswith("/api/") and self.headers.get("X-Isolated-Auth") != "yes":
                    self.send_error(401)
                    return
                request_log.append(("GET", self.path))
                query = parse_qs(urlparse(self.path).query)
                ticker = query.get("ticker", ["AAPL"])[0]
                if self.path.startswith("/api/risk?"):
                    payload = {**risk, "ticker": ticker}
                    if query.get("years") == ["10"]:
                        payload.update(benchmark=None, benchmarkError="SPY comparison is unavailable for the selected period.")
                elif self.path.startswith("/api/spikes?"):
                    payload = {**spikes, "ticker": ticker}
                    if query.get("years") == ["1"]:
                        payload.update(events=[])
                elif self.path.startswith("/api/stocks"):
                    payload = {"stocks": [{"ticker": "AAPL", "name": "Apple Inc."}, {"ticker": "AMD", "name": "Advanced Micro Devices"}]}
                elif self.path == "/api/market-watch":
                    def row(symbol, met, status="no_signal"):
                        return {"ticker": symbol, "met": met, "status": status, "asOf": "2020-01-08", "checks": [
                            {"id": key, "status": "met" if index < met else "not_met", "value": 2. if key == "volume" else .03}
                            for index, key in enumerate(["volume", "momentum", "breakout"])]}
                    payload = {"expectedSession": "2020-01-08", "scan": {"expectedSession": "2020-01-08", "updatedAt": "2020-01-08T23:00:00Z", "universe": ["AAPL", "NVDA", "MSFT", "AMD"], "processed": 4, "complete": True,
                        "rows": [row("AAPL", 1), row("NVDA", 3, "watch"), row("MSFT", 3, "stale")], "unavailable": ["AMD"]}}
                elif self.path.startswith("/api/predictions"):
                    offset = int(query.get("cursor", ["0"])[0])
                    end = offset + (2 if offset == 0 and len(prediction_rows) > 2 else 1)
                    payload = {"rows": prediction_rows[offset:end], "hasMore": end < len(prediction_rows), "cursor": str(end) if end < len(prediction_rows) else None}
                elif self.path.startswith("/api/state?"):
                    payload = {"ticker": ticker, "tickers": ["AAPL", "AMD"], "modelType": query.get("model_type", ["hist_gradient_boosting"])[0], "targetKey": query.get("target_key", ["open_to_close"])[0], "modelReady": True, "quote": {"close": 900, "dailyChange": .01, "date": "2020-01-08"}, "history": risk["history"], "forecast": {"as_of": "2020-01-08", "predicted_return": .005}, "metrics": {"development_walk_forward": {"trading": {"trades": 10, "wins": 6, "losses": 4, "win_rate": .6, "cumulative_return": .02}}}}
                    if payload["targetKey"] == "open_to_close_5d" and (ticker, payload["modelType"], payload["targetKey"]) not in trained_selections:
                        payload.update(modelReady=False, metrics=None, forecast=None)
                    if payload["modelType"] == "hybrid" and payload["forecast"]:
                        payload["forecast"]["hybrid"] = {"ridgeWeight": .6, "feedbackSamples": 2}
                else:
                    return super().do_GET()
                self.send_payload(payload)

            def do_POST(self):
                if self.headers.get("X-Isolated-Auth") != "yes":
                    self.send_error(401)
                    return
                request_log.append(("POST", self.path))
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                request_bodies.append((self.path, body))
                if self.path == "/api/spikes/gemini":
                    payload = {"ticker": body["ticker"], "eventDate": body["date"], "model": "gemini-3.5-flash-lite", "mode": "evidence", "grounded": False,
                        "independentlyVerified": False, "stored": False, "usableForTraining": False,
                        "text": "Fixture research links dated reporting to a company announcement. Dates need independent checking.",
                        "citations": [{"title": "Original company report", "url": "https://www.apple.com/newsroom/"}],
                        }
                    if body["date"] == spikes["events"][1]["date"]:
                        payload["citations"][0]["url"] = "javascript:alert(1)"
                    self.send_payload(payload)
                elif self.path == "/api/prices/refresh":
                    self.send_payload({"asOf": "2020-01-08", "inputAsOf": "2020-01-08"})
                elif self.path == "/api/predictions":
                    prediction_rows.insert(0, {**prediction_rows[0], "_id": f"saved-{len(prediction_rows)}", "ticker": body.get("ticker", "AAPL"), "modelType": body.get("model_type", "hist_gradient_boosting"), "targetKey": body.get("target_key", "open_to_close"), "_creationTime": 1578600000000})
                    payload = {"saved": True, "created": True, "recordId": "fixture", "message": "Fixture receipt", "rows": []}
                    if body.get("model_type") == "hybrid":
                        payload["forecast"] = {"ticker": body["ticker"], "target_key": body["target_key"], "as_of": "2020-01-08", "predicted_return": .005, "hybrid": {"ridgeWeight": .6, "feedbackSamples": 2}}
                    self.send_payload(payload)
                elif self.path == "/api/spikes/research":
                    event_date = body["date"]
                    research_attempts[event_date] = research_attempts.get(event_date, 0) + 1
                    event_index = next(i for i, row in enumerate(spikes["events"]) if row["date"] == event_date)
                    if event_index == 2 and research_attempts[event_date] == 1:
                        return self.send_payload({"error": "Research service temporarily unavailable"}, status=503)
                    if event_index == 3:
                        time.sleep(.7)
                    payload = {"ticker": body["ticker"], "eventDate": event_date, "saved": True, "recordId": "research-fixture",
                        "documentsRetrieved": 25, "documentsMatched": 1, "windowFrom": (pd.Timestamp(event_date) - pd.Timedelta(days=7)).date().isoformat(), "windowTo": (pd.Timestamp(event_date) + pd.Timedelta(days=3)).date().isoformat(),
                        "confidence": "limited", "evidenceStatus": "possible_explanation", "categories": ["Earnings / guidance", "Buybacks / dividends"],
                        "riskScore": {"status": "experimental", "target": "Close at least 10% below rise-day close within the next 20 trading sessions", "probability": .27,
                            "featureMode": "price_only", "message": "Experimental downside estimate, not a manipulation score.", "modelVersion": "risk-fixture", "trainedThrough": "2010-01-01", "evidenceSources": 0,
                            "drivers": [{"feature": "volatility_20", "direction": "raises"}], "validation": {"samples": 200, "brier": .1, "constantBrier": .15, "rocAuc": .72}},
                        "assessment": {"explanation": "The rise is confirmed, but a timely explanation remains unverified.", "evidenceStrength": "insufficient", "confidenceReason": "Publication timing and primary verification are incomplete.",
                            "comparison": {"status": "small_sample", "samples": 3, "declines": 1, "medianReturn20": .04, "riseFrom": .05, "riseTo": .2, "cutoff": event_date,
                                "examples": [{"date": "2015-01-02", "through": "2015-02-02", "rise": .1, "return20": .04, "maxDrawdown": -.12, "fell10Percent": True}]},
                            "label": "Insufficient evidence", "status": "insufficient_evidence", "narrativeSources": [], "businessSources": [], "reportedBusinessSources": [],
                            "gaps": ["Exact publication time is unavailable."], "eventReturn": .1, "outcomeThrough": event_date, "observedSessions": 20, "peakDate": event_date,
                            "return5": -.1, "return20": -.2, "maxDrawdown": -.25, "gainReversed": 1.5},
                        "sources": [{"title": "Company announces earnings and buyback", "summary": "The company reported quarterly results and authorized share repurchases.",
                                     "url": "https://www.apple.com/newsroom/2024/05/apple-reports-second-quarter-results/", "publisher": "Company announcement", "publishedAt": event_date,
                                     "publicationPrecision": "date", "basis": "company_announcement", "timing": "unknown_time", "contextPhase": "earlier_context"},
                                    {"title": "Unsafe link", "summary": "", "url": "javascript:alert(1)", "publisher": "Invalid", "publishedAt": event_date, "basis": "headline_only", "timing": "unknown_time"}],
                        "coverage": [{"provider": "Finnhub", "status": "empty", "truncated": False}, {"provider": "SEC EDGAR", "status": "unavailable", "truncated": False}, {"provider": "Issuer archive", "status": "ok", "truncated": False}]}
                    if event_index == 1:
                        payload.update(sources=[], categories=[], confidence="unknown", evidenceStatus="no_documents_found")
                        payload["riskScore"].update(status="validation_failed", probability=None, message="The experimental model did not pass its validation gates; no score is shown.")
                    elif event_index == 2:
                        payload["assessment"].update(status="documented_development", label="Business development documented", businessSources=[0], evidenceStrength="limited")
                        payload["sources"][0].update(timing="during_move", contextPhase="move_window", publicationPrecision="timestamp", publishedAt=event_date + "T14:00:00Z")
                        payload["sources"].extend([{**payload["sources"][0], "title": f"Additional business report {i}", "url": f"https://example.com/business-{i}"} for i in (2, 3)])
                        payload["sources"].append({**payload["sources"][0], "title": "Public statement report", "url": "https://example.com/statement", "publisher": "Fixture news", "basis": "provider_excerpt"})
                        payload["assessment"].update(status="mixed_evidence", businessSources=[0, 2, 3], narrativeSources=[4])
                    elif event_index == 7:
                        payload["assessment"]["businessSources"] = [99]
                    elif event_index == 4:
                        payload["ticker"] = "WRONG"
                    elif event_index == 5:
                        payload.update(saved=False, recordId="")
                    elif event_index == 6:
                        payload["riskScore"]["probability"] = 1.5
                    self.send_payload(payload)
                elif self.path == "/api/action/train":
                    trained_selections.add((body["ticker"], body["model_type"], body["target_key"]))
                    self.send_payload({"ok": True})
                else:
                    raise AssertionError("Unexpected training/action request: " + self.path)

            def send_payload(self, payload, status=200):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except ConnectionError:
                    pass  # Selecting another spike intentionally aborts the old request.

        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(bundle)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        profile = temporary / "profile"
        browser = subprocess.Popen([
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--remote-debugging-port=0", "--user-data-dir=" + str(profile),
            f"http://127.0.0.1:{server.server_port}",
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

            def call(method, params):
                nonlocal identifier
                identifier += 1
                connection.send(json.dumps({"id": identifier, "method": method, "params": params}))
                while True:
                    message = json.loads(connection.recv(timeout=30))
                    if message.get("id") == identifier:
                        if message.get("error"): raise RuntimeError(message["error"])
                        return message.get("result", {})

            def evaluate(expression):
                result = call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
                if result.get("exceptionDetails"): raise RuntimeError(result["exceptionDetails"])
                return result.get("result", {}).get("value")

            def wait_for(expression, description):
                deadline = time.monotonic() + 30
                while not evaluate(expression):
                    if time.monotonic() > deadline: raise TimeoutError(description + ": " + str(evaluate("document.body?.innerText")))
                    time.sleep(.1)

            if tabs:
                wait_for("!!document.querySelector('.search-examples button')", "Search UI")
                evaluate("document.querySelector('[aria-label=\"Open user profile\"]').click()")
                wait_for("document.body.innerText.includes('Account menu')", "Profile component")
                evaluate("document.querySelector('[aria-label=\"Open user profile\"]').click()")
                wait_for("document.querySelector('.my-prediction-list h3')?.innerText === 'AAPL'", "Personal prediction history before search")
                evaluate("document.querySelector('.search-examples button').click()")
                wait_for("!!document.querySelector('[role=tablist]') && document.body.innerText.includes('Prediction saved')", "Stock result")
                assert evaluate("document.querySelector('#stock-tab-forecast').getAttribute('aria-selected') === 'true'")
                assert evaluate("!document.querySelector('.model-performance').open")
                wait_for("document.querySelectorAll('.my-prediction-list li').length === 2", "History refresh after save")
                evaluate("document.querySelector('.my-predictions-more').click()")
                wait_for("document.querySelector('.my-prediction-list').innerText.includes('MSFT')", "Older predictions across stocks")
                assert evaluate("document.querySelector('.my-predictions').innerText.includes('Ridge regression')")
                evaluate("document.querySelector('#stock-period').value='open_to_close_5d'; document.querySelector('#stock-period').dispatchEvent(new Event('change',{bubbles:true}))")
                wait_for("document.querySelector('.result-label')?.innerText === 'ESTIMATED 5-SESSION RETURN' && document.querySelector('.save-confirmation-success') !== null", "Five-session forecast")
                assert any(path == "/api/action/train" and body.get("target_key") == "open_to_close_5d" for path, body in request_bodies)
                assert request_bodies[-1][1]["target_key"] == "open_to_close_5d"
                evaluate("document.querySelector('#stock-model').value='ridge'; document.querySelector('#stock-model').dispatchEvent(new Event('change',{bubbles:true}))")
                wait_for("document.querySelector('.model-attribution')?.innerText.includes('Ridge regression') && document.querySelector('.save-confirmation-success') !== null", "Model change preserves period")
                assert request_bodies[-1][1]["target_key"] == "open_to_close_5d"
                assert request_bodies[-1][1]["model_type"] == "ridge"
                evaluate("document.querySelector('#stock-model').value='hybrid'; document.querySelector('#stock-model').dispatchEvent(new Event('change',{bubbles:true}))")
                wait_for("document.querySelector('.model-attribution')?.innerText.includes('Hybrid (experimental)') && document.querySelector('.save-confirmation-success') !== null", "Experimental hybrid selection")
                assert evaluate("document.body.innerText.includes('2 completed feedback outcomes')")
                assert request_bodies[-1][1]["model_type"] == "hybrid"
                assert request_bodies[-1][1]["target_key"] == "open_to_close_5d"
                evaluate("document.querySelector('#stock-period').value='open_to_close'; document.querySelector('#stock-period').dispatchEvent(new Event('change',{bubbles:true}))")
                wait_for("document.querySelector('.result-label')?.innerText === 'ESTIMATED NEXT-SESSION RETURN' && document.querySelector('.save-confirmation-success') !== null", "One-session forecast")
                assert request_bodies[-1][1]["target_key"] == "open_to_close"
                expected_prediction_saves = request_log.count(("POST", "/api/predictions"))
                evaluate("document.querySelector('#sidebar-tab-predictions').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))")
                wait_for("document.querySelector('.market-ranking li summary')?.innerText.includes('NVDA')", "Unsearched stock ranked first")
                assert evaluate("document.activeElement.id === 'sidebar-tab-watch'")
                assert evaluate("document.querySelectorAll('.market-ranking > li').length === 2")
                assert evaluate("document.querySelector('.market-watch').innerText.includes('2 unavailable / incomplete')")
                evaluate("document.querySelector('.market-ranking summary').click()")
                assert evaluate("document.querySelector('.market-ranking details').open")
                assert evaluate("document.querySelector('.market-ranking details').innerText.includes('Volume buildup')")
                assert evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                evaluate("document.querySelector('#sidebar-tab-predictions').click()")
                assert request_log.count(("POST", "/api/predictions")) == expected_prediction_saves
                assert evaluate("!performance.getEntriesByType('resource').some(r=>r.name.includes('/api/risk'))")
                evaluate("document.querySelector('.model-performance summary').click()")
                assert evaluate("document.querySelector('.model-performance').open")
                evaluate("document.querySelector('#stock-tab-forecast').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))")
                assert evaluate("document.activeElement.id === 'stock-tab-risk'")
                assert evaluate("document.querySelector('#stock-panel-forecast').hidden")
                assert evaluate("document.querySelector('#stock-model').closest('[role=tabpanel]').hidden")

            wait_for("!!document.querySelector('.risk-event-facts')", "Risk event details")
            assert evaluate("!document.querySelector('.risk-news-heading, .news-summary-button, .risk-news-list')")
            assert not any(path.startswith(('/api/risk/news', '/api/risk/analysis')) for method, path in request_log)
            assert evaluate("document.body.innerText.includes('-50.0%')")
            assert evaluate("document.body.innerText.includes('Jan 4, 2020')")
            assert evaluate("document.querySelectorAll('.risk-chart svg circle[role=button]').length === 2")
            assert evaluate("document.querySelector('.risk-event-list button').getAttribute('aria-pressed') === 'true'")
            assert evaluate("document.querySelectorAll('.risk-recovery-point').length === 2")
            assert evaluate("document.querySelector('.risk-chart .risk-recovery-point').getAttribute('aria-label').includes('Jan 8, 2020')")
            assert evaluate("Number(document.querySelector('.risk-chart .risk-recovery-point').getAttribute('cy')) === 25"), "Recovery marker must use the actual recovery close"
            assert evaluate("document.querySelectorAll('.selected-decline-range').length === 2")
            assert evaluate("document.querySelector('.risk-price-readout').innerText.includes('$900.00')")
            evaluate("document.querySelector('.risk-price-inspector').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))")
            wait_for("document.querySelector('.risk-price-readout').innerText.includes('$850.00')", "Keyboard price inspection")
            evaluate("const target=document.querySelector('.risk-price-inspector'); const bounds=target.getBoundingClientRect(); target.dispatchEvent(new PointerEvent('pointerdown',{clientX:bounds.left+bounds.width*.6,clientY:bounds.top+20,pointerType:'touch',bubbles:true}))")
            wait_for("document.querySelector('.risk-price-readout').innerText.includes('$450.00')", "Touch price inspection uses actual close")
            previous_highlight = evaluate("document.querySelector('.risk-chart .selected-decline-line').getAttribute('points')")
            evaluate("document.querySelectorAll('.risk-event-list button')[1].click()")
            wait_for("document.querySelectorAll('.risk-event-list button')[1].getAttribute('aria-pressed') === 'true'", "Select unrecovered decline")
            assert evaluate("!document.querySelector('.risk-recovery-point')")
            assert evaluate("document.querySelector('.risk-price-readout').innerText.includes('Jan 8, 2020')"), "Changing events resets the inspected price"
            assert evaluate("document.querySelector('.risk-event-legend').innerText.includes('Not recovered')")
            assert evaluate("document.querySelector('.risk-chart .selected-decline-line').getAttribute('points')") != previous_highlight
            evaluate("document.querySelector('.risk-chart circle[role=button]').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}))")
            wait_for("document.querySelectorAll('.risk-recovery-point').length === 2", "Keyboard selection restores recovery marker")
            assert evaluate("!document.querySelector('.risk-market-details').open"), "Secondary chart is collapsed initially"
            evaluate("document.querySelector('.risk-market-details summary').click()")
            assert evaluate("document.querySelector('.risk-comparison table').innerText.includes('SPY')")
            assert evaluate("document.querySelectorAll('.risk-drawdown-chart polyline:not(.selected-decline-line)').length === 2")
            assert evaluate("document.querySelector('.risk-drawdown-chart svg').getAttribute('aria-label').includes('compared with SPY')")
            assert evaluate("!document.querySelector('.risk-drawdown-chart svg').innerHTML.includes('NaN')")
            for width in (1280, 375):
                call("Emulation.setDeviceMetricsOverride", {"width": width, "height": 900, "deviceScaleFactor": 1, "mobile": width < 600})
                evaluate("document.documentElement.dataset.theme='dark'")
                assert evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Horizontal overflow"
                assert evaluate("getComputedStyle(document.querySelector('.risk-panel')).getPropertyValue('--risk-event').trim() === '#ed9791'")
                if tabs:
                    assert evaluate("document.querySelector('.my-predictions').getBoundingClientRect().left >= document.querySelector('.stock-card').getBoundingClientRect().right" if width > 1050 else "document.querySelector('.my-predictions').getBoundingClientRect().top >= document.querySelector('.stock-card').getBoundingClientRect().bottom")
            if tabs:
                assert not any(path.startswith('/api/spikes') for method, path in request_log), "Spikes load only on demand"
                evaluate("document.querySelector('#risk-view-declines').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))")
                wait_for("!!document.querySelector('#spike-event')", "Price spikes tab")
                assert evaluate("document.querySelectorAll('.result-tabs [role=tab]').length === 2")
                assert evaluate("document.querySelector('#risk-content-declines').hidden && !document.querySelector('#risk-content-spikes').hidden")
                assert evaluate("document.querySelectorAll('#spike-event option').length === 12")
                assert evaluate("!document.querySelector('.spike-current-watch, .spike-checklist')")
                assert not any(path == "/api/spikes/watch" for method, path in request_log), "Removed watch must not make requests"
                assert evaluate("document.querySelector('.spike-chart h4').innerText.includes('+15.5%')")
                assert evaluate("document.querySelectorAll('.spike-volume-bars rect').length > 0")
                assert evaluate("document.querySelector('.spike-panel output').innerText.includes('Volume:')")
                wait_for("document.querySelector('.anomaly-title')?.innerText === 'Unusual rise detected'", "Price/volume anomaly results")
                assert evaluate("document.querySelectorAll('.anomaly-signals > div').length === 4")
                assert evaluate("document.querySelector('.anomaly-signals').innerText.includes('3.00x')")
                assert evaluate("!document.querySelector('.rise-details').open")
                assert not any(path in {"/api/spikes/research", "/api/spikes/gemini", "/api/spikes/watch"} for method,path in request_log)
                evaluate("document.querySelector('.rise-details summary').click()")
                assert evaluate("document.querySelector('.anomaly-outcome').innerText.includes('historical only')")
                assert evaluate("document.querySelector('.rise-details').innerText.includes('Price-spike baseline: 2 of 5')")
                previous_price = evaluate("document.querySelector('.spike-panel output').innerText")
                evaluate("document.querySelector('.spike-chart svg').dispatchEvent(new KeyboardEvent('keydown',{key:'Home',bubbles:true}))")
                wait_for("document.querySelector('.spike-chart svg').getAttribute('aria-valuenow') === '0'", "Accessible price inspection")
                assert evaluate("document.querySelector('.spike-panel output').innerText") != previous_price
                evaluate("document.querySelector('#spike-event').value=document.querySelector('#spike-event').options[1].value; document.querySelector('#spike-event').dispatchEvent(new Event('change',{bubbles:true}))")
                wait_for("document.querySelector('#spike-event').selectedIndex === 1", "Select another spike")
                assert evaluate("!document.querySelector('.rise-details').open"), "Outcome drawer resets for another event"
                for width in (1280, 375):
                    call("Emulation.setDeviceMetricsOverride", {"width": width, "height": 900, "deviceScaleFactor": 1, "mobile": width < 600})
                    assert evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Spike tab overflows"
                evaluate("document.querySelector('#spike-history').value='1'; document.querySelector('#spike-history').dispatchEvent(new Event('change',{bubbles:true}))")
                wait_for("document.querySelector('.spike-panel').innerText.includes('No spikes meet this rule')", "No spikes state")
                assert evaluate("!document.querySelector('.spike-chart')")
                assert request_log.count(("POST", "/api/predictions")) == expected_prediction_saves
                evaluate("document.querySelector('#risk-view-declines').click()")
                assert evaluate("!document.querySelector('#risk-content-declines').hidden && document.querySelector('#risk-content-spikes').hidden")
                evaluate("document.querySelector('#risk-view-spikes').click()")
                assert evaluate("document.querySelector('#spike-history').value === '1'")
                evaluate("document.querySelector('#risk-view-declines').click()")
                evaluate("document.querySelector('.risk-controls select').value='10'; document.querySelector('.risk-controls select').dispatchEvent(new Event('change',{bubbles:true}))")
                wait_for("document.querySelector('.risk-controls select').value === '10' && !document.body.innerText.includes('Comparing historical prices')", "History selection")
                wait_for("!!document.querySelector('.risk-market-details')", "Market comparison loaded")
                evaluate("document.querySelector('.risk-market-details summary').click()")
                wait_for("document.querySelector('.risk-comparison').innerText.includes('unavailable')", "Missing benchmark state")
                assert evaluate("document.querySelectorAll('.risk-drawdown-chart polyline:not(.selected-decline-line)').length === 1")
                evaluate("document.querySelector('#stock-tab-forecast').click()")
                assert evaluate("!document.querySelector('#stock-panel-forecast').hidden && document.querySelector('#stock-panel-risk').hidden")
                evaluate("document.querySelector('#stock-tab-risk').click()")
                assert evaluate("document.querySelector('.risk-controls select').value === '10'")
                assert request_log.count(("POST", "/api/predictions")) == expected_prediction_saves
                evaluate("const input=document.querySelector('#stock-symbol'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,'AMD'); input.dispatchEvent(new Event('input',{bubbles:true}));")
                evaluate("document.querySelector('form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))")
                wait_for("document.querySelector('.stock-identity h2')?.innerText === 'AMD'", "New stock")
                assert evaluate("document.querySelector('#stock-tab-forecast').getAttribute('aria-selected') === 'true'")
                assert evaluate("!document.querySelector('.risk-panel')")
                print("Rise investigation shows four measured price/volume signals, historical outcomes and one Details drawer, without news or model calls.")
                print("Stock tabs default to Forecast, support keyboard navigation, preserve risk settings, avoid duplicate saves, and reset on a new stock.")
            print("Risk panel compares SPY, plots stock/market drawdowns, handles missing market data, and keeps mobile/dark layouts without news requests.")
        finally:
            if connection:
                try: connection.send(json.dumps({"id": 99999, "method": "Browser.close"}))
                except Exception: pass
                connection.close()
            try: browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser.terminate()
                browser.wait(timeout=5)
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    import sys
    run(tabs="--tabs" in sys.argv)
