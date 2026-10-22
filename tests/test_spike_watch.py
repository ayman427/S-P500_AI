import asyncio
import os
import threading
import unittest
from http.client import HTTPConnection
from unittest.mock import Mock, patch

import pandas as pd

from sp500_ml.auth import AuthenticatedUser
from sp500_ml.spike_watch import run_watch, save_snapshot
from tests.test_auth import invoke_asgi
from tests.test_spike_checklist import sample


class WatchPipelineTests(unittest.TestCase):
    def test_refresh_precedes_evaluation_and_confirmed_save_has_provenance(self):
        prices = sample()
        now = prices.index[-1].tz_localize("UTC") + pd.Timedelta(hours=23)
        order = []
        def refresh(ticker, **kwargs):
            self.assertFalse(kwargs["include_market_inputs"])
            order.append("refresh")
        def read(*args, **kwargs):
            order.append("read")
            return prices
        def save(record):
            order.append("save")
            self.assertEqual(record["ticker"], "AAPL")
            self.assertEqual(len(record["priceFingerprint"]), 64)
            self.assertEqual(record["expectedSession"], record["asOf"])
            self.assertFalse(record["ruleApproved"])
            self.assertNotEqual(record["status"], "caution")
            return "snapshot-id"
        with patch("sp500_ml.spike_watch.refresh_search_prices", side_effect=refresh), patch("sp500_ml.spike_watch.read_csv", side_effect=read), patch("sp500_ml.spike_watch.utc_now", return_value=now), patch("sp500_ml.spike_watch.collect_insider_evidence", return_value={"status": "coverage_incomplete", "retrievedAt": now.isoformat()}) as insider, patch("sp500_ml.spike_watch.save_snapshot", side_effect=save):
            result = run_watch("aapl")
        self.assertEqual(order, ["refresh", "read", "save"])
        self.assertEqual(result["recordId"], "snapshot-id")
        self.assertTrue(result["saved"])
        insider.assert_called_once()
        self.assertEqual(next(check for check in result["checks"] if check["id"] == "insiders")["status"], "unknown")

    def test_refresh_failure_never_publishes_saved_history_as_current(self):
        with patch("sp500_ml.spike_watch.refresh_search_prices", side_effect=RuntimeError("Provider unavailable")), patch("sp500_ml.spike_watch.read_csv") as read, patch("sp500_ml.spike_watch.save_snapshot") as save:
            with self.assertRaises(RuntimeError):
                run_watch("AAPL")
            read.assert_not_called()
            save.assert_not_called()

    def test_http_200_requires_success_and_record_id(self):
        for payload in ({"status": "error", "errorMessage": "private-key"}, {"status": "success", "value": {}}, []):
            response = Mock()
            response.json.return_value = payload
            with patch("sp500_ml.spike_watch.load_environment"), patch("sp500_ml.spike_watch.configured", return_value=True), patch.dict(os.environ, {"CONVEX_URL": "https://example.convex.cloud", "PREDICTION_STORE_SECRET": "private-key"}), patch("sp500_ml.spike_watch.requests.post", return_value=response):
                with self.assertRaises(RuntimeError) as error:
                    save_snapshot({})
                self.assertNotIn("private-key", str(error.exception))

    def test_invalid_ticker_makes_no_requests(self):
        with patch("sp500_ml.spike_watch.refresh_search_prices") as refresh:
            with self.assertRaises(ValueError):
                run_watch("bad/ticker")
            refresh.assert_not_called()


class WatchApiTests(unittest.TestCase):
    def test_asgi_auth_and_failure_behavior(self):
        self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/watch", body={"ticker": "AAPL"})), 401)
        with patch("vercel_app.authenticate", return_value=AuthenticatedUser("alice", "token")), patch("sp500_ml.spike_watch.run_watch", return_value={"saved": True, "recordId": "id"}) as run:
            self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/watch", body={"ticker": "AAPL"})), 200)
            run.assert_called_once_with("AAPL")
            run.side_effect = RuntimeError("private-provider-response")
            self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/watch", body={"ticker": "AAPL"})), 502)

    def test_local_server_requires_auth_and_confirms_snapshot(self):
        import dashboard_server
        import json
        server = dashboard_server.DashboardServer(("127.0.0.1", 0), dashboard_server.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            client.request("POST", "/api/spikes/watch", body="{}")
            response = client.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
            with patch.object(dashboard_server, "authenticate", return_value=AuthenticatedUser("alice", "token")), patch("sp500_ml.spike_watch.run_watch", return_value={"saved": True, "recordId": "id"}):
                client.request("POST", "/api/spikes/watch", body=json.dumps({"ticker": "AAPL"}))
                response = client.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["recordId"], "id")
                client.request("POST", "/api/spikes/watch", body="[]")
                response = client.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
            client.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
