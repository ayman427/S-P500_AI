import asyncio
import copy
import os
import threading
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pandas as pd

from sp500_ml import market_watch as watch
from sp500_ml.auth import AuthenticatedUser
from tests.test_auth import invoke_asgi


class MarketWatchTests(unittest.TestCase):
    def test_hosted_routes_require_auth_and_share_only_market_results(self):
        for method, path, function in [("GET", "/api/market-watch", "read_scan"), ("POST", "/api/market-watch/scan", "scan_batch")]:
            with patch.object(watch, function, return_value={"scan": None}) as action:
                self.assertEqual(asyncio.run(invoke_asgi(method, path)), 401)
                action.assert_not_called()
                with patch("vercel_app.authenticate", return_value=AuthenticatedUser("alice", "token")):
                    self.assertEqual(asyncio.run(invoke_asgi(method, path)), 200)
                    action.side_effect = RuntimeError("private-provider-response")
                    self.assertEqual(asyncio.run(invoke_asgi(method, path)), 502)

    def test_resume_scans_unsearched_stocks_and_checkpoints_failures(self):
        universe = ["A" + chr(65 + i // 26) + chr(65 + i % 26) for i in range(400)]
        stored = None
        writes = []
        barrier = threading.Barrier(3)
        lease = {"key": "market-checklist", "owner": "test-worker"}
        token = watch._lease.set(lease)
        self.addCleanup(watch._lease.reset, token)
        def rpc(kind, name, **args):
            nonlocal stored
            if name == "latest":
                restored = copy.deepcopy(stored)
                if restored is not None:
                    restored["processed"] = float(restored["processed"])
                return restored
            stored = copy.deepcopy(args["scan"])
            writes.append(stored["processed"])
            return {"id": "confirmed"}
        prices = pd.DataFrame({"Close": [100]}, index=pd.DatetimeIndex(["2026-10-19"]))
        result = {"asOf": "2026-10-19", "status": "watch", "checks": [
            {"id": key, "active": True, "status": "met", "value": value}
            for key, value in [("volume", 2.), ("momentum", .03), ("breakout", .01)]]}
        def refresh(ticker, **kwargs):
            self.assertFalse(kwargs["include_market_inputs"])
            self.assertEqual(watch._lease.get(), lease)
            barrier.wait(timeout=10)
            if ticker == universe[1]:
                raise RuntimeError("Provider failed")
        with patch.object(watch, "exclusive_cloud_operation", side_effect=lambda _: nullcontext()), \
                patch.object(watch, "_rpc", side_effect=rpc), \
                patch.object(watch, "utc_now", return_value=pd.Timestamp("2026-10-19T23:00Z")), \
                patch.object(watch, "get_sp500_tickers", return_value=universe) as constituents, \
                patch.object(watch, "refresh_search_prices", side_effect=refresh) as sync, \
                patch.object(watch, "read_csv", return_value=prices), \
                patch.object(watch, "evaluate_checklist", return_value=result):
            watch.scan_batch()
            resumed = watch.scan_batch()["scan"]
        self.assertEqual(writes, [1, 2, 3, 4, 5, 6])
        self.assertEqual(resumed["processed"], 6)
        self.assertEqual(resumed["unavailable"], [universe[1]])
        self.assertCountEqual([call.args[0] for call in sync.call_args_list], universe[:6])
        self.assertEqual([r["ticker"] for r in resumed["rows"]], [t for t in universe[:6] if t != universe[1]])
        self.assertEqual([r["met"] for r in resumed["rows"]], [3] * 5)
        self.assertFalse(resumed["complete"])
        constituents.assert_called_once()

    def test_complete_scan_does_not_redownload_without_rescan(self):
        scan = {"complete": True, "expectedSession": "2026-10-19"}
        with patch.object(watch, "exclusive_cloud_operation", return_value=nullcontext()), \
                patch.object(watch, "utc_now", return_value=pd.Timestamp("2026-10-19T23:00Z")), \
                patch.object(watch, "_rpc", return_value=scan), \
                patch.object(watch, "refresh_search_prices") as refresh:
            self.assertEqual(watch.scan_batch()["scan"], scan)
            refresh.assert_not_called()

    def test_concurrent_scan_returns_progress_instead_of_starting_again(self):
        with patch.object(watch, "_scan_batch", side_effect=RuntimeError("Another operation holds this cloud lock")), \
                patch.object(watch, "read_scan", return_value={"scan": None}):
            self.assertTrue(watch.scan_batch()["busy"])

    def test_http_success_needs_confirmed_mutation_id(self):
        for payload in ([], {"status": "error"}, {"status": "success", "value": {}}):
            response = Mock()
            response.json.return_value = payload
            with patch.object(watch, "load_environment"), patch.dict(os.environ, {
                    "CONVEX_URL": "https://example.convex.cloud", "PREDICTION_STORE_SECRET": "secret"}), \
                    patch.object(watch.requests, "post", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "storage"):
                    watch._rpc("mutation", "save")
