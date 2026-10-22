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
    def setUp(self):
        cache = patch.object(watch, "_scan_cache", None)
        cache.start(); self.addCleanup(cache.stop)

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
        def refresh(tickers, start):
            ticker = tickers[0]
            self.assertEqual(watch._lease.get(), lease)
            barrier.wait(timeout=10)
            if ticker == universe[1]:
                raise RuntimeError("Provider failed")
            return prices
        with patch.object(watch, "exclusive_cloud_operation", side_effect=lambda _: nullcontext()), \
                patch.object(watch, "_rpc", side_effect=rpc), \
                patch.object(watch, "utc_now", return_value=pd.Timestamp("2026-10-19T23:00Z")), \
                patch.object(watch, "get_sp500_tickers", return_value=universe) as constituents, \
                patch.object(watch, "_download_batch", side_effect=refresh) as sync, \
                patch.object(watch, "_extract_ticker", side_effect=lambda frame, ticker: frame), \
                patch.object(watch, "evaluate_checklist", return_value=result):
            watch.scan_batch()
            resumed = watch.scan_batch()["scan"]
        self.assertEqual(writes, [15, 30])
        self.assertEqual(resumed["processed"], 30)
        self.assertEqual(resumed["unavailable"], [universe[1]])
        self.assertCountEqual([call.args[0][0] for call in sync.call_args_list], universe[:30])
        self.assertEqual([r["ticker"] for r in resumed["rows"]], [t for t in universe[:30] if t != universe[1]])
        self.assertEqual([r["met"] for r in resumed["rows"]], [3] * 29)
        self.assertFalse(resumed["complete"])
        constituents.assert_called_once()

    def test_scanner_uses_only_recent_provider_bars_and_rejects_stale_data(self):
        expected = pd.Timestamp("2026-10-19")
        prices = pd.DataFrame({"Close": [100, 101]}, index=pd.to_datetime(["2026-10-19", "2026-10-20"]))
        result = {"asOf": "2026-10-19", "status": "watch", "checks": []}
        with patch.object(watch, "_download_batch", return_value=prices) as download, \
                patch.object(watch, "_extract_ticker", side_effect=lambda frame, ticker: frame), \
                patch.object(watch, "evaluate_checklist", return_value=result) as evaluate, \
                patch("sp500_ml.cloud_data.read_bytes", side_effect=AssertionError("No cloud files")), \
                patch("sp500_ml.cloud_data.write_bytes", side_effect=AssertionError("No cloud files")):
            watch._scan_stock("AAPL", pd.Timestamp("2026-10-19T23:00Z"), expected)
            download.assert_called_once_with(["AAPL"], "2026-04-22")
            self.assertEqual(evaluate.call_args.args[0].index.max(), expected)
            download.return_value = prices.iloc[:0]
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                watch._scan_stock("AAPL", pd.Timestamp("2026-10-19T23:00Z"), expected)

    def test_complete_scan_does_not_redownload_without_rescan(self):
        scan = {"complete": True, "expectedSession": "2026-10-19"}
        with patch.object(watch, "exclusive_cloud_operation", return_value=nullcontext()), \
                patch.object(watch, "utc_now", return_value=pd.Timestamp("2026-10-19T23:00Z")), \
                patch.object(watch, "_rpc", return_value=scan), \
                patch.object(watch, "_download_batch") as refresh:
            self.assertEqual(watch.scan_batch()["scan"], scan)
            refresh.assert_not_called()

    def test_completed_scan_reads_are_shared_and_session_change_invalidates_cache(self):
        scan = {"complete": True, "expectedSession": "2026-10-19"}
        with patch.object(watch,"load_environment"), patch.object(watch,"utc_now",return_value=pd.Timestamp("2026-10-19T23:00Z")), patch.object(watch,"_rpc",return_value=scan) as rpc:
            first = watch.read_scan()
            first["scan"]["complete"] = False
            self.assertTrue(watch.read_scan()["scan"]["complete"])
            self.assertTrue(watch.scan_batch()["scan"]["complete"])
            rpc.assert_called_once()
        with patch.object(watch,"load_environment"), patch.object(watch,"utc_now",return_value=pd.Timestamp("2026-10-20T23:00Z")), patch.object(watch,"_rpc",return_value=scan) as rpc:
            self.assertEqual(watch.read_scan()["expectedSession"], "2026-10-20")
            rpc.assert_called_once()

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
