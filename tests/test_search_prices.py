import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sp500_ml import data, search_prices


class SearchPriceTests(unittest.TestCase):
    def test_different_stocks_refresh_concurrently_and_same_stock_is_deduplicated(self):
        frame = pd.DataFrame({"Open": [100.], "High": [102.], "Low": [99.], "Close": [101.], "Volume": [1000]},
                             index=pd.DatetimeIndex(["2026-10-13"], name="Date"))
        barrier = threading.Barrier(3)
        def download(symbols, start):
            barrier.wait(timeout=10)
            return pd.concat({symbol: frame for symbol in symbols}, axis=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "DATA_DIR", Path(directory)), \
                patch.object(search_prices, "_checked", {}), \
                patch.object(search_prices, "utc_now", return_value=pd.Timestamp("2026-10-14T18:00:00Z")), \
                patch.object(data, "_download_batch", side_effect=download) as fetch:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(search_prices.refresh_search_prices, symbol, include_market_inputs=False)
                           for symbol in ["AAA", "BBB", "CCC"]]
                self.assertTrue(all(f.result(timeout=15)["asOf"] == "2026-10-13" for f in futures))
            self.assertEqual(fetch.call_count, 3)
            search_prices._checked.clear()
            fetch.reset_mock()
            fetch.side_effect = lambda symbols, start: pd.concat({symbol: frame for symbol in symbols}, axis=1)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(search_prices.refresh_search_prices, "AAA", include_market_inputs=False) for _ in range(3)]
                results = [f.result(timeout=15) for f in futures]
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(sum(result["cached"] for result in results), 2)

    def test_watch_refresh_does_not_depend_on_unrelated_market_inputs(self):
        frame = pd.DataFrame({"Open": [100.], "High": [102.], "Low": [99.], "Close": [101.], "Volume": [1000]},
                             index=pd.DatetimeIndex(["2026-10-13"], name="Date"))
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "DATA_DIR", Path(directory)), patch.object(search_prices, "_checked", {}), patch.object(search_prices, "utc_now", return_value=pd.Timestamp("2026-10-14T18:00:00Z")), patch.object(data, "_download_batch", return_value=pd.concat({"AAA": frame}, axis=1)) as download:
            result = search_prices.refresh_search_prices("AAA", include_market_inputs=False)
            self.assertEqual(result["asOf"], "2026-10-13")
            self.assertEqual(download.call_args.args[0], ["AAA"])
            self.assertFalse(data._ticker_file("SPY").exists())

    def test_completed_session_handles_early_close_and_holiday(self):
        self.assertEqual(search_prices.latest_completed_session(pd.Timestamp("2026-11-27T18:40:00Z")), pd.Timestamp("2026-11-27"))
        self.assertEqual(search_prices.latest_completed_session(pd.Timestamp("2026-07-03T22:00:00Z")), pd.Timestamp("2026-07-02"))
        self.assertEqual(search_prices.latest_completed_session(pd.Timestamp("2026-10-14T18:00:00Z")), pd.Timestamp("2026-10-13"))

    def test_refresh_writes_required_inputs_and_excludes_partial_bar(self):
        now = pd.Timestamp("2026-10-14T18:00:00Z")
        frame = pd.DataFrame({"Open": [100., 101.], "High": [102., 103.], "Low": [99., 100.],
                              "Close": [101., 102.], "Volume": [1000, 1200]},
                             index=pd.DatetimeIndex(["2026-10-13", "2026-10-14"], name="Date"))
        symbols = ["AAA", "SPY", *data.SECTOR_ETFS, data.VIX_TICKER]
        downloaded = pd.concat({symbol: frame for symbol in symbols}, axis=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "DATA_DIR", Path(directory)), patch.object(search_prices, "_checked", {}), patch.object(search_prices, "utc_now", return_value=now), patch.object(data, "_download_batch", return_value=downloaded) as download:
            result = search_prices.refresh_search_prices("AAA")
            self.assertEqual(result["asOf"], "2026-10-13")
            for symbol in symbols:
                stored = pd.read_csv(data._ticker_file(symbol))
                self.assertEqual(stored["Date"].tolist(), ["2026-10-13"])
            self.assertTrue(search_prices.refresh_search_prices("AAA")["cached"])
            self.assertEqual(download.call_count, 1)

    def test_missing_required_market_bar_does_not_publish_partial_update(self):
        frame = pd.DataFrame({"Open": [100.], "High": [102.], "Low": [99.], "Close": [101.], "Volume": [1000]},
                             index=pd.DatetimeIndex(["2026-10-13"], name="Date"))
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "DATA_DIR", Path(directory)), patch.object(search_prices, "_checked", {}), patch.object(search_prices, "utc_now", return_value=pd.Timestamp("2026-10-14T18:00:00Z")), patch.object(data, "_download_batch", return_value=pd.concat({"AAA": frame}, axis=1)):
            with self.assertRaisesRegex(RuntimeError, "SPY"):
                search_prices.refresh_search_prices("AAA")
            self.assertFalse(data._ticker_file("AAA").exists())

    def test_missing_bulk_bar_is_retried_individually(self):
        frame = pd.DataFrame({"Open": [100.], "High": [102.], "Low": [99.], "Close": [101.], "Volume": [1000]},
                             index=pd.DatetimeIndex(["2026-10-13"], name="Date"))
        symbols = ["AAA", "SPY", *data.SECTOR_ETFS, data.VIX_TICKER]
        bulk = pd.concat({symbol: frame for symbol in symbols if symbol != "XLRE"}, axis=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "DATA_DIR", Path(directory)), patch.object(search_prices, "_checked", {}), patch.object(search_prices, "utc_now", return_value=pd.Timestamp("2026-10-14T18:00:00Z")), patch.object(data, "_download_batch", side_effect=[bulk, pd.concat({"XLRE": frame}, axis=1)]) as download:
            self.assertEqual(search_prices.refresh_search_prices("AAA")["asOf"], "2026-10-13")
            self.assertEqual(download.call_args.args[0], ["XLRE"])
            self.assertTrue(data._ticker_file("AAA").exists())

    def test_delayed_input_preserves_latest_stock_prices_and_reports_shared_date(self):
        frame = pd.DataFrame({"Open": [100., 101.], "High": [102., 103.], "Low": [99., 100.], "Close": [101., 102.], "Volume": [1000, 1200]},
                             index=pd.DatetimeIndex(["2026-10-12", "2026-10-13"], name="Date"))
        symbols = ["AAA", "SPY", *data.SECTOR_ETFS, data.VIX_TICKER]
        delayed = frame.iloc[:1]
        bulk = pd.concat({symbol: delayed if symbol == "XLRE" else frame for symbol in symbols}, axis=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(data, "DATA_DIR", Path(directory)), patch.object(search_prices, "_checked", {}), patch.object(search_prices, "utc_now", return_value=pd.Timestamp("2026-10-14T18:00:00Z")), patch.object(data, "_download_batch", side_effect=[bulk, pd.concat({"XLRE": delayed}, axis=1)]):
            result = search_prices.refresh_search_prices("AAA")
            self.assertEqual(result["asOf"], "2026-10-13")
            self.assertEqual(result["inputAsOf"], "2026-10-12")
            self.assertEqual(pd.read_csv(data._ticker_file("AAA"))["Date"].iloc[-1], "2026-10-13")
            cached = search_prices.refresh_search_prices("AAA")
            self.assertTrue(cached["cached"])
            self.assertEqual(cached["inputAsOf"], result["inputAsOf"])
