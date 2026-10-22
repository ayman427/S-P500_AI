import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from sp500_ml.risk import evaluate_risk, read_risk
from sp500_ml.risk_news import archive_headlines, match_event_news, read_event_news


def prices(values, start="2020-01-01"):
    return pd.DataFrame({"Close": values}, index=pd.date_range(start, periods=len(values)))


def headline(date, title="Historical article", url="https://finance.yahoo.com/news/example.html"):
    return {"published_at_utc": date, "title": title, "url": url, "publisher": "Example publisher"}


class RiskTests(unittest.TestCase):
    def compare(self, stock, market):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AAPL.csv").touch()
            with patch("sp500_ml.risk._ticker_file", side_effect=lambda ticker: root / f"{ticker}.csv"), patch("sp500_ml.risk.read_csv", side_effect=lambda path, **kwargs: stock if path.stem == "AAPL" else market), patch("sp500_ml.risk._drop_incomplete_session", side_effect=lambda frame: frame):
                return read_risk("AAPL", years=0)

    def test_market_comparison_uses_identical_sessions_and_resets_peak_at_window_start(self):
        stock = prices([200, 900, 450, 900], start="2020-01-02")
        market = prices([1000, 100, 110, 104.5, 121, 2000])
        result = self.compare(stock, market)
        benchmark = result["benchmark"]
        self.assertEqual((benchmark["from"], benchmark["asOf"], benchmark["sessions"]),
                         (result["from"], result["asOf"], result["sessions"]))
        self.assertAlmostEqual(benchmark["totalReturn"], .21)
        self.assertAlmostEqual(benchmark["maxDrawdown"], -.05)
        self.assertEqual(benchmark["history"][0]["drawdown"], 0)
        self.assertEqual(result["maxDrawdown"], -.5)

    def test_missing_or_invalid_market_data_does_not_hide_stock_risk_or_use_unequal_dates(self):
        stock = prices([100, 120, 60, 100])
        for market in (prices([100, 105, 110]), prices([100, 0, 110, 120]), prices([100, 105, float("nan"), 120])):
            with self.subTest(market=market):
                result = self.compare(stock, market)
                self.assertIsNone(result["benchmark"])
                self.assertIn("unavailable", result["benchmarkError"])
                self.assertEqual(result["maxDrawdown"], -.5)

    def test_sampled_drawdown_chart_keeps_the_true_worst_point_even_without_qualifying_events(self):
        frame = prices([100.] * 1500)
        frame.iloc[713, 0] = 81.
        result = evaluate_risk(frame, years=0, threshold=.30)
        self.assertEqual(result["events"], [])
        self.assertAlmostEqual(min(row["drawdown"] for row in result["history"]), -.19)

    def test_rise_to_peak_then_fall_and_recovery(self):
        result = evaluate_risk(prices([200, 450, 900, 850, 720, 450, 600, 900]), years=0)
        self.assertAlmostEqual(result["maxDrawdown"], -0.5)
        self.assertEqual(result["currentDrawdown"], 0)
        event = result["events"][0]
        self.assertEqual(event["peakDate"], "2020-01-03")
        self.assertEqual(event["startDate"], "2020-01-04")
        self.assertEqual(event["thresholdDate"], "2020-01-05")
        self.assertEqual(event["troughDate"], "2020-01-06")
        self.assertEqual(event["recoveryDate"], "2020-01-08")
        self.assertEqual(event["riseFromWindowStart"], 3.5)
        self.assertEqual(event["reboundNeededAtTrough"], 1.0)
        self.assertEqual(event["daysUnderwater"], 5)

    def test_interim_bounces_do_not_create_separate_declines(self):
        result = evaluate_risk(prices([100, 110, 85, 100, 75, 80]), years=0)
        self.assertEqual(result["qualifyingDeclines"], 1)
        self.assertIsNone(result["events"][0]["recoveryDate"])
        self.assertAlmostEqual(result["currentDrawdown"], 80 / 110 - 1)

    def test_recovery_allows_new_episode_and_flat_prices_are_safe(self):
        result = evaluate_risk(prices([100, 80, 100, 120, 60, 120]), years=0)
        self.assertEqual(result["qualifyingDeclines"], 2)
        self.assertEqual(result["events"][0]["decline"], -0.5)
        flat = evaluate_risk(prices([100, 100, 100]), years=0)
        self.assertEqual(flat["maxDrawdown"], 0)
        self.assertEqual(flat["annualizedVolatility"], 0)
        self.assertEqual(flat["events"], [])

    def test_chart_preserves_event_dates_and_history_window(self):
        series = prices([100] * 1000 + [120, 110, 80] + [90] * 1000)
        result = evaluate_risk(series, years=0)
        dates = {row["date"] for row in result["history"]}
        for key in ("peakDate", "startDate", "troughDate", "thresholdDate"):
            self.assertIn(result["events"][0][key], dates)
        recent = evaluate_risk(series, years=1)
        self.assertGreaterEqual(pd.Timestamp(recent["from"]), series.index[-1] - pd.DateOffset(years=1))
        self.assertEqual(recent["availableFrom"], "2020-01-01")

    def test_invalid_prices_and_options_are_rejected(self):
        for values in ([100], [100, 0], [100, float("nan")], [100, float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_risk(prices(values))
        with self.assertRaises(ValueError): evaluate_risk(prices([100, 90]), threshold=0.01)
        with self.assertRaises(ValueError): evaluate_risk(prices([100, 90]), years=2)


class EventNewsTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"FINNHUB_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)
        settings = patch("sp500_ml.prediction_store.load_environment")
        settings.start()
        self.addCleanup(settings.stop)

    def test_same_day_is_new_york_calendar_date_and_excludes_unrelated_dates(self):
        rows = [headline("2020-01-04T01:00:00Z"), headline("2020-01-04T15:00:00Z", url="https://finance.yahoo.com/news/later.html")]
        result = match_event_news(rows, "2020-01-03")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["publicationDate"], "2020-01-03")
        self.assertTrue(result[0]["sameDay"])
        nearby = match_event_news(rows, "2020-01-03", 3)
        self.assertEqual(nearby[1]["daysFromEvent"], 1)

    def test_future_bad_urls_and_undated_headlines_never_become_evidence(self):
        rows = [headline("2030-01-01T12:00:00Z"), headline("2020-01-03T12:00:00Z", url="javascript:alert(1)"), headline(None)]
        self.assertEqual(match_event_news(rows, "2020-01-03", now=pd.Timestamp("2020-01-04T00:00:00Z")), [])
        for invalid in ("", "today", "2020-01-01T12:00:00Z"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): match_event_news([], invalid)

    def test_archive_accumulates_and_deduplicates_without_local_runtime_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = headline("2020-01-03T12:00:00Z")
            new = headline("2021-01-03T12:00:00Z", url="https://finance.yahoo.com/news/new.html")
            archive_headlines("AAPL", [old], directory=root)
            archive_headlines("AAPL", [old, new], directory=root)
            payload = json.loads((root / "archive" / "AAPL.json").read_text())
            self.assertEqual(len(payload["headlines"]), 2)

    def test_missing_historical_coverage_is_explicit_not_replaced_by_recent_news(self):
        with tempfile.TemporaryDirectory() as directory, patch("sp500_ml.risk_news.NEWS_DIR", Path(directory)), patch("sp500_ml.risk_news.fetch_yahoo_headlines", side_effect=requests.Timeout):
            result = read_event_news("AAPL", "2020-01-03")
            self.assertEqual(result["headlines"], [])
            self.assertEqual(result["providerStatus"], "unavailable")
            self.assertIn("does not mean no news occurred", result["message"])

    def test_recent_cache_serves_matching_event_without_provider_request(self):
        with tempfile.TemporaryDirectory() as directory, patch("sp500_ml.risk_news.NEWS_DIR", Path(directory)), patch("sp500_ml.risk_news.fetch_yahoo_headlines") as fetch:
            path = Path(directory) / "AAPL.json"
            path.write_text(json.dumps({"fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "headlines": [headline("2020-01-03T12:00:00Z")]}))
            result = read_event_news("AAPL", "2020-01-03")
            self.assertEqual(len(result["headlines"]), 1)
            fetch.assert_not_called()
