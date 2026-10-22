import unittest
from unittest.mock import patch
import numpy as np
from sp500_ml.spike_anomalies import analyze_event, read_anomalies, evaluate_outcomes
from sp500_ml.sessions import _calendar


class AnomalyTests(unittest.TestCase):
    def fixture(self):
        dates = [d.date().isoformat() for d in _calendar(2024).sessions_window("2024-01-02", 110)]
        closes = 100 * np.cumprod(1 + np.array([0.] + [.001, -.001] * 54 + [.001]))
        history = [{"date": d, "close": float(c), "volume": 100.} for d, c in zip(dates, closes)]
        i = 70
        history[i]["close"] = history[i - 1]["close"] * 1.12
        history[i]["volume"] = 300.
        e = {"date": dates[i], "previousDate": dates[i-1], "previousClose": history[i-1]["close"], "close": history[i]["close"], "return": .12}
        market = dict.fromkeys(dates, 100.)
        return e, history, market

    def test_exact_volume_market_and_price_observations(self):
        e, h, m = self.fixture()
        a = analyze_event(e, h, m)
        signals = {s["id"]: s for s in a["signals"]}
        self.assertEqual(signals["volume"]["value"], 3.)
        self.assertAlmostEqual(signals["market"]["value"], .12)
        self.assertEqual(signals["price"]["status"], "elevated")
        self.assertEqual(a["baselineReturns"], 60)
        self.assertLess(a["baselineThrough"], e["date"])
        self.assertNotIn("probability", a)

    def test_future_changes_affect_only_outcomes_not_signals(self):
        e, h, m = self.fixture()
        a = analyze_event(e, h, m)
        changed = [{**r, "close": 1., "volume": 1e12} if r["date"] > e["date"] else r for r in h]
        b = analyze_event(e, changed, {d: v if d <= e["date"] else 1e9 for d, v in m.items()})
        self.assertEqual(a["signals"], b["signals"])
        self.assertEqual(a["summary"], b["summary"])
        self.assertNotEqual(a["historicalOutcome"], b["historicalOutcome"])

    def test_market_wide_rally_is_not_stock_outperformance(self):
        e, h, m = self.fixture()
        m[e["date"]] = 112.
        s = analyze_event(e, h, m)["signals"][1]
        self.assertAlmostEqual(s["value"], 0.)
        self.assertEqual(s["status"], "within_rule")

    def test_missing_benchmark_and_invalid_volume_are_unavailable(self):
        e, h, m = self.fixture()
        for row in h: row["volume"] = None
        a = analyze_event(e, h, {})
        self.assertEqual(a["signals"][1]["status"], "unavailable")
        self.assertEqual(a["signals"][2]["status"], "unavailable")
        self.assertEqual(a["historicalOutcome"]["sessions"], 20)

    def test_missing_exchange_session_is_not_skipped(self):
        e, h, m = self.fixture()
        i = next(i for i,r in enumerate(h) if r["date"] == e["date"])
        a = analyze_event(e, h[:i-3] + h[i-2:i+4] + h[i+5:], m)
        self.assertEqual(a["signals"][0]["status"], "unavailable")
        self.assertEqual(a["historicalOutcome"]["sessions"], 3)
        self.assertIsNone(a["historicalOutcome"]["return5"])

    def test_reader_does_not_fetch_news_or_train_models_and_keeps_missing_market(self):
        e, h, _ = self.fixture()
        saved = {"ticker": "AAPL", "from": h[0]["date"], "asOf": h[-1]["date"], "history": h, "events": [e]}
        with patch("sp500_ml.spike_anomalies.read_spikes", return_value=saved), patch("sp500_ml.spike_anomalies.read_csv", side_effect=RuntimeError("No SPY")):
            result = read_anomalies("AAPL", 0)
        self.assertEqual(result["events"][0]["analysis"]["signals"][1]["status"], "unavailable")
        self.assertEqual(result["evaluation"]["eligible"], 0)

    def test_retrospective_counts_do_not_count_overlapping_or_incomplete_events(self):
        e, h, m = self.fixture()
        a = analyze_event(e, h, m)
        a["historicalOutcome"]["largestDecline"] = -.2
        events = [{**e, "analysis": a}, {**e, "analysis": a}]
        result = evaluate_outcomes(events)
        self.assertEqual(result["eligible"], 1)
        self.assertEqual(result["flagged"], 1)
        self.assertEqual(result["flaggedReversals"], 1)
        self.assertEqual(result["excluded"], 1)
