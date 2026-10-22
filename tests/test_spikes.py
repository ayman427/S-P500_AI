import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sp500_ml.spikes import evaluate_spikes, read_spikes


def frame(returns):
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + np.array([0, *returns]))},
                        index=pd.bdate_range("2020-01-01", periods=len(returns) + 1))


class SpikeTests(unittest.TestCase):
    def test_history_volume_is_aligned_and_invalid_values_are_null(self):
        prices = frame([0.] * 30 + [.1])
        prices["Volume"] = [0., -1., float("nan"), float("inf"), *([150.] * 28)]
        rows = evaluate_spikes(prices, 0)["history"]
        self.assertEqual([row["volume"] for row in rows[:5]], [0., None, None, None, 150.])
        shuffled = prices.sample(frac=1, random_state=42)
        self.assertEqual(evaluate_spikes(shuffled, 0)["history"], rows)
        self.assertTrue(all(row["volume"] is None for row in evaluate_spikes(prices.drop(columns="Volume"), 0)["history"]))

    def test_prior_activity_excludes_event_and_future_and_uses_disjoint_baseline(self):
        prices = frame([.001] * 60 + [.12, -.2])
        prices["Volume"] = 100.
        prices.iloc[51:61, prices.columns.get_loc("Volume")] = 200.
        prices.iloc[61:, prices.columns.get_loc("Volume")] = 1e9
        event = evaluate_spikes(prices, 0)["events"][0]
        activity = event["priorActivity"]
        self.assertEqual(activity["asOf"], event["previousDate"])
        ten = activity["windows"][2]
        self.assertEqual(ten["volumeRatio"], 2.)
        self.assertAlmostEqual(ten["priceReturn"], 1.001 ** 10 - 1)
        self.assertAlmostEqual(ten["meanAbsoluteReturn"], .001)
        changed = prices.copy()
        changed.iloc[61:, changed.columns.get_loc("Volume")] = 0
        self.assertEqual(evaluate_spikes(changed, 0)["events"][0]["priorActivity"], activity)

    def test_missing_invalid_and_zero_baseline_volume_is_unavailable(self):
        prices = frame([0.] * 60 + [.1])
        for volume in (None, 0., -1., float("nan"), float("inf")):
            candidate = prices.copy()
            if volume is not None:
                candidate["Volume"] = volume
            activity = evaluate_spikes(candidate, 0)["events"][0]["priorActivity"]
            self.assertTrue(all(row["volumeRatio"] is None for row in activity["windows"]))
        prices["Volume"] = 100.
        duplicate = pd.concat([prices.iloc[:5], prices]).sort_index(kind="stable")
        self.assertEqual(evaluate_spikes(duplicate, 0)["events"][0]["priorActivity"],
                         evaluate_spikes(prices, 0)["events"][0]["priorActivity"])

    def test_spike_uses_only_prior_returns_and_keeps_exact_prices(self):
        prices = frame([.001, -.001] * 30 + [.12, -.2, .001])
        result = evaluate_spikes(prices, years=0)
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertAlmostEqual(event["return"], .12)
        self.assertAlmostEqual(event["threshold"], .05)
        self.assertEqual(event["close"], prices.Close.iloc[61])
        self.assertEqual(event["previousClose"], prices.Close.iloc[60])
        self.assertEqual(len(result["history"]), len(prices))

    def test_high_volatility_and_small_gains_are_not_spikes(self):
        for returns in ([.04] * 70, [.1, -.1] * 30 + [.1], [0.] * 70):
            self.assertEqual(evaluate_spikes(frame(returns))["events"], [])

    def test_warmup_short_history_and_invalid_prices(self):
        self.assertEqual(evaluate_spikes(frame([0.] * 19 + [.5]))["events"], [])
        self.assertEqual(len(evaluate_spikes(frame([0.] * 20 + [.5]))["events"]), 1)
        self.assertEqual(evaluate_spikes(frame([.1]))["assessedSessions"], 0)
        for values in ([100, 0], [100, float("nan")], [100, float("inf")], [100]):
            with self.assertRaises(ValueError):
                evaluate_spikes(pd.DataFrame({"Close": values}, index=pd.date_range("2020", periods=len(values))))
        with self.assertRaises(ValueError):
            evaluate_spikes(frame([0.] * 60), years=2)

    def test_period_filters_events_but_retains_prior_baseline(self):
        prices = frame([0.] * 25 + [.1] + [0.] * 300 + [.2])
        result = evaluate_spikes(prices, years=1)
        self.assertEqual(len(result["events"]), 1)
        self.assertAlmostEqual(result["events"][0]["return"], .2)
        self.assertEqual(len(evaluate_spikes(prices, years=0)["events"]), 2)

    def test_reader_uses_cloud_and_completed_sessions(self):
        prices = frame([0.] * 30 + [.1])
        with patch("sp500_ml.spikes._ticker_file") as path, patch("sp500_ml.spikes.read_csv", return_value=prices) as read, patch("sp500_ml.spikes._drop_incomplete_session", return_value=prices.iloc[:-1]) as complete:
            result = read_spikes("aapl", years=0)
            self.assertEqual(result["ticker"], "AAPL")
            self.assertEqual(result["events"], [])
            read.assert_called_once()
            complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
