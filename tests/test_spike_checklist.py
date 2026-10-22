import unittest

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from sp500_ml.spike_checklist import evaluate_checklist


def sample():
    changes = np.zeros(100)
    changes[-5:] = .01
    prices = pd.DataFrame({"Close": 100 * np.cumprod(1 + changes), "Volume": 100.},
                          index=xcals.get_calendar("XNYS", start="2025-01-01", end="2026-01-01").sessions[:100])
    prices.iloc[-3:, 1] = 200.
    return prices


class ChecklistTests(unittest.TestCase):
    def test_all_conditions_met_without_validation_does_not_claim_caution(self):
        prices = sample()
        row = evaluate_checklist(prices, now=prices.index[-1] + pd.Timedelta(hours=23))
        self.assertEqual(row["status"], "watch")
        self.assertEqual([check["status"] for check in row["checks"]], ["met"] * 3 + ["unknown"] * 2)
        self.assertEqual(row["checks"][0]["value"], 2.)
        self.assertFalse(row["validation"]["supported"])
        self.assertFalse(row["usableForTraining"])

    def test_missing_data_stays_unknown_and_zero_volume_is_not_buildup(self):
        prices = sample()
        for value in (None, -1., float("nan"), 0.):
            candidate = prices.drop(columns="Volume") if value is None else prices.assign(Volume=value)
            row = evaluate_checklist(candidate, now=prices.index[-1])
            self.assertEqual(row["status"], "incomplete")
            self.assertEqual(row["checks"][0]["status"], "unknown")

    def test_stale_snapshot_withholds_alert_and_current_features_ignore_later_data(self):
        prices = sample()
        row = evaluate_checklist(prices, now=prices.index[-1] + pd.Timedelta(days=10))
        self.assertEqual(row["status"], "stale")
        earlier = evaluate_checklist(prices.iloc[:-1], now=prices.index[-2])["checks"]
        prices.iloc[-1] = [1e6, 1e9]
        self.assertEqual(evaluate_checklist(prices.iloc[:-1], now=prices.index[-2])["checks"], earlier)

    def test_holdout_excludes_last_five_unresolved_sessions_and_counts_false_alerts(self):
        prices = sample()
        row = evaluate_checklist(prices, now=prices.index[-1])
        validation = row["validation"]
        self.assertEqual(validation["sessions"], 25)
        self.assertEqual(validation["alerts"], validation["hits"] + validation["falseAlerts"])
        self.assertEqual(validation["alerts"], 0)  # Current setup has no known future outcome yet.

    def test_repeated_synthetic_precursors_can_meet_evidence_gate(self):
        returns, volume = [], []
        for _ in range(100):
            returns.extend([0.] * 35 + [.01, .01, .01, .10] + [0.] * 11)
            volume.extend([100.] * 35 + [300.] * 4 + [100.] * 11)
        prices = pd.DataFrame({"Close": 100 * np.cumprod(1 + np.array(returns)), "Volume": volume},
                              index=xcals.get_calendar("XNYS", start="2000-01-01", end="2021-01-01").sessions[:len(returns)])
        prices = prices.iloc[:4938]  # Ends on a precursor, before the next spike.
        row = evaluate_checklist(prices, now=prices.index[-1])
        self.assertTrue(row["validation"]["supported"])
        self.assertEqual(row["status"], "watch")
        self.assertFalse(row["ruleApproved"])

    def test_disjoint_trials_and_fold_outcomes_do_not_overlap(self):
        from sp500_ml.spike_checklist import _trial_summary
        target = pd.Series([True] * 20)
        result = _trial_summary(range(20), target)
        self.assertEqual(result["trials"], 4)
        self.assertEqual(result["hits"], 4)

    def test_calendar_freshness_and_missing_recent_sessions(self):
        prices = sample()
        now = prices.index[-1] + pd.Timedelta(hours=23)
        missing = prices.drop(prices.index[-4])
        row = evaluate_checklist(missing, now=now)
        self.assertEqual(row["status"], "incomplete")
        self.assertIn(prices.index[-4].date().isoformat(), row["missingSessions"])
        earlier = evaluate_checklist(prices, now=prices.index[-1].tz_localize("UTC"))
        self.assertLess(earlier["asOf"], prices.index[-1].date().isoformat())

    def test_direct_call_rejects_invalid_closes(self):
        for frame in (pd.DataFrame({"Close": [100., -1.]}, index=pd.bdate_range("2025-01-02", periods=2)),
                      pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([]))):
            with self.assertRaises(ValueError):
                evaluate_checklist(frame)
