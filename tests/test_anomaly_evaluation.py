import unittest
from sp500_ml.anomaly_evaluation import metrics, select_rows, block_lift, wilson, build_report


def event(day, through, reversal=False, flags=2, sessions=20):
    ids = ["price", "volume", "market", "acceleration"]
    signals = [{"id": key, "status": "elevated" if i == 0 or i <= flags else "within_rule"} for i,key in enumerate(ids)]
    return {"date": day, "return": .1, "close": 110., "previousClose": 100., "analysis": {"signals": signals,
        "historicalOutcome": {"sessions": sessions, "through": through, "largestDecline": -.2 if reversal else -.01, "return20": -.1 if reversal else .05}}}


class EvaluationTests(unittest.TestCase):
    def test_purge_crossing_boundary_and_remove_overlap_without_cherry_picking_labels(self):
        events = [event("2021-11-01", "2021-11-30"), event("2021-11-10", "2021-12-10", True),
                  event("2021-12-15", "2022-01-15", True), event("2022-01-03", "2022-02-01", True),
                  event("2022-03-01", "2022-03-30", sessions=10)]
        rows, exclusions = select_rows("AAPL", events, "2022-01-01")
        self.assertEqual([r["date"] for r in rows], ["2021-11-01", "2022-01-03"])
        self.assertEqual([r["partition"] for r in rows], ["development", "holdout"])
        self.assertEqual(exclusions["overlap"], 1)
        self.assertEqual(exclusions["crossesSplit"], 1)
        self.assertEqual(exclusions["incompleteOutcome"], 1)

    def test_confusion_counts_include_false_alerts_and_missed_reversals(self):
        rows, _ = select_rows("AAPL", [event("2022-01-01", "2022-01-20", True), event("2022-02-01", "2022-02-20"),
            event("2022-03-01", "2022-03-20", True, flags=0), event("2022-04-01", "2022-04-20", flags=0)], "2022-01-01")
        m = metrics(rows, "multi_signal")
        self.assertEqual([m[k] for k in ["truePositives", "falseAlerts", "missedReversals", "trueNegatives"]], [1,1,1,1])
        self.assertEqual(m["precision"], .5)
        self.assertEqual(m["recall"], .5)
        self.assertEqual(metrics(rows,"price_only")["missedReversals"], 0)

    def test_missing_signal_does_not_become_negative(self):
        e = event("2022-01-01", "2022-02-01")
        e["analysis"]["signals"][1]["status"] = "unavailable"
        rows, excluded = select_rows("AAPL", [e], "2022-01-01")
        self.assertEqual(rows, [])
        self.assertEqual(excluded["missingSignals"], 1)

    def test_empty_and_zero_flag_samples_have_no_invented_rates(self):
        self.assertIsNone(metrics([], "multi_signal")["precision"])
        self.assertIsNone(wilson(0,0))
        self.assertIsNone(block_lift([])["interval95"])
        self.assertEqual(build_report([], {}, [], [])["conclusion"], "insufficient_sample")

    def test_block_bootstrap_is_deterministic_and_never_trains_a_model(self):
        rows = []
        for i in range(1, 10):
            year, month = 2020 + i//4, (i%4)*3+1
            e = event(f"{year}-{month:02}-01", f"{year}-{month:02}-28", reversal=i%2 == 0)
            selected, _ = select_rows("AAPL", [e], "2020-01-01")
            rows.extend(selected)
        self.assertEqual(block_lift(rows, 100), block_lift(rows, 100))
        self.assertEqual(block_lift(rows, 100)["interval95"], [0.,0.])
        self.assertFalse(build_report(rows, {}, [], [])["validatedForPrediction"])
