import unittest
from unittest.mock import patch
import json
import pandas as pd
from sp500_ml import anomaly_tracking as t
from tests import test_spike_anomalies as anomaly_tests
from sp500_ml.sessions import execution_window, _calendar


class TrackingTests(unittest.TestCase):
    def fixture(self):
        e,h,m = anomaly_tests.AnomalyTests().fixture()
        close,opening,_ = execution_window(e["date"])
        p = {"hash":"a"*64,"frozenAt":(close-pd.Timedelta(days=2)).timestamp()*1000}
        return e,h,m,p,close+pd.Timedelta(hours=1),opening

    def test_snapshot_has_no_future_inputs_and_outcomes_are_not_recorded(self):
        e,h,m,p,now,_ = self.fixture()
        r = t.capture_record("AAPL",e,h,m,p,now)
        self.assertIsNotNone(r)
        snapshot = json.loads(r["inputSnapshot"])
        self.assertTrue(all(row["date"] <= e["date"] for row in snapshot["stock"]+snapshot["market"]))
        self.assertEqual(t.digest(r["inputSnapshot"]),r["inputSha256"])
        self.assertNotIn("historicalOutcome",r)
        self.assertTrue(r["flagged"])
        self.assertEqual(len(_calendar(now.year).sessions_in_range(e["date"],r["dueDate"])),21)
        changed = [{**row,"close":1.} if row["date"] > e["date"] else row for row in h]
        self.assertEqual(r,t.capture_record("AAPL",e,changed,m,p,now))

    def test_late_and_pre_registration_events_cannot_be_backfilled(self):
        e,h,m,p,now,opening = self.fixture()
        self.assertIsNone(t.capture_record("AAPL",e,h,m,p,opening))
        self.assertIsNone(t.capture_record("AAPL",e,h,m,{**p,"frozenAt":now.timestamp()*1000},now))

    def test_missing_market_is_unknown_not_a_negative_flag(self):
        e,h,_,p,now,_ = self.fixture()
        r = t.capture_record("AAPL",e,h,{},p,now)
        self.assertIsNone(r["flagged"])
        self.assertEqual(r["signals"][1]["status"],"unavailable")

    def test_outcomes_wait_for_full_exchange_window_and_preserve_adjusted_scale(self):
        e,h,m,p,now,_ = self.fixture()
        r = t.capture_record("AAPL",e,h,m,p,now)
        frame = pd.DataFrame(h).set_index("date").rename(columns={"close":"Close"})
        frame.index = pd.to_datetime(frame.index)
        self.assertIsNone(t.outcome_snapshot(r,frame,now))
        due = pd.Timestamp(r["dueAt"],unit="ms",tz="UTC")
        text = t.outcome_snapshot(r,frame,due)
        self.assertEqual(len(json.loads(text)),21)
        scaled = frame.copy(); scaled["Close"] /= 4
        rows = json.loads(t.outcome_snapshot(r,scaled,due))
        original = json.loads(text)
        self.assertAlmostEqual(rows[-1]["close"]/rows[0]["close"],original[-1]["close"]/original[0]["close"])
        broken = frame.drop(pd.Timestamp(r["dueDate"]))
        self.assertIsNone(t.outcome_snapshot(r,broken,due))

    def test_frozen_protocol_mismatch_stops_capture(self):
        with patch.object(t,"rpc",return_value={"hash":"wrong"}), self.assertRaises(RuntimeError):
            t.run_once()
