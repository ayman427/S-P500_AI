import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sp500_ml.prediction_store import build_prediction, reconcile_once, track
from sp500_ml import prediction_store


class PredictionStoreTests(unittest.TestCase):
    def test_personal_history_forwards_verified_token_and_cursor_without_owner_input(self):
        result = {"page": [{"_id": "mine"}], "isDone": False, "continueCursor": "next"}
        with patch.object(prediction_store, "call", return_value=result) as call:
            page = prediction_store.my_predictions("verified-token", "previous")
            call.assert_called_once_with("query", "mine", {"paginationOpts": {"numItems": 20, "cursor": "previous"}}, token="verified-token")
            self.assertEqual(page, {"rows": [{"_id": "mine"}], "hasMore": True, "cursor": "next"})
        with self.assertRaises(ValueError): prediction_store.my_predictions("verified-token", "x" * 8193)
        from sp500_ml.auth import AuthenticationError
        with self.assertRaises(AuthenticationError): prediction_store.my_predictions(None)

    def test_existing_prediction_still_confirmed_outside_recording_window(self):
        import hashlib
        import json
        record = self.record()
        key = hashlib.sha256(json.dumps(["AAPL", "ridge", "hash", "open_to_close", record["as_of"]]).encode()).hexdigest()
        with patch.object(prediction_store, "configured", return_value=True), patch("sp500_ml.training._load_latest_prediction", return_value=(record, "hash")), patch.object(prediction_store, "build_prediction", side_effect=ValueError("Window closed")), patch.object(prediction_store, "call", return_value=[{"_id": "existing", "key": key, "predictedReturn": record["predicted_return"]}]) as call:
            result = track("AAPL", "ridge", "open_to_close", token="token")
            self.assertTrue(result["saved"])
            self.assertFalse(result["created"])
            self.assertEqual(result["recordId"], "existing")
            call.assert_called_once_with("query", "history", {"ticker": "AAPL", "modelType": "ridge", "targetKey": "open_to_close"}, token="token")

    def test_save_confirmation_requires_committed_record_readback(self):
        for created in (True, False):
            with self.subTest(created=created), patch.object(prediction_store, "configured", return_value=True), patch("sp500_ml.training._load_latest_prediction", return_value=({}, "hash")), patch.object(prediction_store, "build_prediction", return_value={"key": "key"}), patch.object(prediction_store, "call", side_effect=[{"id": "id", "created": created}, [{"_id": "id", "key": "key"}]]):
                result = track("AAPL", "ridge", "open_to_close", token="token")
                self.assertTrue(result["saved"])
                self.assertEqual(result["created"], created)
                self.assertEqual(result["recordId"], "id")
        with patch.object(prediction_store, "configured", return_value=True), patch("sp500_ml.training._load_latest_prediction", return_value=({}, "hash")), patch.object(prediction_store, "build_prediction", return_value={"key": "key"}), patch.object(prediction_store, "call", side_effect=[{"id": "id", "created": True}, []]):
            with self.assertRaisesRegex(RuntimeError, "could not be confirmed"):
                track("AAPL", "ridge", "open_to_close", token="token")

    def test_file_configuration_reloads_without_overwriting_process_override(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sp500_ml" / "prediction_store.py"
            source.parent.mkdir()
            env = Path(directory) / ".env.local"
            with patch.object(prediction_store, "__file__", str(source)), patch.dict(os.environ, {}, clear=True), patch.object(prediction_store, "_file_environment", {}):
                env.write_text("CONVEX_URL=http://localhost:3210\nPREDICTION_STORE_SECRET=test\n")
                prediction_store.load_environment()
                env.write_text("CONVEX_URL=https://example.convex.cloud\nPREDICTION_STORE_SECRET=test\n")
                prediction_store.load_environment()
                self.assertEqual(os.environ["CONVEX_URL"], "https://example.convex.cloud")
                os.environ["CONVEX_URL"] = "https://override.convex.cloud"
                prediction_store.load_environment()
                self.assertEqual(os.environ["CONVEX_URL"], "https://override.convex.cloud")

    def record(self, target="open_to_close"):
        return dict(ticker="AAPL", as_of="2026-07-02", target_key=target,
                    predicted_return=0.02, model_trained_through="2026-07-01",
                    model_trained_at_utc="2026-07-02T18:00:00Z")

    def test_holiday_and_five_session_dates(self):
        now = pd.Timestamp("2026-07-02T21:00:00Z")
        one = build_prediction(self.record(), "ridge", "abc", now)
        five = build_prediction(self.record("open_to_close_5d"), "ridge", "abc", now)
        self.assertEqual(one["entryDate"], "2026-07-06")
        self.assertEqual(one["exitDate"], "2026-07-06")
        self.assertEqual(five["exitDate"], "2026-07-10")
        self.assertNotEqual(one["key"], five["key"])

    def test_stale_and_lookahead_rejected(self):
        with self.assertRaises(ValueError):
            build_prediction(self.record(), "ridge", "abc", pd.Timestamp("2026-07-06T15:00:00Z"))
        record = self.record()
        record["model_trained_through"] = "2026-07-07"
        with self.assertRaises(ValueError):
            build_prediction(record, "ridge", "abc", pd.Timestamp("2026-07-02T21:00:00Z"))

    @patch("sp500_ml.prediction_store.configured", return_value=False)
    def test_unconfigured_does_not_claim_saved(self, _):
        self.assertEqual(track("AAPL", "ridge", "open_to_close", token="token")["status"], "unconfigured")

    @patch("sp500_ml.prediction_store.configured", return_value=True)
    @patch("sp500_ml.prediction_store.yf.Ticker")
    @patch("sp500_ml.prediction_store.call")
    def test_missing_exact_bar_stays_pending(self, call, ticker, _):
        call.return_value = [{"_id": "1", "ticker": "AAPL", "entryDate": "2026-07-06", "exitDate": "2026-07-10"}]
        ticker.return_value.history.return_value = pd.DataFrame(
            {"Open": [100], "Close": [102]}, index=pd.to_datetime(["2026-07-09"]))
        self.assertEqual(reconcile_once(), 0)
        call.assert_called_once_with("query", "due", {})

    @patch("sp500_ml.prediction_store.configured", return_value=True)
    @patch("sp500_ml.prediction_store.yf.Ticker")
    @patch("sp500_ml.prediction_store.call")
    def test_settles_using_entry_open_and_final_close(self, call, ticker, _):
        call.side_effect = [[{"_id": "1", "ticker": "AAPL", "entryDate": "2026-07-06", "exitDate": "2026-07-10"}], "1"]
        ticker.return_value.history.return_value = pd.DataFrame(
            {"Open": [100, 105], "Close": [101, 110]}, index=pd.to_datetime(["2026-07-06", "2026-07-10"]))
        self.assertEqual(reconcile_once(), 1)
        self.assertEqual(call.call_args.args[2]["entryOpen"], 100)
        self.assertEqual(call.call_args.args[2]["exitClose"], 110)
