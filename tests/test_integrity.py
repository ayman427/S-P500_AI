import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import dashboard_server
from sp500_ml.artifacts import atomic_csv, exclusive_file, publish_generation, resolve_generation
from sp500_ml.models import _model_artifact_paths
from sp500_ml.paper import _paper_ledger_path, _paper_summary, _settle_paper_rows, _validate_ledger_identity, paper_trade
from sp500_ml.sessions import execution_window, validate_signal_time


def pending():
    return pd.DataFrame([{
        "ticker": "AAA", "model_type": "ridge", "model_sha256": "fingerprint",
        "signal_date": "2026-01-05", "created_at_utc": "2026-01-05T22:00:00Z",
        "expected_execution_date": "2026-01-06", "predicted_return": .02,
        "entry_threshold": 0., "transaction_cost_bps_per_side": 10.,
        "signal_edge_after_costs": .018, "enter_long": True, "status": "pending",
        "next_open": np.nan, "next_close": np.nan, "net_return_if_traded": np.nan,
        "strategy_daily_return": np.nan,
    }])


def settled():
    prices = pd.DataFrame({"Open": [100.], "Close": [102.]}, index=pd.to_datetime(["2026-01-06"]))
    return _settle_paper_rows(pending(), prices)


class PaperIntegrityTests(unittest.TestCase):
    def test_missing_execution_bar_never_uses_a_later_day(self):
        prices = pd.DataFrame({"Open": [100.], "Close": [120.]}, index=pd.to_datetime(["2026-01-07"]))
        with self.assertRaisesRegex(ValueError, "Missing execution-session"):
            _settle_paper_rows(pending(), prices)

    def test_missing_execution_bar_stays_pending_until_available(self):
        prices = pd.DataFrame({"Open": [100.], "Close": [102.]}, index=pd.to_datetime(["2026-01-05"]))
        self.assertEqual(_settle_paper_rows(pending(), prices).iloc[0]["status"], "pending")

    def test_exchange_holidays_and_early_close_are_respected(self):
        self.assertEqual(str(execution_window("2026-01-16")[2].date()), "2026-01-20")
        self.assertEqual(str(execution_window("2026-04-02")[2].date()), "2026-04-06")
        validate_signal_time("2026-11-27", "2026-11-27T18:30:00Z")

    def test_signals_at_open_after_open_before_close_or_without_timezone_are_rejected(self):
        for timestamp in ["2026-01-06T14:30:00Z", "2026-01-06T15:00:00Z", "2026-01-05T20:59:59Z", "2026-01-05T22:00:00"]:
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                validate_signal_time("2026-01-05", timestamp)

    def test_summary_rejects_the_reproduced_contradictory_profit(self):
        ledger = settled()
        ledger.loc[0, "net_return_if_traded"] = -.1
        ledger.loc[0, "strategy_daily_return"] = .5
        with self.assertRaisesRegex(ValueError, "contradicts"):
            _paper_summary(ledger)

    def test_summary_reconciles_net_return_against_prices(self):
        ledger = settled()
        ledger.loc[0, ["net_return_if_traded", "strategy_daily_return"]] = .5
        with self.assertRaisesRegex(ValueError, "execution prices"):
            _paper_summary(ledger)

    def test_duplicate_dates_invalid_decisions_and_nan_returns_are_rejected(self):
        ledger = settled()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _paper_summary(pd.concat([ledger, ledger], ignore_index=True))
        for field, value in [("enter_long", "maybe"), ("net_return_if_traded", np.nan)]:
            broken = ledger.copy()
            broken[field] = broken[field].astype(object)
            broken.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _paper_summary(broken)

    def test_selected_model_identity_timing_and_decision_are_required(self):
        _validate_ledger_identity(settled(), "AAA", "ridge", "fingerprint")
        for field, value in [("model_type", "hist_gradient_boosting"), ("model_sha256", "another"), ("created_at_utc", "2026-01-06T15:00:00Z"), ("execution_date", "2026-01-07"), ("enter_long", False)]:
            ledger = settled()
            ledger.loc[0, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_ledger_identity(ledger, "AAA", "ridge", "fingerprint")

    def test_absent_model_never_resolves_a_legacy_ledger(self):
        with tempfile.TemporaryDirectory() as directory, patch("sp500_ml.paper.PAPER_DIR", Path(directory)), patch("sp500_ml.paper._model_artifact_paths", return_value=(Path(directory) / "missing", None, None)):
            (Path(directory) / "AAA.csv").write_text("legacy")
            with self.assertRaisesRegex(RuntimeError, "No trained ridge"):
                _paper_ledger_path("AAA", model_type="ridge", schema_version=3)

    def test_paper_creation_is_timestamped_and_late_creation_writes_nothing(self):
        prediction = {"ticker": "AAA", "as_of": "2026-01-05", "last_close": 100., "predicted_return": .02}
        for created, allowed in [("2026-01-05T22:00:00Z", True), ("2026-01-06T14:30:00Z", False)]:
            with self.subTest(created=created), tempfile.TemporaryDirectory() as directory, patch("sp500_ml.paper.PAPER_DIR", Path(directory)), patch("sp500_ml.paper.ROOT", Path(directory)), patch("sp500_ml.paper._load_latest_prediction", return_value=(prediction, "fingerprint")), patch("sp500_ml.paper._fetch_recent_news", return_value=[]), patch("sp500_ml.paper._save_news_cache"), patch("sp500_ml.paper.utc_now", return_value=pd.Timestamp(created)):
                if allowed:
                    paper_trade("AAA", model_type="ridge")
                    ledger = pd.read_csv(next(Path(directory).glob("*.csv")))
                    self.assertEqual(ledger.loc[0, "expected_execution_date"], "2026-01-06")
                    self.assertEqual(pd.Timestamp(ledger.loc[0, "created_at_utc"]), pd.Timestamp(created))
                else:
                    with self.assertRaisesRegex(ValueError, "before the next session opens"):
                        paper_trade("AAA", model_type="ridge")
                    self.assertEqual(list(Path(directory).glob("*.csv")), [])

    def test_atomic_ledger_write_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.csv"
            path.write_text("original")
            with patch("sp500_ml.artifacts.os.replace", side_effect=OSError("failure")), self.assertRaises(OSError):
                atomic_csv(pd.DataFrame({"x": [1]}), path)
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class ArtifactIntegrityTests(unittest.TestCase):
    def publish(self, root):
        return publish_generation(root, "AAA_ridge_open_to_close", {"model": "dummy"}, {"ticker": "AAA"}, {"sessions": 252}, pd.DataFrame({"return": [.01]}))

    def test_complete_generation_is_selected_and_all_files_share_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, metrics, holdout, trades = self.publish(root)
            with patch("sp500_ml.models.MODEL_DIR", root):
                self.assertEqual(_model_artifact_paths("AAA", "ridge"), (model, metrics, trades))
            self.assertEqual(joblib.load(model)["run_id"], json.loads(metrics.read_text())["run_id"])
            self.assertEqual(json.loads(metrics.read_text())["run_id"], json.loads(holdout.read_text())["run_id"])

    def test_corrupted_report_fails_instead_of_falling_back_to_old_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, metrics, _, _ = self.publish(root)
            metrics.write_text('{"ticker":"FAKE"}')
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                resolve_generation(root, "AAA_ridge_open_to_close")

    def test_interrupted_publication_keeps_the_previous_complete_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = self.publish(root)
            manifest = root / "AAA_ridge_open_to_close_manifest.json"
            original = manifest.read_bytes()
            with patch("sp500_ml.artifacts.os.replace", side_effect=OSError("interrupted")), self.assertRaises(OSError):
                self.publish(root)
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(resolve_generation(root, "AAA_ridge_open_to_close"), (previous[0], previous[1], previous[3]))

    def test_competing_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with exclusive_file(root / ".AAA_ridge_open_to_close.publish.lock"):
                with self.assertRaisesRegex(RuntimeError, "Another operation"):
                    self.publish(root)
            self.assertIsNone(resolve_generation(root, "AAA_ridge_open_to_close"))

    def test_dashboard_does_not_display_a_legacy_ledger_when_model_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = root / "AAA.csv"
            pd.DataFrame({"Close": [100., 101.], "Volume": [100, 100]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"])).to_csv(prices, index_label="Date")
            with patch.object(dashboard_server.sp500_ml, "_ticker_file", return_value=prices), patch.object(dashboard_server.sp500_ml, "_model_artifact_paths", return_value=(root / "missing", root / "metrics.json", root / "trades.csv")), patch.object(dashboard_server.sp500_ml, "_paper_ledger_path") as paper_path, patch.object(dashboard_server.sp500_ml, "_price_files", return_value=[prices]), patch.object(dashboard_server.sp500_ml, "NEWS_DIR", root), patch.object(dashboard_server.sp500_ml, "MODEL_DIR", root):
                state = dashboard_server._read_dashboard_state("AAA", model_type="ridge")
                self.assertIsNone(state["paper"])
                self.assertEqual(state["paperRows"], [])
                paper_path.assert_not_called()


class DashboardIntegrityTests(unittest.TestCase):
    def test_corrupt_paper_totals_are_withheld_without_losing_model_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = root / "AAA.csv"
            pd.DataFrame({"Close": [100., 101.], "Volume": [100, 100]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"])).to_csv(prices, index_label="Date")
            prefix = "AAA_ridge_open_to_close"
            model, _, _, _ = publish_generation(root, prefix, {"model": "dummy"}, {"ticker": "AAA", "model_type": "ridge", "target_key": "open_to_close", "target": "next-session open-to-close return"}, {"sessions": 252}, pd.DataFrame({"return": [.01]}))
            ledger = settled()
            fingerprint = hashlib.sha256(model.read_bytes()).hexdigest()
            ledger["model_sha256"] = fingerprint
            ledger["strategy_daily_return"] = .5
            ledger.to_csv(root / f"AAA_ridge_{fingerprint[:10]}_v3.csv", index=False)
            with patch("sp500_ml.models.MODEL_DIR", root), patch("sp500_ml.paper.PAPER_DIR", root), patch.object(dashboard_server.sp500_ml, "_ticker_file", return_value=prices), patch.object(dashboard_server.sp500_ml, "_load_latest_prediction", return_value=({"predicted_return": .02}, fingerprint)), patch("dashboard_server._read_model_inspection", return_value={}), patch.object(dashboard_server.sp500_ml, "_price_files", return_value=[prices]), patch.object(dashboard_server.sp500_ml, "NEWS_DIR", root), patch.object(dashboard_server.sp500_ml, "MODEL_DIR", root):
                state = dashboard_server._read_dashboard_state("AAA", model_type="ridge")
                self.assertEqual(state["artifactIntegrity"], "verified")
                self.assertEqual(state["metrics"]["ticker"], "AAA")
                self.assertIsNone(state["holdout"])
                self.assertIsNone(state["paper"])
                self.assertEqual(state["paperRows"], [])
                self.assertIn("contradicts", state["paperError"])


class TrainingPublicationTests(unittest.TestCase):
    def test_training_recovers_a_bad_manifest_and_publishes_one_complete_run(self):
        from sklearn.dummy import DummyRegressor
        from sp500_ml.training import train_model

        dates = pd.bdate_range("2023-01-03", periods=400)
        features = pd.DataFrame({"x": np.arange(400, dtype=float), "next_open_to_close_return": np.sin(np.arange(400)) / 100, "next_date": pd.Series(dates, index=dates).shift(-1)}, index=dates)
        features.loc[dates[-1], "next_open_to_close_return"] = np.nan
        estimator = DummyRegressor().fit(features[["x"]].iloc[:-1], features["next_open_to_close_return"].iloc[:-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = root / "prices.csv"
            pd.DataFrame({"Close": [100.]}, index=pd.to_datetime(["2023-01-03"])).to_csv(prices, index_label="Date")
            (root / "AAA_ridge_open_to_close_manifest.json").write_text("incomplete")
            with patch("sp500_ml.models.MODEL_DIR", root), patch("sp500_ml.training.MODEL_DIR", root), patch("sp500_ml.training.BACKTEST_DIR", root / "backtests"), patch("sp500_ml.training.ROOT", root), patch("sp500_ml.training._ticker_file", return_value=prices), patch("sp500_ml.training._load_market_data", return_value=pd.DataFrame()), patch("sp500_ml.training._load_training_fundamentals", return_value=(None, None)), patch("sp500_ml.training.build_features", return_value=features), patch("sp500_ml.training._available_features", return_value=["x"]), patch("sp500_ml.training._walk_forward_metrics", return_value={"validation_rows": 100}), patch("sp500_ml.training._final_holdout_metrics", return_value=({"sessions": 252}, pd.DataFrame({"return": [.01]}))), patch("sp500_ml.training._fit_tuned_model", return_value=(estimator, {})):
                train_model("AAA", model_type="ridge")
                model, metrics, _ = _model_artifact_paths("AAA", "ridge")
                bundle = joblib.load(model)
                report = json.loads(metrics.read_text())
                self.assertEqual(bundle["run_id"], report["run_id"])
                self.assertEqual(bundle["training_rows"], 399)
                self.assertEqual(bundle["trained_through"], dates[-2].date().isoformat())
                self.assertEqual(bundle["entry_threshold"], 0.)
                self.assertEqual(report["development_walk_forward"]["validation_rows"], 100)


class ForecastAvailabilityTests(unittest.TestCase):
    def test_retraining_is_distinguished_from_missing_inputs_and_inspection_failure(self):
        from sp500_ml.models import ModelRetrainingRequired
        cases = [
            (ModelRetrainingRequired("older target"), True, False),
            (RuntimeError("SEC fundamentals unavailable"), False, False),
            (None, False, True),
        ]
        for error, needs_retraining, ready in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prices = root / "AAA.csv"
                pd.DataFrame({"Close": [100., 101.], "Volume": [100, 100]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"])).to_csv(prices, index_label="Date")
                model = root / "model.joblib"
                model.write_bytes(b"model")
                metrics = root / "metrics.json"
                metrics.write_text(json.dumps({"ticker": "AAA", "model_type": "ridge", "target_key": "open_to_close", "target": "next-session open-to-close return"}))
                with patch.object(dashboard_server.sp500_ml, "_ticker_file", return_value=prices), patch.object(dashboard_server.sp500_ml, "_model_artifact_paths", return_value=(model, metrics, root / "trades.csv")), patch.object(dashboard_server.sp500_ml, "_load_latest_prediction", side_effect=error, return_value=({"predicted_return": .01}, "fingerprint")), patch("dashboard_server._read_model_inspection", side_effect=ValueError("inspection failed")), patch.object(dashboard_server.sp500_ml, "_paper_ledger_path", return_value=root / "no-ledger.csv"), patch.object(dashboard_server.sp500_ml, "_price_files", return_value=[prices]), patch.object(dashboard_server.sp500_ml, "NEWS_DIR", root), patch.object(dashboard_server.sp500_ml, "MODEL_DIR", root):
                    state = dashboard_server._read_dashboard_state("AAA", "ridge")
                    self.assertEqual(state["forecastNeedsRetraining"], needs_retraining)
                    self.assertEqual(state["modelReady"], ready)
                    if ready:
                        self.assertIsNone(state["forecastError"])
                        self.assertEqual(state["modelInspectionError"], "inspection failed")
                    else:
                        self.assertEqual(state["forecastError"], str(error))


class ShortHistoryTests(unittest.TestCase):
    def test_training_rejects_short_history_before_fitting(self):
        from sp500_ml.training import train_model
        dates = pd.bdate_range("2026-01-05", periods=100)
        features = pd.DataFrame({"next_open_to_close_return": .01}, index=dates)
        with tempfile.TemporaryDirectory() as directory:
            prices = Path(directory) / "prices.csv"
            pd.DataFrame({"Close": [100.]}, index=pd.to_datetime(["2026-01-05"])).to_csv(prices, index_label="Date")
            with patch("sp500_ml.paper._paper_model_freeze_status", return_value=None), patch("sp500_ml.training._ticker_file", return_value=prices), patch("sp500_ml.training._load_market_data", return_value=pd.DataFrame()), patch("sp500_ml.training._load_training_fundamentals", return_value=(None, None)), patch("sp500_ml.training.build_features", return_value=features), patch("sp500_ml.training._fit_tuned_model") as fit:
                with self.assertRaisesRegex(RuntimeError, "Need more than 352 labeled sessions"):
                    train_model("AAA")
                fit.assert_not_called()


class StockDirectoryTests(unittest.TestCase):
    def test_directory_uses_cached_names_and_only_local_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fundamentals = root / "fundamentals"
            fundamentals.mkdir()
            prices = root / "prices"
            prices.mkdir()
            (fundamentals / "company_tickers.json").write_text(json.dumps({"0": {"ticker": "AAPL", "title": "Apple Inc."}, "1": {"ticker": "NVDA", "title": "NVIDIA CORP"}, "2": {"ticker": "BRK.B", "title": "Berkshire Hathaway Inc."}}))
            (root / "companies.json").write_text(json.dumps({"NVDA": "Nvidia"}))
            with patch.object(dashboard_server.sp500_ml, "FUNDAMENTAL_DIR", fundamentals), patch.object(dashboard_server.sp500_ml, "DATA_DIR", prices), patch.object(dashboard_server.sp500_ml, "_price_files", return_value=[prices / "AAPL.csv", prices / "BRK-B.csv", prices / "NVDA.csv"]):
                self.assertEqual(dashboard_server._read_stock_directory()["stocks"], [{"ticker": "AAPL", "name": "Apple Inc."}, {"ticker": "BRK-B", "name": "Berkshire Hathaway Inc."}, {"ticker": "NVDA", "name": "Nvidia"}])

    def test_price_update_caches_company_names_and_normalizes_symbols(self):
        from sp500_ml.data import get_sp500_tickers
        from unittest.mock import Mock
        table = pd.DataFrame({"Symbol": ["AAPL", "BRK.B"], "Security": ["Apple Inc.", "Berkshire Hathaway"]})
        with tempfile.TemporaryDirectory() as directory, patch("sp500_ml.data.DATA_DIR", Path(directory) / "prices"), patch("sp500_ml.data.requests.get", return_value=Mock(text="table")), patch("sp500_ml.data.pd.read_html", return_value=[table]):
            self.assertEqual(get_sp500_tickers(), ["AAPL", "BRK-B"])
            self.assertEqual(json.loads((Path(directory) / "companies.json").read_text()), {"AAPL": "Apple Inc.", "BRK-B": "Berkshire Hathaway"})

    def test_serverless_api_exports_stock_directory(self):
        from vercel_app import app
        self.assertIn("/api/stocks", {route.path for route in app.routes})
