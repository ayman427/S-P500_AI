import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from sp500_ml import risk_model as risk
from sp500_ml.sessions import _calendar


class RiskDatasetTests(unittest.TestCase):
    def prices(self):
        days = _calendar(2024).sessions_in_range("2024-01-02", "2024-10-01")
        values = np.full(len(days), 100.)
        values[80:] = 120.
        values[85:] = 108.
        return pd.DataFrame({"Close": values, "Volume": 1000.}, index=days)

    def test_target_is_any_ten_percent_close_decline_and_missing_futures_are_not_negatives(self):
        prices = self.prices()
        frame = risk.price_frame(prices)
        self.assertEqual(frame.iloc[80].outcome, 1)
        self.assertTrue(frame.iloc[-20:].outcome.isna().all())
        prices.iloc[85:, prices.columns.get_loc("Close")] = 115.
        self.assertEqual(risk.price_frame(prices).iloc[80].outcome, 0)
        prices = prices.drop(prices.index[84])
        self.assertTrue(pd.isna(risk.price_frame(prices).iloc[80].outcome))

    def test_feature_values_cannot_see_future_prices(self):
        prices = self.prices()
        before = risk.price_frame(prices).iloc[80][risk.PRICE_FEATURES]
        prices.iloc[81:, prices.columns.get_loc("Close")] = 1.
        after = risk.price_frame(prices).iloc[80][risk.PRICE_FEATURES]
        pd.testing.assert_series_equal(before, after)

    def test_evidence_needs_observed_version_before_cutoff_not_just_an_old_publication_date(self):
        row = {"title": "Apple shares surge after US president said tariffs will fall", "excerpt": "Apple shares surged.",
               "published_at_utc": "2024-05-03T19:00:00Z", "retrievedAt": "2024-05-03T20:20:00Z", "url": "https://example.com/report", "source": "Finnhub"}
        cutoff = pd.Timestamp("2024-05-03T20:30Z")
        valid = risk.evidence_features({"headlines": [row, row]}, cutoff, ["AAPL", "Apple"])
        self.assertEqual(valid["dated_sources"], 1)
        self.assertEqual(valid["public_statements"], 1)
        for extra in ({"retrievedAt": "2026-01-01T12:00:00Z"}, {"published_at_utc": "2024-05-04T12:00:00Z"},
                      {"publicationPrecision": "date"}, {"source": "Gemini"}):
            self.assertEqual(risk.evidence_features({"headlines": [{**row, **extra}]}, cutoff, ["Apple"])["dated_sources"], 0)
        self.assertEqual(risk.evidence_features({"headlines": [row]}, cutoff, ["NVIDIA"])["dated_sources"], 0)

    def test_training_events_have_nonoverlapping_outcomes(self):
        prices = self.prices()
        prices.iloc[90:, prices.columns.get_loc("Close")] *= 1.5
        self.assertTrue(risk.price_frame(prices).iloc[90].is_rise)
        rows = risk.event_rows("AAA", prices)
        self.assertEqual(len(rows), 1)


class RiskTrainingTests(unittest.TestCase):
    def rows(self):
        rng = np.random.default_rng(91)
        dates = pd.bdate_range("2010-01-01", periods=1000)
        rows = pd.DataFrame({name: rng.normal(size=1000) for name in risk.PRICE_FEATURES})
        rows["date"], rows["label_end"], rows["ticker"] = dates, dates + pd.Timedelta(days=30), "AAA"
        rows["outcome"] = (rows.rise_return + rows.volatility_20 > .4).astype(int)
        for name in risk.EVIDENCE_FEATURES:
            rows[name] = 0.
        return rows

    def test_splits_purge_all_training_and_calibration_outcomes_across_date_boundaries(self):
        train, cal, test = risk.chronological_splits(self.rows())
        self.assertLess(train.label_end.max(), cal.date.min())
        self.assertLess(cal.label_end.max(), test.date.min())
        self.assertLess(train.date.max(), cal.date.min())

    def test_training_calibrates_probabilities_reports_real_baseline_and_gates_evidence(self):
        model, report, held = risk.train_dataset(self.rows())
        self.assertEqual(report["status"], "experimental_validated")
        self.assertLess(report["priceValidation"]["brier"], report["priceValidation"]["constantBrier"])
        self.assertEqual(report["combinedStatus"], "insufficient_point_in_time_evidence")
        self.assertIsNone(model["combined"])
        self.assertTrue(held.probability.between(0, 1).all())
        self.assertEqual(len(held), report["priceValidation"]["samples"])

    def test_evidence_model_is_not_enabled_without_improvement_on_the_same_holdout(self):
        rows = self.rows()
        rows["dated_sources"] = 1.
        model, report, _ = risk.train_dataset(rows)
        self.assertIn("combinedValidation", report)
        self.assertEqual(report["combinedStatus"], "validation_failed")
        self.assertIsNone(model["combined"])

    def test_scoring_uses_a_verified_out_of_sample_model_and_crops_later_prices(self):
        model, report, _ = risk.train_dataset(self.rows())
        model.update(run_id="test-generation", companyNames={"AAA": "Example Inc."})
        selected = self.rows().iloc[[-1]].copy()
        selected["date"] = pd.Timestamp("2024-05-03")
        history = [{"date": "2024-05-02", "close": 100., "volume": 1000.}, {"date": "2024-05-03", "close": 120., "volume": 1000.},
                   {"date": "2024-05-06", "close": 1., "volume": 1000.}]
        market = pd.DataFrame({"Close": [100., 101.]}, index=pd.to_datetime(["2024-05-02", "2024-05-03"]))
        with tempfile.TemporaryDirectory() as directory, patch.object(risk, "NEWS_DIR", Path(directory)), \
                patch.object(risk, "resolve_generation", return_value=[Mock(read_bytes=lambda: b"fixture")]), \
                patch.object(risk.joblib, "load", return_value=model), patch.object(risk, "read_csv", return_value=market), \
                patch.object(risk, "event_rows", return_value=selected) as events:
            result = risk.risk_score("AAA", "2024-05-03", history)
        self.assertEqual(result["status"], "experimental")
        self.assertEqual(result["featureMode"], "price_only")
        self.assertTrue(0 <= result["probability"] <= 1)
        self.assertEqual(result["modelVersion"], "test-generation")
        self.assertEqual(events.call_args.args[1].index.max(), pd.Timestamp("2024-05-03"))

    def test_insufficient_support_and_duplicate_labels_do_not_produce_a_model(self):
        for rows in (self.rows().iloc[:50], pd.concat([self.rows(), self.rows().iloc[:1]])):
            with self.assertRaises(ValueError):
                risk.train_dataset(rows)

    def test_in_sample_and_unvalidated_models_never_return_a_probability(self):
        for report, status in [({"status": "validation_failed"}, "validation_failed"),
                               ({"status": "experimental_validated", "trainedThrough": "2025-01-01"}, "historical_in_sample")]:
            with patch.object(risk, "resolve_generation", return_value=[Mock(read_bytes=lambda: b"fixture")]), \
                    patch.object(risk.joblib, "load", return_value={"version": risk.PREFIX, "target": risk.TARGET, "report": report}):
                result = risk.risk_score("AAA", "2024-05-03", [])
                self.assertIsNone(result["probability"])
                self.assertEqual(result["status"], status)
