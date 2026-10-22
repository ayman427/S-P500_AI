from io import BytesIO
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator

from sp500_ml.models import _new_model, _artifact_prefix, _prediction_output_path
from sp500_ml.prediction_store import apply_hybrid_feedback


class HybridEstimatorTests(unittest.TestCase):
    def test_components_blend_and_cloud_serialization_roundtrip(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
        X.loc[::10, "b"] = np.nan
        y = .01*X.a + .003*X.c**2
        model = _new_model("hybrid").fit(X, y)
        ridge, boosting = model.predict_components(X.tail(5))
        np.testing.assert_allclose(model.predict(X.tail(5)), model.ridge_weight_*ridge + model.boosting_weight_*boosting)
        self.assertAlmostEqual(model.ridge_weight_ + model.boosting_weight_, 1)
        self.assertEqual(model.calibration_["purge_gap_sessions"], 5)
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        np.testing.assert_allclose(joblib.load(buffer).predict(X.tail(5)), model.predict(X.tail(5)))
        self.assertEqual(clone(model).get_params(), model.get_params())

    def test_calibration_is_later_than_component_training_with_purge(self):
        fits = []
        class Estimator(BaseEstimator):
            def __init__(self, name): self.name = name
            def fit(self, X, y):
                fits.append((self.name, len(X), X[-1, 0]))
                self.fitted_ = True
                return self
            def predict(self, X): return np.full(len(X), .02 if self.name == "ridge" else -.02)
        model = _new_model("hybrid")
        X = pd.DataFrame({"session": range(100)})
        with patch("sp500_ml.models._new_model", side_effect=lambda name, *args: Estimator(name)):
            model.fit(X, pd.Series(np.full(100, .02)))
        self.assertEqual(fits[:2], [("ridge", 75, 74), ("hist_gradient_boosting", 75, 74)])
        self.assertEqual([row[1] for row in fits[2:]], [100, 100])
        self.assertEqual(model.ridge_weight_, 1.)
        self.assertEqual(model.calibration_["calibration_start_row"], 80)

    def test_missing_and_constant_columns_are_filtered_inside_each_boosting_fit(self):
        X = pd.DataFrame({"moving": np.arange(120, dtype=float), "constant": 1., "unavailable": np.nan})
        model = _new_model("hybrid").fit(X, pd.Series(np.linspace(0., .01, 120)))
        self.assertEqual(model.boosting_.named_steps["variancefilter"].valid_features_, [0])
        self.assertTrue(np.isfinite(model.predict(X.tail(1))).all())

    def test_bad_chronology_targets_and_inputs_are_rejected(self):
        X = pd.DataFrame({"a": np.arange(100, dtype=float)})
        y = pd.Series(np.zeros(100))
        for frame, targets in ((X.iloc[::-1], y), (X, y.iloc[::-1]), (X.iloc[:10], y.iloc[:10]), (X, y.mask(y.index == 0))):
            with self.assertRaises(ValueError): _new_model("hybrid").fit(frame, targets)
        self.assertNotEqual(_artifact_prefix("AAPL", "hybrid"), _artifact_prefix("AAPL", "ridge"))
        self.assertNotEqual(_prediction_output_path("hybrid"), _prediction_output_path("ridge"))


def forecast():
    return {"ticker": "AAPL", "target_key": "open_to_close", "as_of": "2026-10-19", "predicted_return": .02,
            "hybrid": {"ridgeReturn": .01, "boostingReturn": .03, "ridgeWeight": .5, "feedbackSamples": 0, "feedbackThrough": None, "learningRule": "settled_error_v1"}}


def observation(**changes):
    return {"ticker": "AAPL", "targetKey": "open_to_close", "modelHash": "hash", "status": "settled",
            "actualReturn": .01, "entryAt": 1000., "exitAt": 2000., "settledAt": 3000.,
            "hybrid": {"ridgeReturn": .01, "boostingReturn": .03}, **changes}


class HybridFeedbackTests(unittest.TestCase):
    def apply(self, rows, existing=None):
        with patch("sp500_ml.prediction_store.call", return_value={"existing": existing, "observations": rows}) as call:
            result = apply_hybrid_feedback(forecast(), "hash", "private-token")
            self.assertEqual(call.call_args.kwargs["token"], "private-token")
            return result

    def test_realized_component_errors_shift_the_next_blend(self):
        result = self.apply([observation()])
        self.assertGreater(result["hybrid"]["ridgeWeight"], .5)
        self.assertLess(result["predicted_return"], .02)
        self.assertEqual(result["hybrid"]["feedbackSamples"], 1)
        self.assertEqual(result["hybrid"]["feedbackThrough"], 3000.)

    def test_pending_future_other_versions_and_overlapping_results_do_not_leak(self):
        bad = [observation(status="pending"), observation(modelHash="other"), observation(ticker="MSFT"), observation(exitAt=1e20), observation(settledAt=1e20), observation(actualReturn=float("nan"))]
        self.assertEqual(self.apply(bad)["hybrid"]["feedbackSamples"], 0)
        rows = [observation(), observation(entryAt=1500., exitAt=2500.), observation(entryAt=3000., exitAt=4000.)]
        self.assertEqual(self.apply(rows)["hybrid"]["feedbackSamples"], 2)

    def test_saved_forecast_is_frozen_and_no_feedback_is_initial_calibration(self):
        original = forecast()
        original["hybrid"]["ridgeWeight"] = .7
        existing = {"signalDate": original["as_of"], "predictedReturn": .016, "hybrid": original["hybrid"]}
        result = self.apply([observation()], existing)
        self.assertEqual(result["predicted_return"], .016)
        self.assertEqual(result["hybrid"]["ridgeWeight"], .7)
        self.assertEqual(self.apply([]), forecast())

    def test_feedback_history_is_bounded_and_authentication_is_required(self):
        rows = [observation(entryAt=i*10000., exitAt=i*10000.+5000.) for i in range(100)]
        self.assertEqual(self.apply(rows)["hybrid"]["feedbackSamples"], 60)
        with self.assertRaises(Exception): apply_hybrid_feedback(forecast(), "hash", None)
