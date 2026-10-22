"""Experimental chronological blend of Ridge and histogram gradient boosting."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted, validate_data

class HybridRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, ridge_alpha=10.0, purge_gap=5):
        self.ridge_alpha = ridge_alpha
        self.purge_gap = purge_gap

    def fit(self, X, y):
        from .models import _new_model
        from .backtest import VarianceFilter
        def new_boosting():
            return make_pipeline(VarianceFilter(min_unique=2), _new_model("hist_gradient_boosting"))
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0 or not isinstance(self.purge_gap, int) or self.purge_gap < 5:
            raise ValueError("Hybrid requires a positive Ridge alpha and a purge gap of at least five sessions.")
        if hasattr(X, "index"):
            if not X.index.is_monotonic_increasing or X.index.has_duplicates:
                raise ValueError("Hybrid training rows must be unique and chronologically ordered.")
            if hasattr(y, "index") and not X.index.equals(y.index):
                raise ValueError("Hybrid features and targets must have matching dates.")
        values, target = validate_data(self, X, y, ensure_all_finite="allow-nan", y_numeric=True)
        if not np.isfinite(target).all() or len(target) < 60:
            raise ValueError("Hybrid needs at least 60 finite completed training outcomes.")
        calibration_rows = max(20, len(target) // 5)
        calibration_start = len(target) - calibration_rows
        fit_end = calibration_start - self.purge_gap
        ridge = _new_model("ridge", self.ridge_alpha).fit(values[:fit_end], target[:fit_end])
        boosting = new_boosting().fit(values[:fit_end], target[:fit_end])
        ridge_predictions = ridge.predict(values[calibration_start:])
        boosting_predictions = boosting.predict(values[calibration_start:])
        # Calibration outcomes choose only the blend, never evaluate the final test window.
        weights = np.linspace(0., 1., 11)
        errors = [mean_absolute_error(target[calibration_start:], weight * ridge_predictions + (1 - weight) * boosting_predictions) for weight in weights]
        best = min(range(len(weights)), key=lambda index: (errors[index], abs(weights[index] - .5)))
        self.ridge_weight_ = float(weights[best])
        self.boosting_weight_ = 1. - self.ridge_weight_
        self.calibration_ = {"training_rows": fit_end, "calibration_start_row": calibration_start,
                             "calibration_rows": calibration_rows, "purge_gap_sessions": self.purge_gap,
                             "ridge_weight": self.ridge_weight_, "boosting_weight": self.boosting_weight_,
                             "blend_mae": float(errors[best]),
                             "ridge_mae": float(mean_absolute_error(target[calibration_start:], ridge_predictions)),
                             "boosting_mae": float(mean_absolute_error(target[calibration_start:], boosting_predictions))}
        self.ridge_ = _new_model("ridge", self.ridge_alpha).fit(values, target)
        self.boosting_ = new_boosting().fit(values, target)
        return self

    def predict(self, X):
        ridge, boosting = self.predict_components(X)
        return self.ridge_weight_ * ridge + self.boosting_weight_ * boosting

    def predict_components(self, X):
        check_is_fitted(self, ["ridge_", "boosting_", "ridge_weight_"])
        values = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return self.ridge_.predict(values), self.boosting_.predict(values)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
