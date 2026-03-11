"""
PD probability calibration for HF benchmark alignment.

Fits Platt scaling (LogisticRegression) or isotonic regression on HF validation set
to improve probability alignment with default rate. Use when model is under-calibrated
on HF (e.g. mean pd_prob ~0.09 vs 19% default rate).

Usage:
    from credit_risk.models.pd_calibration import fit_pd_calibrator, CalibratedPDWrapper

    calibrator = fit_pd_calibrator(pd_probs, y_true, method="platt")
    wrapped = CalibratedPDWrapper(base_model, calibrator)
    pd_prob = wrapped.predict_proba(X)[:, 1]
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def fit_pd_calibrator(
    pd_probs: np.ndarray,
    y_true: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
) -> LogisticRegression | IsotonicRegression:
    """
    Fit calibrator on (pd_prob, y_true), e.g. from HF validation set.

    Args:
        pd_probs: Raw model probabilities (n_samples,)
        y_true: Binary labels (n_samples,)
        method: "platt" (LogisticRegression) or "isotonic"

    Returns:
        Fitted calibrator with predict_proba (platt) or predict (isotonic)
    """
    X = np.asarray(pd_probs).reshape(-1, 1)
    y = np.asarray(y_true).ravel()
    if method == "platt":
        cal = LogisticRegression(max_iter=500, random_state=42)
        cal.fit(X, y)
        return cal
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(X.ravel(), y)
        return cal
    raise ValueError(f"method must be 'platt' or 'isotonic', got {method!r}")


def apply_calibration(
    pd_probs: np.ndarray,
    calibrator: LogisticRegression | IsotonicRegression,
) -> np.ndarray:
    """Apply calibrator to raw probabilities."""
    X = np.asarray(pd_probs).reshape(-1, 1)
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(X)[:, 1]
    return calibrator.predict(X.ravel())


class CalibratedPDWrapper:
    """Wraps a PD model and applies calibration to its probabilities."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        p_raw = self.base_model.predict_proba(X)[:, 1]
        p_cal = apply_calibration(p_raw, self.calibrator)
        return np.column_stack([1 - p_cal, p_cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
