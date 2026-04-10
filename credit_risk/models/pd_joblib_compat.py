"""Pickle / sklearn compatibility for notebook-trained PD models.

``joblib`` restores wrapper instances with the class object from the training
session (e.g. ``__main__._StackedPDWrapper``). Rebind them to the module classes
so sklearn 1.6+ tag checks (``__sklearn_tags__``) work with
``FrozenEstimator`` / ``CalibratedClassifierCV``.
"""

from __future__ import annotations

import numpy as np


def rebind_sklearn_pd_wrappers(obj) -> None:
    """Reattach pickled notebook wrappers to current ``pd_model`` classes."""
    from credit_risk.models import pd_model as _pm

    _StackedPDWrapper = _pm._StackedPDWrapper
    _LRWithScaler = _pm._LRWithScaler
    # Added in newer 02a calibration fallback; older pd_model.py may omit it.
    _PreIsoCal = getattr(_pm, "_PrecomputedBinaryIsotonicCalibrator", None)

    seen: set[int] = set()

    def _walk(x):
        if x is None or isinstance(x, (str, int, float, bool, bytes)):
            return
        if isinstance(x, (np.integer, np.floating)):
            return
        mid = id(x)
        if mid in seen:
            return
        seen.add(mid)

        cls = type(x)
        n = cls.__name__
        if n == "_StackedPDWrapper" and cls is not _StackedPDWrapper:
            x.__class__ = _StackedPDWrapper
        elif n == "_LRWithScaler" and cls is not _LRWithScaler:
            x.__class__ = _LRWithScaler
        elif (
            _PreIsoCal is not None
            and n == "_PrecomputedBinaryIsotonicCalibrator"
            and cls is not _PreIsoCal
        ):
            x.__class__ = _PreIsoCal

        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)
            return

        if type(x).__name__ == "FrozenEstimator" and hasattr(x, "estimator"):
            _walk(x.estimator)
        ccl = getattr(x, "calibrated_classifiers_", None)
        if ccl:
            for item in ccl:
                _walk(item)
                if hasattr(item, "estimator"):
                    _walk(item.estimator)
        be = getattr(x, "base_estimator", None)
        if be is not None:
            _walk(be)

    _walk(obj)
