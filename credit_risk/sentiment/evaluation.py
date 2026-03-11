"""
Evaluation helpers: F1 on full set and on failure-mode subsets (negation, conditional, numeric, hedged, entity).
Report before/after when applying pipeline fixes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def _labels_to_ids(labels: List[str]) -> np.ndarray:
    return np.array([LABEL2ID.get(str(x).strip().lower(), 1) for x in labels])


def compute_f1_macro(y_true: List[str], y_pred: List[str]) -> float:
    if not SKLEARN_AVAILABLE:
        return 0.0
    yt = _labels_to_ids(y_true)
    yp = _labels_to_ids(y_pred)
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def evaluate_on_subsets(
    texts: List[str],
    y_true: List[str],
    y_pred: List[str],
    negation_fn: Optional[Callable[[str], bool]] = None,
    conditional_fn: Optional[Callable[[str], bool]] = None,
    numeric_fn: Optional[Callable[[str], bool]] = None,
    hedged_fn: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute F1 macro on full set and on each failure-mode subset (where mask is True).
    Returns e.g. { "full": {"f1": 0.86, "n": 112 }, "negation": {"f1": 0.7, "n": 15 }, ... }.
    """
    results = {}
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(texts)
    if n == 0:
        return results

    # Full
    if SKLEARN_AVAILABLE:
        f1_full = float(f1_score(_labels_to_ids(y_true), _labels_to_ids(y_pred), average="macro", zero_division=0))
    else:
        f1_full = 0.0
    results["full"] = {"f1": f1_full, "n": n}

    def subset_f1(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        return float(f1_score(
            _labels_to_ids(y_true_arr[mask].tolist()),
            _labels_to_ids(y_pred_arr[mask].tolist()),
            average="macro",
            zero_division=0,
        ))

    if negation_fn:
        mask = np.array([bool(negation_fn(t)) for t in texts])
        results["negation"] = {"f1": subset_f1(mask), "n": int(mask.sum())}
    if conditional_fn:
        mask = np.array([bool(conditional_fn(t)) for t in texts])
        results["conditional"] = {"f1": subset_f1(mask), "n": int(mask.sum())}
    if numeric_fn:
        mask = np.array([bool(numeric_fn(t)) for t in texts])
        results["numeric_comparison"] = {"f1": subset_f1(mask), "n": int(mask.sum())}
    if hedged_fn:
        mask = np.array([bool(hedged_fn(t)) for t in texts])
        results["hedged"] = {"f1": subset_f1(mask), "n": int(mask.sum())}

    return results


def print_evaluation_report(
    results: Dict[str, Dict[str, float]],
    title: str = "Evaluation",
) -> None:
    """Print before/after style report."""
    print(title)
    print("-" * 50)
    for key, v in results.items():
        print(f"  {key}: F1 macro = {v['f1']:.4f} (n = {v['n']})")
    print()
