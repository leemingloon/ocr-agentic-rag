"""
Train-only feature screening utilities (LendingClub-style).

Implements a lightweight, automated variant of common Lending Club PD best practices:
- Drop columns with high missingness (train-set only).
- Rank numeric/binary features by Kolmogorov–Smirnov (K-S) statistic between classes.
- Optionally prune highly correlated numeric feature pairs by keeping the higher K-S feature.

Design goals:
- Deterministic and easy to audit (returns a report dict).
- Safe against leakage by operating on TRAIN ONLY (caller responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except Exception as e:  # pragma: no cover
    ks_2samp = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FeatureScreeningResult:
    selected_features: List[str]
    dropped_missingness: List[str]
    dropped_low_ks: List[str]
    dropped_correlated: List[str]
    ks_stat: Dict[str, float]
    missing_frac: Dict[str, float]


def _ks_statistic(x0: np.ndarray, x1: np.ndarray) -> float:
    """Return K-S statistic (0..1). Safe defaults for small/empty samples."""
    if x0.size < 2 or x1.size < 2:
        return 0.0
    if ks_2samp is None:
        # Fallback: approximate with difference in means (still monotonic-ish)
        m0 = float(np.nanmean(x0)) if x0.size else 0.0
        m1 = float(np.nanmean(x1)) if x1.size else 0.0
        denom = float(np.nanstd(np.concatenate([x0, x1]))) or 1.0
        return float(min(abs(m0 - m1) / denom, 1.0))
    try:
        return float(ks_2samp(x0, x1, alternative="two-sided", mode="auto").statistic)
    except Exception:
        return 0.0


def compute_ks_stats(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    assume_binary_target: bool = True,
) -> Dict[str, float]:
    """
    Compute per-feature K-S statistics on TRAIN only.

    Assumes y_train is binary (0/1). For non-binary targets, the caller should not use this.
    """
    if assume_binary_target:
        y = np.asarray(y_train).astype(int)
    else:
        y = np.asarray(y_train)

    ks: Dict[str, float] = {}
    x0_mask = y == 0
    x1_mask = y == 1
    for col in X_train.columns:
        s = X_train[col]
        if not pd.api.types.is_numeric_dtype(s):
            ks[col] = 0.0
            continue
        a = s.to_numpy(dtype=float, copy=False)
        x0 = a[x0_mask]
        x1 = a[x1_mask]
        x0 = x0[~np.isnan(x0)]
        x1 = x1[~np.isnan(x1)]
        ks[col] = _ks_statistic(x0, x1)
    return ks


def prune_correlated_features(
    X_train: pd.DataFrame,
    feature_scores: Dict[str, float],
    *,
    corr_threshold: float = 0.95,
) -> Tuple[List[str], List[str]]:
    """
    Drop highly correlated numeric features (|corr| >= corr_threshold).

    Keeps the feature with higher `feature_scores` (e.g., K-S statistic).
    Returns (kept_features, dropped_features).
    """
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    if len(numeric_cols) <= 1:
        return list(X_train.columns), []

    corr = X_train[numeric_cols].corr(numeric_only=True).abs()
    # Upper triangle mask (excluding diagonal)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    # Greedy: iterate pairs in descending correlation magnitude
    pairs: List[Tuple[str, str, float]] = []
    for c in upper.columns:
        s = upper[c].dropna()
        for r, v in s.items():
            if v >= corr_threshold:
                pairs.append((r, c, float(v)))
    pairs.sort(key=lambda t: t[2], reverse=True)

    for a, b, _v in pairs:
        if a in to_drop or b in to_drop:
            continue
        sa = float(feature_scores.get(a, 0.0))
        sb = float(feature_scores.get(b, 0.0))
        # Drop lower score; tie-breaker: drop the second for stability
        drop = b if sa >= sb else a
        to_drop.add(drop)

    kept = [c for c in X_train.columns if c not in to_drop]
    return kept, sorted(to_drop)


def screen_features_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    missingness_threshold: float = 0.50,
    min_ks: float = 0.0,
    corr_threshold: float = 0.95,
) -> FeatureScreeningResult:
    """
    End-to-end screening on TRAIN ONLY:
    1) drop high-missingness columns
    2) compute K-S stats and optionally drop low-KS columns
    3) correlation pruning (keep higher K-S)
    """
    missing_frac = X_train.isna().mean().to_dict()
    drop_missing = sorted([c for c, frac in missing_frac.items() if float(frac) > missingness_threshold])
    X1 = X_train.drop(columns=drop_missing, errors="ignore")

    ks = compute_ks_stats(X1, y_train)
    drop_low_ks = sorted([c for c, v in ks.items() if float(v) < float(min_ks)])
    X2 = X1.drop(columns=drop_low_ks, errors="ignore")

    kept, drop_corr = prune_correlated_features(X2, ks, corr_threshold=corr_threshold)

    selected = kept
    return FeatureScreeningResult(
        selected_features=selected,
        dropped_missingness=drop_missing,
        dropped_low_ks=drop_low_ks,
        dropped_correlated=drop_corr,
        ks_stat={k: float(v) for k, v in ks.items()},
        missing_frac={k: float(v) for k, v in missing_frac.items()},
    )

