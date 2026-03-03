#!/usr/bin/env python3
"""
Train PD model with HF-benchmark robustness improvements.

Implements:
- HF-style feature dropout (5-15% NaN per column) during training
- HF-robust Optuna search space (max_depth 3-6, min_child_weight 3-10, etc.)
- XGB + LGB + CatBoost stacking with meta-learner
- Saves to models/pd/pd_model_local_v2.pkl

Run after 01_lendingclub_feature_engineering.ipynb to create lendingclub_engineered.parquet.
Then: python scripts/train_pd_model_hf_robust.py
"""

from __future__ import annotations

import argparse
import joblib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def _apply_hf_dropout(X: pd.DataFrame, p_drop: float = 0.10, seed: int = 42) -> pd.DataFrame:
    """Randomly set p_drop fraction of values per column to NaN (HF-style missing)."""
    X_noisy = X.copy()
    rng = random.Random(seed)
    for col in X_noisy.columns:
        n = len(X_noisy)
        n_drop = int(n * p_drop)
        if n_drop > 0:
            idx = rng.sample(range(n), n_drop)
            X_noisy.iloc[idx, X_noisy.columns.get_loc(col)] = np.nan
    return X_noisy


def _fill_median(X: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fill NaN with column median. Return (filled_df, medians)."""
    if medians is None:
        medians = X.median().to_dict()
    X_filled = X.fillna(medians)
    return X_filled, medians


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/credit_risk_pd/LendingClub/processed/lendingclub_engineered.parquet"))
    ap.add_argument("--out", type=Path, default=Path("models/pd/pd_model_local_v2.pkl"))
    ap.add_argument("--p_drop", type=float, default=0.10, help="HF dropout fraction per column")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Add repo root to path
    repo_root = Path(__file__).resolve().parent.parent
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from credit_risk.feature_engineering.common_features import get_feature_names_no_leakage

    df = pd.read_parquet(args.data)
    feature_names = get_feature_names_no_leakage()
    X = df[[c for c in feature_names if c in df.columns]].copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_names]
    y = df["default"]

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=args.seed, stratify=y_rest)

    # HF simulation: create noisy copy and concatenate (or use noisy only for robustness)
    X_train_noisy = _apply_hf_dropout(X_train, p_drop=args.p_drop, seed=args.seed)
    medians = X_train.median().to_dict()
    X_train_noisy, _ = _fill_median(X_train_noisy, medians)
    X_train_aug = pd.concat([X_train, X_train_noisy], axis=0, ignore_index=True)
    y_train_aug = pd.concat([y_train, y_train], axis=0, ignore_index=True)
    X_val_filled, _ = _fill_median(X_val, medians)

    scale = (y_train_aug == 0).sum() / max((y_train_aug == 1).sum(), 1)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 150, 400),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.7, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2.0),
                "objective": "binary:logistic",
                "scale_pos_weight": scale,
                "eval_metric": "auc",
                "random_state": args.seed,
                "tree_method": "hist",
            }
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            aucs = []
            for train_idx, val_idx in skf.split(X_train_aug, y_train_aug):
                X_f, X_v = X_train_aug.iloc[train_idx], X_train_aug.iloc[val_idx]
                y_f, y_v = y_train_aug.iloc[train_idx], y_train_aug.iloc[val_idx]
                import xgboost as xgb
                clf = xgb.XGBClassifier(**params)
                clf.fit(X_f, y_f, eval_set=[(X_v, y_v)], verbose=False)
                p = clf.predict_proba(X_v)[:, 1]
                aucs.append(roc_auc_score(y_v, p))
            return np.mean(aucs)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8))
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        best_params = {k: v for k, v in study.best_params.items() if k in ["max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
        best_params.update(objective="binary:logistic", scale_pos_weight=scale, eval_metric="auc", random_state=args.seed, tree_method="hist")
        print("Best Optuna AUC:", round(study.best_value, 4))
    except ImportError:
        best_params = {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 250, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 0.5, "objective": "binary:logistic", "scale_pos_weight": scale, "eval_metric": "auc", "random_state": args.seed, "tree_method": "hist"}

    import xgboost as xgb
    final_xgb = xgb.XGBClassifier(**best_params)
    final_xgb.fit(X_train_aug, y_train_aug, eval_set=[(X_val_filled, y_val)], verbose=False)

    import lightgbm as lgb
    lgb_scale = (y_train_aug == 0).sum() / max((y_train_aug == 1).sum(), 1)
    lgb_full = lgb.train(
        {"objective": "binary", "metric": "auc", "boosting_type": "gbdt", "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "random_state": args.seed, "scale_pos_weight": lgb_scale},
        lgb.Dataset(X_train_aug, y_train_aug),
        num_boost_round=200,
    )

    catboost_model = None
    try:
        import catboost as cb
        cb_model = cb.CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05, verbose=0, random_state=args.seed, scale_pos_weight=lgb_scale)
        cb_model.fit(X_train_aug, y_train_aug)
        meta_inputs = [
            final_xgb.predict_proba(X_train_aug)[:, 1],
            lgb_full.predict(X_train_aug),
            cb_model.predict_proba(X_train_aug)[:, 1],
        ]
        catboost_model = cb_model
    except ImportError:
        meta_inputs = [final_xgb.predict_proba(X_train_aug)[:, 1], lgb_full.predict(X_train_aug)]

    meta = LogisticRegression(max_iter=500, random_state=args.seed)
    meta.fit(np.column_stack(meta_inputs), y_train_aug)

    from credit_risk.models.pd_model import _StackedPDWrapper
    final_model = _StackedPDWrapper(final_xgb, lgb_full, meta, catboost_model=catboost_model)

    p_val = final_model.predict_proba(X_val_filled)[:, 1]
    auc_val = roc_auc_score(y_val, p_val)
    print("Val AUC:", round(auc_val, 4))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": final_model,
        "feature_names": feature_names,
        "params": best_params,
        "metadata": {"trained_with": "train_pd_model_hf_robust", "p_drop": args.p_drop, "n_train": len(X_train_aug)},
    }
    joblib.dump(model_data, args.out)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
