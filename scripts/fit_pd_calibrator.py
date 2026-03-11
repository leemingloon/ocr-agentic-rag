#!/usr/bin/env python3
"""
Fit Platt/isotonic calibrator on HF validation set and add to saved PD model.

Usage:
    python scripts/export_hf_pd_samples.py --max-per-split 500 --out data/credit_risk_pd/LendingClub/processed/hf_valid.parquet
    python scripts/fit_pd_calibrator.py --data data/credit_risk_pd/LendingClub/processed/hf_valid.parquet --model models/pd/pd_model_local_v2.pkl --method platt

Then the model pkl will contain a calibrator; PDModel.load() applies it automatically.
"""

from __future__ import annotations

import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Parquet with features + default (from export_hf_pd_samples)")
    ap.add_argument("--model", type=Path, default=Path("models/pd/pd_model_local_v2.pkl"))
    ap.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    ap.add_argument("--split", type=str, default="valid", help="Split to use for calibration (train/valid/test)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    df = pd.read_parquet(args.data)
    df = df[df["split"] == args.split]
    if len(df) == 0:
        raise SystemExit(f"No rows for split={args.split} in {args.data}")

    model_data = joblib.load(args.model)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    X = df[[c for c in feature_names if c in df.columns]].copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_names]
    y = df["default"].values

    pd_probs = model.predict_proba(X)[:, 1]

    from credit_risk.models.pd_calibration import fit_pd_calibrator

    calibrator = fit_pd_calibrator(pd_probs, y, method=args.method)
    model_data["calibrator"] = calibrator
    joblib.dump(model_data, args.model)
    print(f"Calibrator ({args.method}) fitted on {len(df)} {args.split} samples, saved to {args.model}")


if __name__ == "__main__":
    main()
