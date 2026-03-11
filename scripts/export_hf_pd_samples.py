#!/usr/bin/env python3
"""
Export HF LendingClub benchmark samples to parquet for calibration or HF validation selection.

Usage:
    python scripts/export_hf_pd_samples.py --out data/credit_risk_pd/LendingClub/processed/hf_samples.parquet
    python scripts/export_hf_pd_samples.py --max-per-split 500 --out hf_valid.parquet

Then use for:
- Platt/isotonic calibration: fit on (pd_prob, y_true) from this export
- HF validation selection: use for Optuna trial selection or early stopping
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/credit_risk_pd/LendingClub/processed/hf_pd_samples.parquet"))
    ap.add_argument("--max-per-split", type=int, default=None, help="Max samples per split (train/valid/test)")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    import os
    import sys
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from eval_dataset_adapters import LendingClubAdapter
    from credit_risk.feature_engineering.common_features import get_feature_names_no_leakage_v2

    adapter = LendingClubAdapter(category="credit_risk_PD", dataset_name="LendingClub")
    feature_names = get_feature_names_no_leakage_v2()
    rows = []
    for sample in adapter.load_split(
        dataset_split=None,
        max_samples_per_split=args.max_per_split,
    ):
        features = sample.get("input", {}).get("features", {})
        label = sample.get("ground_truth", {}).get("label")
        split = sample.get("metadata", {}).get("split", "unknown")
        if label is None:
            continue
        row = {"split": split, "default": int(label)}
        for fn in feature_names:
            row[fn] = features.get(fn, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Exported {len(df)} samples to {args.out}")
    for s in args.splits:
        n = (df["split"] == s).sum()
        if n > 0:
            p = df.loc[df["split"] == s, "default"].mean()
            print(f"  {s}: n={n}, default_rate={p:.2%}")


if __name__ == "__main__":
    main()
