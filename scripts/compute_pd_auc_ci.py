#!/usr/bin/env python3
"""
Compute AUC-ROC and 95% bootstrap confidence interval for PD benchmark.

Reads per-sample proof files from data/proof/credit_risk_pd/lendingclub/{train,valid,test}/
and writes auc_roc_mean_ci_low, auc_roc_mean_ci_high to the dataset avg JSON.

Usage:
  python scripts/compute_pd_auc_ci.py
  # Or with custom proof dir:
  python scripts/compute_pd_auc_ci.py --proof_dir data/proof/credit_risk_pd/lendingclub
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _samples_filename(dataset_name: str, split_name: str) -> str:
    return f"{dataset_name.lower()}_{split_name}_samples.json"


def bootstrap_auc_ci(
    y_true: list[int | float],
    y_score: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute AUC and 95% CI via bootstrap. Returns (auc, ci_low, ci_high)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return 0.5, 0.5, 0.5

    n = len(y_true)
    if n < 2:
        return 0.5, 0.5, 0.5

    rng = random.Random(seed)
    aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        y_b = [y_true[i] for i in idx]
        s_b = [y_score[i] for i in idx]
        try:
            auc_b = roc_auc_score(y_b, s_b)
        except ValueError:
            auc_b = 0.5
        aucs.append(auc_b)

    aucs.sort()
    try:
        auc_point = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_point = 0.5

    ci_low = aucs[int(0.025 * n_bootstrap)]
    ci_high = aucs[int(0.975 * n_bootstrap)]
    return auc_point, ci_low, ci_high


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--proof_dir",
        type=Path,
        default=Path("data/proof/credit_risk_pd/lendingclub"),
        help="Path to dataset proof dir (contains train/valid/test subdirs)",
    )
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    proof_dir = args.proof_dir
    if not proof_dir.exists():
        print(f"Proof dir not found: {proof_dir}")
        return

    dataset_name = proof_dir.name
    if dataset_name == "lendingclub":
        dataset_name = "LendingClub"

    y_true_all: list[int | float] = []
    y_score_all: list[float] = []

    for split_dir in proof_dir.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        samples_path = split_dir / _samples_filename(dataset_name, split_name)
        if not samples_path.exists():
            continue
        try:
            with open(samples_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            print(f"Could not load {samples_path}: {e}")
            continue
        for row in rows:
            if row.get("prediction_error"):
                continue
            m = row.get("metrics") or {}
            gt = m.get("gt_binary")
            pd_prob = m.get("pd_prob")
            if gt is not None and pd_prob is not None:
                y_true_all.append(gt)
                y_score_all.append(float(pd_prob))

    if len(y_true_all) < 2:
        print("Too few samples for bootstrap")
        return

    auc_point, ci_low, ci_high = bootstrap_auc_ci(
        y_true_all, y_score_all, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    print(f"AUC: {auc_point:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
    print(f"  n_samples: {len(y_true_all)}")

    avg_path = proof_dir / f"{proof_dir.name}_avg.json"
    if avg_path.exists():
        with open(avg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}

    payload["auc_roc_mean"] = auc_point
    payload["auc_roc_mean_ci_low"] = ci_low
    payload["auc_roc_mean_ci_high"] = ci_high
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Updated {avg_path}")


if __name__ == "__main__":
    main()
