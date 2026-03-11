#!/usr/bin/env python3
"""
Run a small-sample comparison of Claude Sonnet 4 vs 4.6 for RAG and Risk Memo (FinanceBench).
Use this to confirm which model gives better evaluation scores at the same cost before
running full evals. Requires ANTHROPIC_API_KEY.

Usage:
  python scripts/compare_llm_models_for_eval.py [--n 5] [--rag-only | --memo-only]

Reads metrics from data/proof/ after each run; second run overwrites predictions, so we
save metrics after the first model then run the second model and compare.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROOF_DIR = Path("data/proof")
MODEL_4 = "claude-sonnet-4-20250514"
MODEL_46 = "claude-sonnet-4-6"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _aggregate_metrics(rows: list[dict]) -> dict[str, float]:
    """Compute mean of numeric metrics over per-sample rows. Ignores missing keys."""
    if not rows:
        return {}
    metrics_keys = set()
    for r in rows:
        m = r.get("metrics")
        if isinstance(m, dict):
            metrics_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
    out = {}
    for k in sorted(metrics_keys):
        vals = []
        for r in rows:
            m = r.get("metrics")
            if isinstance(m, dict) and isinstance(m.get(k), (int, float)):
                vals.append(m[k])
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def _load_samples_json(category: str, dataset: str, split: str) -> list[dict]:
    """Load per-sample proof file; return list of rows. Returns [] if file missing."""
    cat_dir = PROOF_DIR / category.lower()
    ds_dir = cat_dir / dataset.lower()
    path = ds_dir / split / f"{dataset.lower()}_{split}_samples.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_eval(category: str, dataset: str, model_env_var: str, model_value: str, n: int, force_reeval: bool) -> None:
    env = os.environ.copy()
    env[model_env_var] = model_value
    cmd = [
        sys.executable,
        str(_repo_root() / "eval_runner.py"),
        "--category", category,
        "--dataset", dataset,
        "--max_split", str(n),
    ]
    if force_reeval:
        cmd.append("--force_reeval")
    subprocess.run(cmd, env=env, cwd=str(_repo_root()), check=True)


def _infer_split(category: str, dataset: str) -> str | None:
    """Infer the split name used in proof dir (e.g. train, default)."""
    cat_dir = PROOF_DIR / category.lower()
    ds_dir = cat_dir / dataset.lower()
    if not ds_dir.exists():
        return None
    for d in ds_dir.iterdir():
        if d.is_dir():
            samples_file = d / f"{dataset.lower()}_{d.name}_samples.json"
            if samples_file.exists():
                return d.name
    return None


def compare_rag(n: int) -> dict:
    """Run RAG (FinQA) with Sonnet 4 then 4.6, return metrics for both."""
    category, dataset = "rag", "FinQA"
    split = "train"  # FinQA adapter uses train
    results = {}

    print(f"[RAG] Running {n} samples with {MODEL_4} ...")
    _run_eval(category, dataset, "RAG_LLM_MODEL", MODEL_4, n, force_reeval=False)
    rows4 = _load_samples_json(category, dataset, split)
    if rows4:
        results[MODEL_4] = _aggregate_metrics(rows4)
        print(f"      -> {len(rows4)} samples, metrics: {results[MODEL_4]}")
    else:
        print("      -> No samples file found; run may have failed or path differs.")

    print(f"[RAG] Running same {n} samples with {MODEL_46} (--force_reeval) ...")
    _run_eval(category, dataset, "RAG_LLM_MODEL", MODEL_46, n, force_reeval=True)
    rows46 = _load_samples_json(category, dataset, split)
    if rows46:
        results[MODEL_46] = _aggregate_metrics(rows46)
        print(f"      -> {len(rows46)} samples, metrics: {results[MODEL_46]}")
    else:
        print("      -> No samples file found.")

    return results


def compare_memo(n: int) -> dict:
    """Run credit_risk_memo_generator (FinanceBench) with Sonnet 4 then 4.6."""
    category, dataset = "credit_risk_memo_generator", "FinanceBench"
    split = "default"  # FinanceBench FILE_MAPPING uses "default"
    results = {}

    print(f"[Memo] Running {n} samples with {MODEL_4} ...")
    _run_eval(category, dataset, "CREDIT_RISK_MEMO_MODEL", MODEL_4, n, force_reeval=False)
    rows4 = _load_samples_json(category, dataset, split)
    if rows4:
        results[MODEL_4] = _aggregate_metrics(rows4)
        print(f"       -> {len(rows4)} samples, metrics: {results[MODEL_4]}")
    else:
        # Try inferred split (e.g. if adapter emits different split name)
        split_infer = _infer_split(category, dataset)
        if split_infer:
            rows4 = _load_samples_json(category, dataset, split_infer)
            if rows4:
                results[MODEL_4] = _aggregate_metrics(rows4)
                print(f"       -> {len(rows4)} samples (split={split_infer}), metrics: {results[MODEL_4]}")
        if not results:
            print("       -> No samples file found.")

    print(f"[Memo] Running same {n} samples with {MODEL_46} (--force_reeval) ...")
    _run_eval(category, dataset, "CREDIT_RISK_MEMO_MODEL", MODEL_46, n, force_reeval=True)
    rows46 = _load_samples_json(category, dataset, split)
    if not rows46 and _infer_split(category, dataset):
        rows46 = _load_samples_json(category, dataset, _infer_split(category, dataset))
    if rows46:
        results[MODEL_46] = _aggregate_metrics(rows46)
        print(f"       -> {len(rows46)} samples, metrics: {results[MODEL_46]}")
    else:
        print("       -> No samples file found.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Sonnet 4 vs 4.6 on a few RAG and Memo samples")
    parser.add_argument("--n", type=int, default=5, help="Max samples per split (default 5)")
    parser.add_argument("--rag-only", action="store_true", help="Only run RAG (FinQA) comparison")
    parser.add_argument("--memo-only", action="store_true", help="Only run Memo (FinanceBench) comparison")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; comparison will fail for API-based evals.")
        sys.exit(1)

    all_results = {}
    if not args.memo_only:
        all_results["RAG (FinQA)"] = compare_rag(args.n)
    if not args.rag_only:
        all_results["Memo (FinanceBench)"] = compare_memo(args.n)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY (higher is better for all metrics)")
    print("=" * 60)
    for task, model_metrics in all_results.items():
        print(f"\n{task}:")
        if not model_metrics:
            print("  (no data)")
            continue
        for model, metrics in model_metrics.items():
            label = "4.6" if "4-6" in model else "4"
            parts = [f"  {k}={v:.4f}" for k, v in sorted(metrics.items())]
            print(f"  Sonnet {label}: " + ", ".join(parts))
    print()


if __name__ == "__main__":
    main()
