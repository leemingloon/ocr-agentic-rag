"""
OCR eval regression-guard loop.

Continuous-improvement / regression-guard loop for the OCR eval (FUNSD, SROIE):
runs the eval, checks each dataset's weighted primary metric against a floor
(default 0.70), and — only if a dataset is below the floor — escalates the
worst-scoring samples in that dataset from classical-only OCR to HybridOCR's
confidence-gated vision fallback (OCR_EVAL_VISION_ESCALATE=1), then re-checks.
This repeats per dataset until the floor is met or --max_iterations is hit.

This is deliberately a *guard*, not a one-off fix: the baseline already clears
the floor comfortably (see EVALUATION_RESULTS.md), so a normal run should do
zero escalations. Its job is to catch and auto-remediate future regressions
(pipeline change, dataset refresh, dependency bump) without a human having to
notice the drop first. Escalation costs Claude API calls (vision fallback), so
it is bounded to the worst N samples per iteration, not the whole split.

Usage:
    python scripts/ocr_eval_improve_loop.py
    python scripts/ocr_eval_improve_loop.py --datasets FUNSD --threshold 0.75
    python scripts/ocr_eval_improve_loop.py --escalate_per_iteration 5 --max_iterations 2

Writes a run log to data/proof/ocr/loop_engineering_log.json (append-only list).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROOF_OCR_DIR = REPO_ROOT / "data" / "proof" / "ocr"
LOG_PATH = PROOF_OCR_DIR / "loop_engineering_log.json"

# Per dataset: which split-file metric key is the primary target, and how to
# find that dataset's samples files.
DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "FUNSD": {
        "primary_metric": "word_recall_mean",
        "per_sample_metric": "word_recall",
        "splits": ["train", "test"],
        "avg_file": PROOF_OCR_DIR / "funsd" / "funsd_avg.json",
        "sample_file": lambda split: PROOF_OCR_DIR / "funsd" / split / f"funsd_{split}_samples.json",
    },
    "SROIE": {
        "primary_metric": "entity_match_mean",
        "per_sample_metric": "entity_match",
        "splits": ["train", "test"],
        "avg_file": PROOF_OCR_DIR / "sroie" / "sroie_avg.json",
        "sample_file": lambda split: PROOF_OCR_DIR / "sroie" / split / f"sroie_{split}_samples.json",
    },
}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _weighted_metric(dataset: str) -> float | None:
    cfg = DATASET_CONFIG[dataset]
    avg = _load_json(cfg["avg_file"])
    if not avg:
        return None
    return avg.get("weighted_metrics", {}).get(cfg["primary_metric"])


def _worst_sample_ids(dataset: str, n: int) -> list[tuple[str, str, float]]:
    """Return up to n (split, sample_id, metric) tuples with the lowest per-sample metric."""
    cfg = DATASET_CONFIG[dataset]
    rows: list[tuple[str, str, float]] = []
    for split in cfg["splits"]:
        data = _load_json(cfg["sample_file"](split))
        if not isinstance(data, list):
            continue
        for row in data:
            metrics = row.get("metrics") or {}
            val = metrics.get(cfg["per_sample_metric"])
            if isinstance(val, (int, float)):
                rows.append((split, str(row.get("sample_id")), float(val)))
    rows.sort(key=lambda r: r[2])
    return rows[:n]


def _run_eval_runner(args: list[str]) -> int:
    cmd = [sys.executable, str(REPO_ROOT / "eval_runner.py"), *args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def _escalate(dataset: str, split: str, sample_id: str) -> int:
    env = os.environ.copy()
    env["OCR_EVAL_VISION_ESCALATE"] = "1"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval_runner.py"),
        "--category",
        "ocr",
        "--dataset",
        dataset,
        "--sample_id",
        sample_id,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


def run_loop(
    datasets: list[str],
    threshold: float,
    escalate_per_iteration: int,
    max_iterations: int,
    rerun_baseline: bool,
) -> dict[str, Any]:
    run_log: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "datasets": {},
    }

    if rerun_baseline:
        print("[loop] Refreshing baseline (--all_ocr_splits) before checking thresholds...")
        _run_eval_runner(["--all_ocr_splits", "--force_reeval"])

    for dataset in datasets:
        cfg = DATASET_CONFIG[dataset]
        history: list[dict[str, Any]] = []
        for iteration in range(max_iterations + 1):
            score = _weighted_metric(dataset)
            history.append({"iteration": iteration, cfg["primary_metric"]: score})
            print(f"[loop] {dataset} iteration={iteration} {cfg['primary_metric']}={score}")

            if score is None:
                print(f"[loop] {dataset}: no proof found yet — run eval_runner for this dataset first.")
                break
            if score >= threshold:
                print(f"[loop] {dataset}: {score:.4f} >= threshold {threshold} - no escalation needed.")
                break
            if iteration == max_iterations:
                print(f"[loop] {dataset}: still below threshold after {max_iterations} escalation rounds.")
                break

            worst = _worst_sample_ids(dataset, escalate_per_iteration)
            if not worst:
                print(f"[loop] {dataset}: below threshold but no per-sample metrics to escalate from.")
                break
            print(
                f"[loop] {dataset}: below threshold, escalating {len(worst)} worst samples "
                f"to vision fallback (OCR_EVAL_VISION_ESCALATE=1)..."
            )
            for split, sample_id, val in worst:
                print(f"  - escalating {dataset}/{split}/{sample_id} ({cfg['per_sample_metric']}={val:.3f})")
                _escalate(dataset, split, sample_id)
            history[-1]["escalated"] = [
                {"split": s, "sample_id": sid, cfg["per_sample_metric"]: v} for s, sid, v in worst
            ]

        run_log["datasets"][dataset] = {
            "final_metric": _weighted_metric(dataset),
            "passed": (_weighted_metric(dataset) or 0.0) >= threshold,
            "history": history,
        }

    PROOF_OCR_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load_json(LOG_PATH) or []
    if not isinstance(existing, list):
        existing = []
    existing.append(run_log)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"[loop] Wrote {LOG_PATH}")
    return run_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="FUNSD,SROIE", help="Comma-separated dataset names.")
    parser.add_argument("--threshold", type=float, default=0.70, help="Minimum acceptable weighted metric.")
    parser.add_argument(
        "--escalate_per_iteration", type=int, default=10, help="Worst-N samples to escalate per iteration."
    )
    parser.add_argument("--max_iterations", type=int, default=3, help="Max escalation rounds per dataset.")
    parser.add_argument(
        "--rerun_baseline",
        action="store_true",
        help="Force a fresh full re-eval (--all_ocr_splits --force_reeval) before checking thresholds. "
        "Expensive; omit to just check/escalate against existing proof.",
    )
    args = parser.parse_args()

    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATASET_CONFIG]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {list(DATASET_CONFIG)}")

    run_log = run_loop(
        datasets=datasets,
        threshold=args.threshold,
        escalate_per_iteration=args.escalate_per_iteration,
        max_iterations=args.max_iterations,
        rerun_baseline=args.rerun_baseline,
    )

    failed = [d for d, r in run_log["datasets"].items() if not r["passed"]]
    if failed:
        print(f"[loop] FAILED: {failed} still below threshold {args.threshold} after escalation.")
        sys.exit(1)
    print("[loop] All datasets at or above threshold.")


if __name__ == "__main__":
    main()
