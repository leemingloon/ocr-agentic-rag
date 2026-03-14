#!/usr/bin/env python3
"""
One-time restructure of TATQA proof files to 3 metrics: substring_match, exact_match, f1.

Reads data/proof/rag/tatqa/test/tatqa_test_samples.json, recomputes exact_match (strict:
normalized pred == normalized GT) and f1 (token-level F1) per sample, carries substring_match
from old relaxed_match, rewrites each row's metrics and saves the JSON. Then you can run
eval_runner.py --reevaluate_only --category rag --dataset TATQA to refresh avg files and
predictions.txt, or this script can optionally write the avg JSONs and trigger export.

Usage:
  python scripts/restructure_tatqa_metrics.py [--dry-run]
  Then run: python eval_runner.py --reevaluate_only --category rag --dataset TATQA
  to refresh tatqa_test_avg.json, tatqa_avg.json, and tatqa_test_predictions.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_postprocess_utils import TATQAUtils


def main() -> None:
    ap = argparse.ArgumentParser(description="Restructure TATQA samples to 3 metrics only")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    args = ap.parse_args()

    samples_path = ROOT / "data" / "proof" / "rag" / "tatqa" / "test" / "tatqa_test_samples.json"
    if not samples_path.exists():
        print(f"Not found: {samples_path}")
        sys.exit(1)

    with open(samples_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        print("Expected list of samples")
        sys.exit(1)

    utils = TATQAUtils()
    n_sub, n_ex, f1_sum = 0, 0, 0.0
    for row in rows:
        met = row.get("metrics") or {}
        # substring_match = old relaxed_match (carry forward)
        substring_val = met.get("relaxed_match", met.get("substring_match", 0.0))
        if isinstance(substring_val, (int, float)):
            substring_val = float(substring_val)
        else:
            substring_val = 0.0

        pred = row.get("prediction") or ""
        if isinstance(pred, dict):
            pred = pred.get("answer", "") or ""
        pred = str(pred).strip()
        gt_obj = row.get("ground_truth") or {}
        gt_answer = gt_obj.get("answer") if isinstance(gt_obj, dict) else gt_obj
        ref = str(gt_answer or "").strip()

        # True exact match: normalized equality only
        exact = 1.0 if utils.normalize_text(pred) == utils.normalize_text(ref) else 0.0
        f1 = utils.token_f1(pred, ref)

        row["metrics"] = {
            "substring_match": substring_val,
            "exact_match": exact,
            "f1": round(f1, 4),
        }
        n_sub += 1 if substring_val == 1.0 else 0
        n_ex += 1 if exact == 1.0 else 0
        f1_sum += f1

    total = len(rows)
    sub_mean = n_sub / total if total else 0.0
    ex_mean = n_ex / total if total else 0.0
    f1_mean = f1_sum / total if total else 0.0

    print(f"Samples: {total}")
    print(f"substring_match: {sub_mean:.4f} ({n_sub}/{total})")
    print(f"exact_match:    {ex_mean:.4f} ({n_ex}/{total})")
    print(f"f1:             {f1_mean:.4f}")
    if ex_mean >= sub_mean and total > 0:
        print("WARNING: exact_match >= substring_match — override may still be present or data already strict.")

    if not args.dry_run:
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote {samples_path}")

        # Write split avg
        avg_path = samples_path.parent / "tatqa_test_avg.json"
        payload = {
            "sample_count": total,
            "substring_match": f"{sub_mean:.4f} ({n_sub}/{total})",
            "exact_match": f"{ex_mean:.4f} ({n_ex}/{total})",
            "f1": f"{f1_mean:.4f}",
        }
        with open(avg_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {avg_path}")

        # Write dataset avg if we only have one split
        dataset_avg_path = samples_path.parent.parent / "tatqa_avg.json"
        with open(dataset_avg_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {dataset_avg_path}")
    else:
        print("Dry run — no files written. Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
