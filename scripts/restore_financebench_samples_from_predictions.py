"""
Restore financebench_train_samples.json from financebench_train_predictions.txt
after a crash (e.g. disk full) that truncated the samples JSON.

Reads the predictions file, parses each block, and writes the expected per-sample
JSON structure so --reevaluate_only and summaries work again.
"""
import ast
import json
import re
import sys
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_TXT = REPO_ROOT / "data" / "proof" / "credit_risk_memo_generator" / "financebench" / "train" / "financebench_train_predictions.txt"
SAMPLES_JSON = REPO_ROOT / "data" / "proof" / "credit_risk_memo_generator" / "financebench" / "train" / "financebench_train_samples.json"


def parse_predictions_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n========================================================================\n", text.strip())
    rows = []
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("gt_override_count:"):
            continue
        lines = block.split("\n")
        i = 0
        sample_id = None
        split_name = "train"
        ground_truth = None
        question = None
        prediction = ""
        metrics = None

        while i < len(lines):
            line = lines[i]
            if line.startswith("sample_id:"):
                sample_id = line.split(":", 1)[1].strip()
                i += 1
                continue
            if line.startswith("split:"):
                split_name = line.split(":", 1)[1].strip()
                i += 1
                continue
            if line.startswith("ground_truth:"):
                gt_raw = line.split(":", 1)[1].strip()
                try:
                    ground_truth = ast.literal_eval(gt_raw)
                except Exception:
                    ground_truth = {"reference": gt_raw}
                i += 1
                continue
            if line.startswith("question:"):
                question = line.split(":", 1)[1].strip()
                i += 1
                continue
            if line.strip() == "------------------------------------------------------------------------":
                i += 1
                if i < len(lines) and lines[i].strip() == "prediction:":
                    i += 1
                pred_lines = []
                while i < len(lines) and lines[i].strip() != "------------------------------------------------------------------------":
                    pred_lines.append(lines[i])
                    i += 1
                prediction = "\n".join(pred_lines).strip()
                if i < len(lines):
                    i += 1  # consume ------------
                if i < len(lines) and lines[i].strip().startswith("metrics:"):
                    try:
                        metrics = json.loads(lines[i].split(":", 1)[1].strip())
                    except Exception:
                        metrics = {}
                    i += 1
                break
            i += 1

        if sample_id is not None and ground_truth is not None and metrics is not None:
            rows.append({
                "sample_id": sample_id,
                "split": split_name,
                "ground_truth": ground_truth,
                "input_text": {
                    "question": question or "",
                    "context": question or "",
                    "prompt": question or "",
                },
                "prediction": prediction,
                "prediction_error": None,
                "metrics": metrics,
            })
    return rows


def main() -> int:
    if not PREDICTIONS_TXT.exists():
        print(f"Predictions file not found: {PREDICTIONS_TXT}", file=sys.stderr)
        return 1
    rows = parse_predictions_file(PREDICTIONS_TXT)
    if not rows:
        print("No samples parsed from predictions file.", file=sys.stderr)
        return 1
    SAMPLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLES_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Restored {len(rows)} samples to {SAMPLES_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
