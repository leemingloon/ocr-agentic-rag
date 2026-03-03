#!/usr/bin/env bash
# Run RAG eval one sample at a time per dataset. For each dataset:
#   - After each run, check the *latest evaluated sample*: if all its metrics are 1, run again; otherwise stop.
# Default: (1) FinQA until first failure, then (2) TATQA until first failure.
#
# Usage: from repo root
#   bash scripts/run_rag_eval_until_fail.sh              # run FinQA then TATQA (default)
#   bash scripts/run_rag_eval_until_fail.sh --dataset finqa   # run only FinQA
#   bash scripts/run_rag_eval_until_fail.sh --dataset tatqa  # run only TATQA

set -e
LOG=".rag_eval_run.log"

# Parse --dataset finqa | tatqa | both (default: both)
RUN_FINQA=0
RUN_TATQA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      case "$(echo "$2" | tr '[:upper:]' '[:lower:]')" in
        finqa)  RUN_FINQA=1 ;;
        tatqa)  RUN_TATQA=1 ;;
        both)   RUN_FINQA=1; RUN_TATQA=1 ;;
        *)      echo "Unknown --dataset: $2 (use finqa, tatqa, or both)" >&2; exit 1 ;;
      esac
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2; exit 1
      ;;
  esac
done
if [[ $RUN_FINQA -eq 0 && $RUN_TATQA -eq 0 ]]; then
  RUN_FINQA=1
  RUN_TATQA=1
fi

# Run one dataset until the latest evaluated sample has any metric != 1. Returns 0 when should stop, 1 when should continue.
# Stops when: (1) latest sample has any metric != 1, or (2) no progress (run evaluated 0 new samples) to avoid infinite loop.
run_dataset_until_fail() {
  local DATASET="$1"
  # Proof samples path: data/proof/rag/<dataset_lower>/<split>/<dataset>_<split>_samples.json
  # For TATQA we may have train or dev; the checker will try both if needed.
  local SAMPLES_DIR="data/proof/rag/$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
  local CMD="python eval_runner.py --category rag --dataset $DATASET --max_split 1 --max_category 1 --debug --export_predictions_txt"

  while true; do
    # Capture last block sample_id BEFORE run to detect "no progress" (all samples already evaluated).
    LAST_SAMPLE_BEFORE=$(python -c "
import glob
import os
samples_dir = '''$SAMPLES_DIR'''
pred_candidates = glob.glob(os.path.join(samples_dir, '*', '*_predictions.txt')) if os.path.isdir(samples_dir) else []
if not pred_candidates:
    print('')
    exit(0)
latest_pred = max(pred_candidates, key=os.path.getmtime)
with open(latest_pred, encoding='utf-8') as f:
    content = f.read()
blocks = [b.strip() for b in content.split('========') if b.strip()]
if not blocks:
    print('')
    exit(0)
last_block = blocks[-1]
for line in last_block.splitlines():
    line = line.strip()
    if line.startswith('sample_id:'):
        print(line.split('sample_id:', 1)[1].strip())
        break
else:
    print('')
" 2>/dev/null || true)

    echo "=== Running: $CMD ==="
    $CMD 2>&1 | tee "$LOG"
    CONTINUE=$(python -c "
import json
import re
import sys
import glob
import os

dataset_name = '''$DATASET'''
samples_dir = '''$SAMPLES_DIR'''

# Prefer: last evaluated sample from the predictions file (works when corpus_id is None in log)
metrics = None
pred_candidates = []
if os.path.isdir(samples_dir):
    pred_candidates = glob.glob(os.path.join(samples_dir, '*', '*_predictions.txt'))
if pred_candidates:
    # Use most recently modified predictions file (just written by this run)
    latest_pred = max(pred_candidates, key=os.path.getmtime)
    try:
        with open(latest_pred, encoding='utf-8') as f:
            content = f.read()
        blocks = [b.strip() for b in content.split('========') if b.strip()]
        if blocks:
            last_block = blocks[-1]
            # If last sample had prediction_error, it is now recorded so next run will skip it; continue to advance.
            if 'prediction_error:' in last_block:
                print('CONTINUE')
                sys.exit(0)
            for line in last_block.splitlines():
                line = line.strip()
                if line.startswith('metrics:'):
                    try:
                        metrics = json.loads(line.split('metrics:', 1)[1].strip())
                    except Exception:
                        pass
                    break
    except Exception:
        pass

# Fallback: get last sample from log and metrics from samples JSON (fails when corpus_id=None)
if metrics is None:
    log_path = '''$LOG'''
    with open(log_path, encoding='utf-8', errors='replace') as f:
        log = f.read()
    m = list(re.finditer(r\"\\[DEBUG\\] RAG query corpus_id='([^']+)'\", log))
    if not m:
        print('STOP')
        sys.exit(0)
    sample_id = m[-1].group(1)
    candidates = []
    if os.path.isdir(samples_dir):
        candidates = glob.glob(os.path.join(samples_dir, '*', '*_samples.json'))
    if not candidates:
        prefix = dataset_name.lower().replace('-', '')
        single = os.path.join(samples_dir, 'train', prefix + '_train_samples.json')
        if os.path.isfile(single):
            candidates = [single]
    if not candidates:
        print('STOP', file=sys.stderr)
        sys.exit(1)
    for path in candidates:
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, list):
            row = next((r for r in data if str(r.get('sample_id')) == sample_id or str(r.get('ground_truth', {}).get('corpus_id')) == sample_id), None)
            if row is not None:
                metrics = (row or {}).get('metrics') or {}
                break
    if metrics is None:
        print('STOP', file=sys.stderr)
        sys.exit(1)

required = {'program_accuracy', 'numerical_exact_match', 'exact_match', 'f1'}
optional = {'numerical_near_match'}
all_one = all(metrics.get(k) == 1.0 or metrics.get(k) == 1 for k in required)
if optional & set(metrics):
    all_one = all_one and all(metrics.get(k) == 1.0 or metrics.get(k) == 1 for k in optional if k in metrics)
print('CONTINUE' if all_one else 'STOP')
")
    if [[ "$CONTINUE" != "CONTINUE" ]]; then
      echo "Latest evaluated sample does not have all metrics 1. Stopping $DATASET."
      return 0
    fi
    # No progress check: if this run evaluated 0 new samples, last block sample_id is unchanged → stop to avoid infinite loop.
    LAST_SAMPLE_AFTER=$(python -c "
import glob
import os
samples_dir = '''$SAMPLES_DIR'''
pred_candidates = glob.glob(os.path.join(samples_dir, '*', '*_predictions.txt')) if os.path.isdir(samples_dir) else []
if not pred_candidates:
    print('')
    exit(0)
latest_pred = max(pred_candidates, key=os.path.getmtime)
with open(latest_pred, encoding='utf-8') as f:
    content = f.read()
blocks = [b.strip() for b in content.split('========') if b.strip()]
if not blocks:
    print('')
    exit(0)
last_block = blocks[-1]
for line in last_block.splitlines():
    line = line.strip()
    if line.startswith('sample_id:'):
        print(line.split('sample_id:', 1)[1].strip())
        break
else:
    print('')
" 2>/dev/null || true)
    if [[ -n "$LAST_SAMPLE_BEFORE" && -n "$LAST_SAMPLE_AFTER" && "$LAST_SAMPLE_BEFORE" == "$LAST_SAMPLE_AFTER" ]]; then
      echo "No progress (last sample unchanged: $LAST_SAMPLE_AFTER). All samples likely evaluated. Stopping $DATASET."
      return 0
    fi
    echo "Latest sample: all metrics 1. Running again..."
  done
}

# Phase 1: FinQA until first failure (if selected)
if [[ $RUN_FINQA -eq 1 ]]; then
  echo "========== FinQA (until first failure) =========="
  run_dataset_until_fail "FinQA"
fi

# Phase 2: TATQA until first failure (if selected)
if [[ $RUN_TATQA -eq 1 ]]; then
  echo "========== TATQA (until first failure) =========="
  run_dataset_until_fail "TATQA"
fi

echo "========== Done. =========="
