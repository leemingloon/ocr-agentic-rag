#!/usr/bin/env bash
# Run RAG eval one sample at a time per dataset. For each dataset:
#   - After each run, check the *latest evaluated sample*: if all its metrics are 1, run again; otherwise stop.
# Default: (1) FinQA until first failure, then (2) TATQA until first failure.
#
# Milestones: pauses at 100, 150, 200 samples (first time only).
# State file: .eval_milestones_reached — delete or edit to reset milestones.
# Re-run the script after a pause to continue from where it stopped.
#
# Usage: from repo root
#   bash scripts/run_rag_eval_until_fail.sh              # run FinQA then TATQA (default)
#   bash scripts/run_rag_eval_until_fail.sh --dataset finqa   # run only FinQA
#   bash scripts/run_rag_eval_until_fail.sh --dataset tatqa  # run only TATQA

set -e
LOG=".rag_eval_run.log"
MILESTONE_FILE=".eval_milestones_reached"
MILESTONES=(100 150 200)

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

check_milestone() {
  local avg_json="$1"
  if [[ ! -f "$avg_json" ]]; then return; fi
  local count
  local tmp_py
  tmp_py=$(mktemp)
  cat << 'PYEOF' > "$tmp_py"
import json
import sys
try:
    with open(sys.argv[1], encoding='utf-8') as f:
        d = json.load(f)
    print(d.get('sample_count', 0))
except Exception:
    print(0)
PYEOF
  count=$(python "$tmp_py" "$avg_json")
  rm -f "$tmp_py"
  for milestone in "${MILESTONES[@]}"; do
    if [[ "$count" -ge "$milestone" ]]; then
      if ! grep -qx "$milestone" "$MILESTONE_FILE" 2>/dev/null; then
        echo "$milestone" >> "$MILESTONE_FILE"
        echo ""
        echo "=========================================="
        echo "  MILESTONE REACHED: $count samples evaluated (>= $milestone)"
        echo "  Pausing for review."
        echo "  Re-run the script to continue."
        echo "=========================================="
        exit 0
      fi
    fi
  done
}

# Run one dataset until the latest evaluated sample has any metric != 1. Returns 0 when should stop, 1 when should continue.
# Stops when: (1) latest sample has any metric != 1, or (2) no progress (run evaluated 0 new samples) to avoid infinite loop.
run_dataset_until_fail() {
  local DATASET="$1"
  # Proof samples path: data/proof/rag/<dataset_lower>/<split>/<dataset>_<split>_samples.json
  # For TATQA we may have train or dev; the checker will try both if needed.
  local SAMPLES_DIR="data/proof/rag/$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
  local CMD="python eval_runner.py --category rag --dataset $DATASET --max_split 1 --max_category 1 --debug --export_predictions_txt"
  local AVG_JSON="${SAMPLES_DIR}/$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')_avg.json"

  while true; do
    echo "=== Running: $CMD ==="
    $CMD 2>&1 | tee "$LOG"
    check_milestone "$AVG_JSON"
    CONTINUE=$(python -c "
import json
import re
import sys
import glob
import os

dataset_name = '''$DATASET'''
samples_dir = '''$SAMPLES_DIR'''
log_path = '''$LOG'''

# Determine the *just-evaluated* sample from this run so we base CONTINUE/STOP on the right sample
# (avoids wrong decision when predictions file \"last block\" is not the one we just ran, e.g. after in-place update).
with open(log_path, encoding='utf-8', errors='replace') as f:
    log = f.read()
just_evaluated_id = None
m = re.search(r'\[EVAL_PROGRESS\] new_samples=(\d+) last_sample_id=(\S*)', log)
if m:
    just_evaluated_id = (m.group(2).strip() or None) if m.group(2) else None
if not just_evaluated_id:
    rag_m = list(re.finditer(r\"\\[DEBUG\\] RAG query corpus_id='([^']+)'\", log))
    if rag_m:
        just_evaluated_id = rag_m[-1].group(1)

# No sample was evaluated in this run: do not use \"latest has 0\" message; stop as no progress.
if not just_evaluated_id:
    print('STOP_NO_SAMPLE')
    sys.exit(0)

# Look up that sample's metrics in the samples JSON (authoritative; not \"last block\" of predictions file).
metrics = None
candidates = []
if os.path.isdir(samples_dir):
    candidates = glob.glob(os.path.join(samples_dir, '*', '*_samples.json'))
if not candidates:
    prefix = dataset_name.lower().replace('-', '')
    single = os.path.join(samples_dir, 'train', prefix + '_train_samples.json')
    if os.path.isfile(single):
        candidates = [single]
for path in candidates:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue
    if isinstance(data, list):
        row = next((r for r in data if str(r.get('sample_id')) == just_evaluated_id or str(r.get('ground_truth', {}).get('corpus_id')) == just_evaluated_id), None)
        if row is not None:
            metrics = (row or {}).get('metrics') or {}
            break
if metrics is None:
    print('STOP', file=sys.stderr)
    sys.exit(1)

# If that sample had prediction_error, continue so next run skips it and advances.
if row and row.get('prediction_error'):
    print('CONTINUE')
    sys.exit(0)

required = {'program_accuracy', 'numerical_exact_match', 'exact_match', 'f1'}
optional = {'numerical_near_match'}
all_one = all(metrics.get(k) == 1.0 or metrics.get(k) == 1 for k in required)
if optional & set(metrics):
    all_one = all_one and all(metrics.get(k) == 1.0 or metrics.get(k) == 1 for k in optional if k in metrics)
print('CONTINUE' if all_one else 'STOP')
")
    if [[ "$CONTINUE" == "STOP_NO_SAMPLE" ]]; then
      echo "No new samples evaluated or could not determine last sample. Stopping $DATASET."
      return 0
    fi
    if [[ "$CONTINUE" != "CONTINUE" ]]; then
      echo "Latest evaluated sample does not have all metrics 1. Stopping $DATASET."
      return 0
    fi
    # No progress: stop only when this run actually evaluated 0 new samples (so we do not stop when last sample was accurate and we should continue).
    NEW_SAMPLES=$(grep -o '\[EVAL_PROGRESS\] new_samples=[0-9]*' "$LOG" 2>/dev/null | tail -1 | sed 's/.*new_samples=//')
    if [[ -n "$NEW_SAMPLES" && "$NEW_SAMPLES" -eq 0 ]]; then
      echo "No new samples evaluated (new_samples=0). All samples done. Stopping $DATASET."
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
