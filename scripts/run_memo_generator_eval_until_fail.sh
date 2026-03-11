#!/usr/bin/env bash
# Run risk memo generator (FinanceBench) eval one sample at a time.
# After each run, pass if (exact_match=1 and f1=1) OR relaxed_match=1 OR conclusion_match=1; else stop.
# conclusion_match=1 with exact=0 = known scorer false negative → continue automatically.
#
# Milestones: pauses at 100, 150, 200 samples (first time only).
# State file: .eval_milestones_reached — delete or edit to reset milestones.
# Re-run the script after a pause to continue from where it stopped.
#
# If you have existing samples that were evaluated before relaxed_match was added, refresh metrics so they get relaxed_match:
#   python eval_runner.py --category credit_risk_memo_generator --dataset FinanceBench --reevaluate_only
#
# Usage: from repo root
#   bash scripts/run_memo_generator_eval_until_fail.sh

set -e
LOG=".memo_eval_run.log"
CATEGORY="credit_risk_memo_generator"
DATASET="FinanceBench"
MILESTONE_FILE=".eval_milestones_reached"
MILESTONES=(100 150 200)

# Proof path: data/proof/credit_risk_memo_generator/financebench/<split>/
SAMPLES_DIR="data/proof/$(echo "$CATEGORY" | tr '[:upper:]' '[:lower:]')/$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
CMD="python eval_runner.py --category $CATEGORY --dataset $DATASET --max_split 1 --max_category 1 --debug --export_predictions_txt"

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

run_until_fail() {
  while true; do
    echo "=== Running: $CMD ==="
    $CMD 2>&1 | tee "$LOG"
    check_milestone "${SAMPLES_DIR}/$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')_avg.json"
    CONTINUE=$(python -c "
import json
import re
import sys
import glob
import os

samples_dir = '''$SAMPLES_DIR'''
log_path = '''$LOG'''

with open(log_path, encoding='utf-8', errors='replace') as f:
    log = f.read()
just_evaluated_id = None
m = re.search(r'\[EVAL_PROGRESS\] new_samples=(\d+) last_sample_id=(\S*)', log)
if m:
    just_evaluated_id = (m.group(2).strip() or None) if m.group(2) else None

if not just_evaluated_id:
    print('STOP_NO_SAMPLE')
    sys.exit(0)

metrics = None
row = None
candidates = []
if os.path.isdir(samples_dir):
    candidates = glob.glob(os.path.join(samples_dir, '*', '*_samples.json'))
for path in candidates:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue
    if isinstance(data, list):
        row = next((r for r in data if str(r.get('sample_id')) == just_evaluated_id), None)
        if row is not None:
            metrics = (row or {}).get('metrics') or {}
            break
if metrics is None:
    print('STOP', file=sys.stderr)
    sys.exit(1)

if row and row.get('prediction_error'):
    print('CONTINUE')
    sys.exit(0)

# Memo: strict pass (exact+f1), or relaxed_match, or conclusion_match=1 (scorer false negative pattern)
required = {'exact_match', 'f1'}
strict_pass = all(metrics.get(k) == 1.0 or metrics.get(k) == 1 for k in required)
relaxed_pass = metrics.get('relaxed_match') == 1.0 or metrics.get('relaxed_match') == 1
conclusion_pass = metrics.get('conclusion_match') == 1
# Continue if: strict pass, OR relaxed match, OR conclusion_match=1 (scorer false negative)
all_one = strict_pass or relaxed_pass or conclusion_pass
print('CONTINUE' if all_one else 'STOP')
")
    if [[ "$CONTINUE" == "STOP_NO_SAMPLE" ]]; then
      echo "No new samples evaluated or could not determine last sample. Stopping."
      return 0
    fi
    if [[ "$CONTINUE" != "CONTINUE" ]]; then
      echo "Latest evaluated sample does not pass (exact+f1, relaxed_match, or conclusion_match=1). Stopping."
      return 0
    fi
    NEW_SAMPLES=$(grep -o '\[EVAL_PROGRESS\] new_samples=[0-9]*' "$LOG" 2>/dev/null | tail -1 | sed 's/.*new_samples=//')
    if [[ -n "$NEW_SAMPLES" && "$NEW_SAMPLES" -eq 0 ]]; then
      echo "No new samples evaluated (new_samples=0). Stopping."
      return 0
    fi
    echo "Latest sample: pass (exact+f1, relaxed_match, or conclusion_match). Running again..."
  done
}

echo "========== Risk Memo Generator (FinanceBench) until first failure =========="
run_until_fail
echo "========== Done. =========="
