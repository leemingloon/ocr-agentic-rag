#!/usr/bin/env bash
# Out-of-sample RAG eval: run FinQA test and TATQA dev in sequence, recording terminal output.
# Ensures exactly 200 samples evaluated per split: counts existing samples in *_samples.json
# and runs only (200 - existing) so the total after the run is 200.
# Output is written incrementally via tee so that if a run crashes, the log file still contains
# everything printed up to that point (no buffered loss).
#
# Usage: from repo root
#   bash scripts/run_rag_out_of_sample.sh
#
# Code trace when FinQA test already has 200 samples (FINQA_TO_RUN=0):
#   - Script skips the FinQA eval_runner call (no Python run for FinQA).
#   - TATQA dev runs as usual (e.g. TATQA_TO_RUN=200 or 199).
#   - Exit codes: FINQA_RC=0 (skipped), TATQA_RC from eval_runner.
# If you did not skip: eval_runner with --max_split 0 would break at first loop
#   (len(per_sample_rows) >= 0) and return using existing split avg files; no crash.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FINQA_SAMPLES="data/proof/rag/finqa/test/finqa_test_samples.json"
TATQA_SAMPLES="data/proof/rag/tatqa/dev/tatqa_dev_samples.json"
FINQA_OUT="data/proof/rag/finqa/test/terminal_output.txt"
TATQA_OUT="data/proof/rag/tatqa/dev/terminal_output.txt"
TARGET_PER_SPLIT=200

# Count existing samples (0 if file missing or invalid). Prints one integer to stdout.
count_samples() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  python -c '
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    print(len(data) if isinstance(data, list) else 0)
except Exception:
    print(0)
' "$path"
}

mkdir -p "$(dirname "$FINQA_OUT")" "$(dirname "$TATQA_OUT")"

# Unbuffered Python output so tee receives lines immediately; reduces risk of losing output on crash.
export PYTHONUNBUFFERED=1

run_one() {
  local label="$1"
  local outfile="$2"
  shift 2
  echo "=== $label ==="
  # 2>&1: merge stderr into stdout so everything is captured
  # tee: write to file and pass through to terminal; writes line-by-line so file is safe on crash
  "$@" 2>&1 | tee "$outfile"
  return "${PIPESTATUS[0]}"
}

# Compute how many new samples to run so total per split equals TARGET_PER_SPLIT
FINQA_EXISTING=$(count_samples "$FINQA_SAMPLES")
FINQA_TO_RUN=$((TARGET_PER_SPLIT - FINQA_EXISTING))
[[ $FINQA_TO_RUN -lt 0 ]] && FINQA_TO_RUN=0

TATQA_EXISTING=$(count_samples "$TATQA_SAMPLES")
TATQA_TO_RUN=$((TARGET_PER_SPLIT - TATQA_EXISTING))
[[ $TATQA_TO_RUN -lt 0 ]] && TATQA_TO_RUN=0

echo "FinQA test: existing=$FINQA_EXISTING -> will run up to $FINQA_TO_RUN (target total=$TARGET_PER_SPLIT)"
echo "TATQA dev:  existing=$TATQA_EXISTING -> will run up to $TATQA_TO_RUN (target total=$TARGET_PER_SPLIT)"
echo ""

# Run both in sequence; do not use set -e so the second run executes even if the first fails.
# When target already reached (TO_RUN=0), skip that invocation so we do not call eval_runner with --max_split 0
# (eval_runner would break immediately and still work, but skipping is faster and avoids any edge case).
if [[ $FINQA_TO_RUN -gt 0 ]]; then
  run_one "Running FinQA test" "$FINQA_OUT" python eval_runner.py --category rag --dataset FinQA --max_split "$FINQA_TO_RUN" --max_category "$FINQA_TO_RUN" --debug --export_predictions_txt --split test
  FINQA_RC=$?
else
  echo "=== FinQA test (skipped: already $FINQA_EXISTING samples, target=$TARGET_PER_SPLIT) ==="
  FINQA_RC=0
fi

if [[ $TATQA_TO_RUN -gt 0 ]]; then
  run_one "Running TATQA dev" "$TATQA_OUT" python eval_runner.py --category rag --dataset TATQA --max_split "$TATQA_TO_RUN" --max_category "$TATQA_TO_RUN" --debug --export_predictions_txt --split dev
  TATQA_RC=$?
else
  echo "=== TATQA dev (skipped: already $TATQA_EXISTING samples, target=$TARGET_PER_SPLIT) ==="
  TATQA_RC=0
fi

echo ""
echo "=== Summary ==="
echo "FinQA test: exit_code=$FINQA_RC  output=$FINQA_OUT"
echo "TATQA dev: exit_code=$TATQA_RC  output=$TATQA_OUT"
exit $((FINQA_RC != 0 || TATQA_RC != 0 ? 1 : 0))
