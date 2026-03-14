#!/usr/bin/env bash
# Out-of-sample RAG eval: run FinQA test and TATQA dev in sequence, recording terminal output.
# Output is written incrementally via tee so that if a run crashes, the log file still contains
# everything printed up to that point (no buffered loss).
#
# Usage: from repo root
#   bash scripts/run_rag_out_of_sample.sh

set -u

FINQA_OUT="data/proof/rag/finqa/test/terminal_output.txt"
TATQA_OUT="data/proof/rag/tatqa/dev/terminal_output.txt"

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

# Run both in sequence; do not use set -e so the second run executes even if the first fails
run_one "Running FinQA test" "$FINQA_OUT" python eval_runner.py --category rag --dataset FinQA --max_split 200 --max_category 200 --debug --export_predictions_txt --split test
FINQA_RC=$?

run_one "Running TATQA dev" "$TATQA_OUT" python eval_runner.py --category rag --dataset TATQA --max_split 200 --max_category 200 --debug --export_predictions_txt --split dev
TATQA_RC=$?

echo ""
echo "=== Summary ==="
echo "FinQA test: exit_code=$FINQA_RC  output=$FINQA_OUT"
echo "TATQA dev: exit_code=$TATQA_RC  output=$TATQA_OUT"
exit $((FINQA_RC != 0 || TATQA_RC != 0 ? 1 : 0))
