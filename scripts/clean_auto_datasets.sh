#!/usr/bin/env bash

# ==========================================
# clean_auto_datasets.sh
# Removes ONLY auto-downloadable datasets
# Leaves manual datasets untouched
# ==========================================

DATA_DIR="data"

# -------------------------
# OCR
# -------------------------
AUTO_OCR=(
  "OmniDocBench"
)

# -------------------------
# Vision
# -------------------------
AUTO_VISION=(
  "ChartQA"
  "AI2D"
  "VisualMRC"
)

# -------------------------
# RAG
# -------------------------
AUTO_RAG=(
  "HotpotQA"
  "FinQA"
)

# -------------------------
# Credit Risk
# -------------------------
AUTO_CREDIT=(
  "LendingClub"
  "FiQA"
  "FinanceBench"
  "ECTSum"
)

delete_category() {
  CATEGORY=$1
  shift
  DATASETS=("$@")

  for dataset in "${DATASETS[@]}"; do
    TARGET="${DATA_DIR}/${CATEGORY}/${dataset}"

    if [ -d "$TARGET" ]; then
      echo "Deleting $TARGET ..."
      rm -rf "$TARGET"
    else
      echo "Skipping $TARGET (not found)"
    fi
  done
}

echo "======================================="
echo "Cleaning auto-downloadable datasets..."
echo "======================================="

delete_category "ocr" "${AUTO_OCR[@]}"
delete_category "vision" "${AUTO_VISION[@]}"
delete_category "rag" "${AUTO_RAG[@]}"
delete_category "credit_risk" "${AUTO_CREDIT[@]}"

echo "======================================="
echo "Cleanup complete."
echo "======================================="
