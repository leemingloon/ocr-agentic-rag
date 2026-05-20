#!/usr/bin/env python3
"""Merge cloud OCR proof bundle into data/proof/ocr for local agent iteration.

Kaggle (after kernels output):
  kaggle kernels output leemingloon/ocr-funsd-sroie-eval-kaggle \\
    -p notebooks/kaggle-kernels/ocr_funsd_sroie_eval_kaggle/kaggle_outputs
  python scripts/sync_kaggle_ocr_proof.py

Colab (after downloading ocr_proof_bundle.zip):
  unzip -o ~/Downloads/ocr_proof_bundle.zip -d /tmp/ocr_colab_out
  OCR_PROOF_SRC=/tmp/ocr_colab_out python scripts/sync_kaggle_ocr_proof.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import os

REPO = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OCR_PROOF_SRC", "")).resolve() if os.environ.get("OCR_PROOF_SRC") else (
    REPO / "notebooks" / "kaggle-kernels" / "ocr_funsd_sroie_eval_kaggle" / "kaggle_outputs"
)
DST = REPO / "data" / "proof" / "ocr"
BUNDLE = SRC / "ocr_proof_bundle"


def main() -> int:
    if not SRC.is_dir():
        print(
            f"Missing {SRC}; Kaggle: run kaggle kernels output first. "
            "Colab: set OCR_PROOF_SRC to the folder that contains ocr_proof_bundle/."
        )
        return 1
    if BUNDLE.is_dir():
        for item in BUNDLE.iterdir():
            target = DST / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        print(f"Synced bundle -> {DST}")
    summary = SRC / "ocr_eval_summary.json"
    if summary.is_file():
        print(json.dumps(json.loads(summary.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
