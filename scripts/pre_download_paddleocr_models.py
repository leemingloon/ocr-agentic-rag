#!/usr/bin/env python3
"""
Pre-download PaddleOCR model weights so the first run inside eval_runner or
other scripts does not time out in restricted environments.

Run once (e.g. after pip install paddleocr):
  python scripts/pre_download_paddleocr_models.py

Set PADDLEOCR_SHOW_LOG=1 to see download progress. Uses the same instantiation
as ocr_pipeline.detection.paddleocr_detector (PaddleOCR with use_angle_cls=True, lang='en').
"""
import os
import sys

# Show log by default for this script so user sees download progress
os.environ.setdefault("PADDLEOCR_SHOW_LOG", "1")

def main():
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        print("PaddleOCR not installed:", e)
        print("  pip install paddleocr==2.6.1.3 paddlepaddle==2.5.2")
        sys.exit(1)
    print("Instantiating PaddleOCR (first run downloads models ~150MB)...")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=True)
        print("✓ PaddleOCR models ready. Eval runner and HybridOCR can use native PaddleOCR.")
    except Exception as e:
        print("✗ PaddleOCR initialization failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
