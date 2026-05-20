#!/usr/bin/env python3
"""
Pre-download PaddleOCR model weights so the first run inside eval_runner or
other scripts does not time out in restricted environments.

Run once (e.g. after pip install paddleocr + paddlepaddle):
  python scripts/pre_download_paddleocr_models.py

Set PADDLEOCR_SHOW_LOG=1 to see download progress. Uses build_native_paddle_ocr
so both legacy (2.x kwargs) and current (3.x) PaddleOCR APIs work.
"""
import os
import sys
from pathlib import Path

# Show log by default for this script so user sees download progress
os.environ.setdefault("PADDLEOCR_SHOW_LOG", "1")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib.util

    det_path = root / "ocr_pipeline" / "detection" / "paddleocr_detector.py"
    spec = importlib.util.spec_from_file_location("_paddleocr_detector_dl", det_path)
    if spec is None or spec.loader is None:
        print("Could not load paddleocr_detector from", det_path)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not getattr(mod, "PADDLEOCR_AVAILABLE", False):
        print("PaddleOCR not installed or failed to import. Run: python scripts/diagnose_paddleocr_env.py")
        sys.exit(1)
    show_log = os.environ.get("PADDLEOCR_SHOW_LOG", "").lower() in ("1", "true", "yes")
    print("Instantiating PaddleOCR (first run downloads models ~150MB)...")
    try:
        mod.build_native_paddle_ocr(show_log=show_log)
        print("OK: PaddleOCR models ready. HybridOCR can use native PaddleOCR.")
    except Exception as e:
        print("PaddleOCR initialization failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
