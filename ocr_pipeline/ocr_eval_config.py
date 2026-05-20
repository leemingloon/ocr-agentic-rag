"""OCR eval runtime flags (env). Used by eval_runner and Kaggle notebook."""

from __future__ import annotations

import os


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def ocr_use_gpu() -> bool:
    return _truthy("OCR_USE_GPU")


def ocr_skip_tesseract_ensemble() -> bool:
    """Skip second Tesseract pass (large CPU saving; Paddle-only)."""
    return _truthy("OCR_SKIP_TESSERACT_ENSEMBLE")


def ocr_fast_mode() -> bool:
    """Faster Paddle: less upscale, optional no angle cls, lighter preprocess."""
    return _truthy("OCR_FAST") or (ocr_skip_tesseract_ensemble() and ocr_use_gpu())


def ocr_cpu_workers() -> int:
    """Process-pool workers for OCR (CPU only; keep 0 when OCR_USE_GPU=1)."""
    if ocr_use_gpu():
        return 0
    try:
        return max(0, int(os.environ.get("OCR_CPU_WORKERS", "0")))
    except ValueError:
        return 0


def ocr_prefetch_enabled() -> bool:
    return _truthy("OCR_PREFETCH", "1")


def paddle_use_angle_cls() -> bool:
    if ocr_fast_mode():
        return _truthy("OCR_PADDLE_USE_CLS", "0")
    return _truthy("OCR_PADDLE_USE_CLS", "1")


def paddle_min_side() -> int:
    if ocr_fast_mode():
        try:
            return max(480, int(os.environ.get("OCR_PADDLE_MIN_SIDE", "720")))
        except ValueError:
            return 720
    try:
        return max(480, int(os.environ.get("OCR_PADDLE_MIN_SIDE", "900")))
    except ValueError:
        return 900


def paddle_rec_batch_num() -> int:
    """PaddleOCR rec_batch_num: larger uses more GPU VRAM, faster on dense pages (SROIE)."""
    raw = os.environ.get("OCR_REC_BATCH_NUM", "").strip()
    if raw:
        try:
            return max(1, min(96, int(raw)))
        except ValueError:
            pass
    if ocr_use_gpu() and ocr_fast_mode():
        return 16
    if ocr_use_gpu():
        return 8
    return 6
