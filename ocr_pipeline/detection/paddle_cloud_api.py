"""
Paddle full-page OCR API for cloud notebooks when GitHub paddleocr_detector.py lags behind local.

Used by eval_runner warmup and hybrid_ocr.run_paddle_full_ocr imports.
"""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

try:
    from ..compat.paddle_langchain_shim import install_paddle_langchain_shim

    install_paddle_langchain_shim()
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except Exception as e:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None  # type: ignore[misc, assignment]
    _PADDLE_IMPORT_ERR = e
else:
    _PADDLE_IMPORT_ERR = None


def _paddle_use_gpu() -> bool:
    return os.environ.get("OCR_USE_GPU", "").strip().lower() in ("1", "true", "yes")


def build_native_paddle_ocr(*, show_log: bool = False):
    if not PADDLEOCR_AVAILABLE or PaddleOCR is None:
        raise RuntimeError(f"PaddleOCR is not available: {_PADDLE_IMPORT_ERR}")
    from ocr_pipeline.paddle_gpu_check import build_paddle_ocr_engine

    return build_paddle_ocr_engine(show_log=show_log)


_CACHED_NATIVE_FULL_PADDLE: Any = None


def _paddle_lines_reading_order(ocr_lines: list | None) -> list:
    if not ocr_lines:
        return []
    keyed: list[tuple[int, float, float, object]] = []
    for line in ocr_lines:
        if line is None or len(line) < 2:
            continue
        try:
            pts = np.asarray(line[0], dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            y_ctr = float(pts[:, 1].mean())
            x_left = float(pts[:, 0].min())
        except Exception:
            continue
        row_band = int(round(y_ctr / 15.0))
        keyed.append((row_band, x_left, y_ctr, line))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in keyed]


def _parse_paddle_result(result: Any) -> tuple[str, float, int]:
    lines_raw: list = []
    if result is None:
        return "", 0.0, 0
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], list):
            lines_raw = result[0] or []
        elif result and isinstance(result[0], (list, tuple)) and len(result[0]) >= 2:
            if isinstance(result[0][0], (list, tuple, np.ndarray)):
                lines_raw = result
            else:
                lines_raw = result[0] if isinstance(result[0], list) else result
    text_parts: list[str] = []
    conf_sum, conf_n = 0.0, 0
    for line in _paddle_lines_reading_order(lines_raw):
        if not line or len(line) < 2:
            continue
        rec = line[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 1:
            t = str(rec[0]).strip()
            if t:
                text_parts.append(t)
            if len(rec) >= 2:
                try:
                    conf_sum += float(rec[1])
                    conf_n += 1
                except (TypeError, ValueError):
                    pass
    text = "\n".join(text_parts)
    confidence = (conf_sum / conf_n * 100.0) if conf_n else 85.0
    return text, confidence, len(text_parts)


def get_or_build_native_paddle_ocr(*, show_log: bool = False) -> Any:
    global _CACHED_NATIVE_FULL_PADDLE
    if _CACHED_NATIVE_FULL_PADDLE is not None:
        return _CACHED_NATIVE_FULL_PADDLE
    _CACHED_NATIVE_FULL_PADDLE = build_native_paddle_ocr(show_log=show_log)
    return _CACHED_NATIVE_FULL_PADDLE


def run_paddle_full_ocr(
    image: np.ndarray,
    paddle_ocr: Any | None = None,
) -> tuple[str, float, int]:
    if paddle_ocr is None:
        paddle_ocr = get_or_build_native_paddle_ocr(show_log=False)
    try:
        from ocr_pipeline.ocr_eval_config import paddle_min_side, paddle_use_angle_cls
    except ImportError:
        from ..ocr_eval_config import paddle_min_side, paddle_use_angle_cls

    work = image
    if len(work.shape) == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    h, w = work.shape[:2]
    target_min = paddle_min_side()
    min_side = min(h, w)
    if min_side < target_min:
        scale = target_min / min_side
        work = cv2.resize(
            work,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    use_cls = paddle_use_angle_cls()
    result = None
    ocr_kw_attempts: list[dict] = []
    if use_cls:
        ocr_kw_attempts.append({"cls": True})
    ocr_kw_attempts.extend([{"det": True, "rec": True, "cls": use_cls}, {}])
    for kw in ocr_kw_attempts:
        try:
            result = paddle_ocr.ocr(work, **kw)
            if result:
                break
        except TypeError:
            try:
                result = paddle_ocr.ocr(work)
                if result:
                    break
            except Exception:
                continue
        except Exception:
            continue
    if result is None:
        legacy_calls = []
        if use_cls:
            legacy_calls.append(lambda: paddle_ocr.ocr(work, cls=True))
        legacy_calls.extend(
            [
                lambda: paddle_ocr.ocr(work, det=True, rec=True, cls=True),
                lambda: paddle_ocr.ocr(work),
            ]
        )
        for call in legacy_calls:
            try:
                result = call()
                if result:
                    break
            except TypeError:
                continue
            except Exception:
                continue
    if result is None and hasattr(paddle_ocr, "predict"):
        try:
            result = list(paddle_ocr.predict(work))
        except Exception:
            result = None
    return _parse_paddle_result(result)
