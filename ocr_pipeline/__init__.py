"""
OCR Pipeline Module

Import submodules directly, e.g.:
  from ocr_pipeline.recognition.hybrid_ocr import HybridOCR
  from ocr_pipeline.detection.paddleocr_detector import get_or_build_native_paddle_ocr

Avoid eager imports here so GPU OCR eval does not require anthropic / vision extras.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
