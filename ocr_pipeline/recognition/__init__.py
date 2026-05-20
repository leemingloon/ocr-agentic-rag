"""
Text Recognition Module

Import submodules directly, e.g. HybridOCR from hybrid_ocr.
Lazy __getattr__ avoids pulling vision_ocr/anthropic on package import.
"""
from __future__ import annotations

from .tesseract_ocr import TesseractOCR, OCRResult
from .ocr_schema import StandardOCROutput, OCRWord, OCRLine

__all__ = [
    "TesseractOCR",
    "HybridOCR",
    "OCRResult",
    "StandardOCROutput",
    "OCRWord",
    "OCRLine",
]


def __getattr__(name: str):
    if name == "HybridOCR":
        from .hybrid_ocr import HybridOCR

        return HybridOCR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
