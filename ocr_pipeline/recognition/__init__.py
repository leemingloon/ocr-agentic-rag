"""
Text Recognition Module

Hybrid recognition using Tesseract and confidence-based routing
"""

from .tesseract_ocr import TesseractOCR
from .hybrid_ocr import HybridOCR

__all__ = ["TesseractOCR", "HybridOCR"]