"""
Text Recognition Module

Hybrid recognition using Tesseract and confidence-based routing.
StandardOCROutput: model-agnostic {text, bbox, confidence} per word for downstream table/metrics.
"""
from .tesseract_ocr import TesseractOCR, OCRResult
from .hybrid_ocr import HybridOCR
from .ocr_schema import StandardOCROutput, OCRWord, OCRLine

__all__ = ["TesseractOCR", "HybridOCR", "OCRResult", "StandardOCROutput", "OCRWord", "OCRLine"]