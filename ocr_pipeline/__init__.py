"""
OCR Pipeline Module

Hybrid OCR system with 3-tier detection and confidence-based recognition:

Detection (Tier-based):
- Tier 1 (65%): Template cache
- Tier 2 (25%): Classical detection (OpenCV)
- Tier 3 (10%): PaddleOCR detection (DL)

Recognition (Confidence-based):
- Tesseract (70%): High confidence
- PaddleOCR (25%): Medium confidence
- Human Review (5%): Low confidence
"""

from .quality_assessment import ImageQualityAssessor
from .template_detector import TemplateDetector
from .detection import ClassicalDetector, PaddleOCRDetector, DetectionRouter
from .recognition import TesseractOCR, HybridOCR

__version__ = "1.0.0"

__all__ = [
    "ImageQualityAssessor",
    "TemplateDetector",
    "ClassicalDetector",
    "PaddleOCRDetector",
    "DetectionRouter",
    "TesseractOCR",
    "HybridOCR",
]