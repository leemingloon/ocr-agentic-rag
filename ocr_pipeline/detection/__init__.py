# ocr_pipeline/detection/__init__.py

"""
Text detection module
"""

from .classical_detector import ClassicalDetector
from .detection_router import DetectionRouter

# Make PaddleOCR import optional
try:
    from .paddleocr_detector import PaddleOCRDetector
except Exception as e:
    print(f"⚠ PaddleOCR detector unavailable: {e}")
    # Create a dummy class
    class PaddleOCRDetector:
        def __init__(self, *args, **kwargs):
            self.mode = "unavailable"
        
        def detect(self, image):
            return []

__all__ = [
    'ClassicalDetector',
    'PaddleOCRDetector',
    'DetectionRouter',
]