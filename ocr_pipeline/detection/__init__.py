# ocr_pipeline/detection/__init__.py

"""
Text detection module
"""

from .classical_detector import ClassicalDetector

__all__ = [
    "ClassicalDetector",
    "PaddleOCRDetector",
    "DetectionRouter",
]


def __getattr__(name: str):
    if name == "DetectionRouter":
        from .detection_router import DetectionRouter

        return DetectionRouter
    if name == "PaddleOCRDetector":
        try:
            from .paddleocr_detector import PaddleOCRDetector

            return PaddleOCRDetector
        except Exception as e:
            print(f"⚠ PaddleOCR detector unavailable: {e}")

            class _UnavailablePaddleOCRDetector:
                def __init__(self, *args, **kwargs):
                    self.mode = "unavailable"

                def detect(self, image):
                    return []

            return _UnavailablePaddleOCRDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")