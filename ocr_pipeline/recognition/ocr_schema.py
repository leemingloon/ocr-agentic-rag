"""
Standardised OCR output schema: model-agnostic {text, bbox, confidence} per word/line.

Table reconstruction and metric calculation happen downstream of OCR,
independent of which engine (Tesseract, PaddleOCR, etc.) produced the raw output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict


@dataclass
class OCRWord:
    """Single word or token from OCR with bounding box and confidence."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float  # 0–100

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "bbox": list(self.bbox), "confidence": self.confidence}


@dataclass
class OCRLine:
    """Line of text (one or more words) with aggregate bbox and confidence."""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    words: List[OCRWord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class StandardOCROutput:
    """
    Model-agnostic OCR result: flat list of words and optional lines/table.

    Every engine should produce at least:
    - words: list of {text, bbox, confidence}
    - full_text: concatenated text (for backward compatibility)
    - low_confidence_words: words with confidence < threshold (e.g. 60) for human review
    """
    words: List[OCRWord]
    full_text: str
    confidence: float  # overall 0–100
    low_confidence_words: List[OCRWord] = field(default_factory=list)
    lines: List[OCRLine] = field(default_factory=list)
    table_rows: List[List[str]] = field(default_factory=list)  # optional: reconstructed table

    def to_dict(self) -> Dict[str, Any]:
        return {
            "words": [w.to_dict() for w in self.words],
            "full_text": self.full_text,
            "confidence": self.confidence,
            "low_confidence_words": [w.to_dict() for w in self.low_confidence_words],
            "lines": [ln.to_dict() for ln in self.lines],
            "table_rows": self.table_rows,
        }
