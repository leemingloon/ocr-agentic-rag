"""
Document preprocessing for OCR (binarisation, deskew, DPI normalisation).

Used before passing images to Tesseract or other engines to improve
recognition on financial documents.
"""
from .document_preprocessor import (
    binarise_otsu,
    deskew_image,
    normalise_dpi,
    preprocess_for_ocr,
)

__all__ = [
    "binarise_otsu",
    "deskew_image",
    "normalise_dpi",
    "preprocess_for_ocr",
]
