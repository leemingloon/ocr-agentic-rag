"""
Document preprocessing for OCR: binarisation, deskew, DPI normalisation, morphology.

Preprocessing matters enormously for Tesseract on financial documents.
- Binarisation (Otsu) gives clean black/white for character recognition.
- Deskewing corrects slight rotations from scanning.
- 300 DPI minimum ensures Tesseract has enough resolution.
- Morphological cleanup (closing/opening) connects broken strokes and removes noise;
  especially important for receipts (thermal, faded) and invoices.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple

# Default target DPI for OCR (Tesseract works best at 300+)
TARGET_DPI = 300
# Assumed DPI when unknown (e.g. in-memory image); will scale to TARGET_DPI
ASSUMED_DPI = 150
# Morphology kernel size for receipt/invoice cleanup (small = connect broken strokes, remove speckle)
MORPH_CLOSE_KERNEL = (2, 2)
MORPH_OPEN_KERNEL = (1, 1)


def binarise_otsu(image: np.ndarray) -> np.ndarray:
    """
    Binarise image using Otsu's threshold. Works well for financial documents.

    Args:
        image: Grayscale or BGR image.

    Returns:
        Binary image (0 or 255).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew image by detecting dominant text angle and rotating.
    Uses the minimum-area bounding box of the binary content to estimate angle.

    Args:
        image: Grayscale or binary image.

    Returns:
        Deskewed image (same type).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary == 0))
    if coords.size < 100:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def normalise_dpi(
    image: np.ndarray,
    target_dpi: int = TARGET_DPI,
    assumed_dpi: int = ASSUMED_DPI,
) -> np.ndarray:
    """
    Scale image so effective DPI is target_dpi (e.g. 300 for OCR).
    Use when source DPI is unknown (e.g. in-memory image from PDF).

    Args:
        image: Input image.
        target_dpi: Desired effective DPI.
        assumed_dpi: Assumed current DPI if unknown.

    Returns:
        Resized image (same aspect ratio).
    """
    if assumed_dpi <= 0 or target_dpi <= 0:
        return image
    scale = target_dpi / assumed_dpi
    if abs(scale - 1.0) < 0.05:
        return image
    h, w = image.shape[:2]
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if new_w < 10 or new_h < 10:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def morphology_cleanup(
    image: np.ndarray,
    *,
    close_kernel: Tuple[int, int] = MORPH_CLOSE_KERNEL,
    open_kernel: Tuple[int, int] = MORPH_OPEN_KERNEL,
    apply_close: bool = True,
    apply_open: bool = True,
) -> np.ndarray:
    """
    Morphological cleanup for receipts/invoices: connect broken character strokes and remove small noise.

    - Closing (dilation then erosion) with a small kernel fills gaps in characters (e.g. thermal receipts).
    - Opening (erosion then dilation) with a tiny kernel removes isolated speckle without harming text.

    Use after binarisation. Applies to both receipts and invoices; especially helpful for low-quality scans.
    """
    if image.size == 0:
        return image
    out = image.copy()
    if len(out.shape) == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    # Otsu gives text=0 (black), background=255. OpenCV morphology treats 255 as foreground;
    # invert so text becomes 255, then we close (connect text), open (remove speckle), then invert back.
    inverted = np.median(out) > 127
    if inverted:
        out = cv2.bitwise_not(out)
    if apply_close and close_kernel[0] > 0 and close_kernel[1] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if apply_open and open_kernel[0] > 0 and open_kernel[1] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if inverted:
        out = cv2.bitwise_not(out)
    return out


def preprocess_for_ocr(
    image: np.ndarray,
    *,
    binarise: bool = True,
    deskew: bool = True,
    normalise_dpi_value: bool = True,
    morphology_cleanup_enabled: bool = True,
    target_dpi: int = TARGET_DPI,
    assumed_dpi: int = ASSUMED_DPI,
) -> np.ndarray:
    """
    Full preprocessing pipeline for OCR: normalise DPI → deskew → binarise → morphology.

    Order: DPI first (so deskew and binarise work on correct scale), then
    deskew, then binarise (Otsu), then optional morphological cleanup (for receipts/invoices).

    Args:
        image: BGR or grayscale image.
        binarise: Apply Otsu binarisation.
        deskew: Apply deskew.
        normalise_dpi_value: Scale to target_dpi.
        morphology_cleanup_enabled: Apply closing/opening to connect broken strokes and remove noise (receipts/invoices).
        target_dpi: Target DPI (default 300).
        assumed_dpi: Assumed current DPI when unknown.

    Returns:
        Preprocessed image (binary if binarise=True, else grayscale).
    """
    out = image.copy()
    if len(out.shape) == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if normalise_dpi_value:
        out = normalise_dpi(out, target_dpi=target_dpi, assumed_dpi=assumed_dpi)
    if deskew:
        out = deskew_image(out)
    if binarise:
        out = binarise_otsu(out)
    if morphology_cleanup_enabled and binarise:
        out = morphology_cleanup(out, apply_close=True, apply_open=True)
    return out
