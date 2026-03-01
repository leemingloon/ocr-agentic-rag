"""
Tesseract OCR Wrapper

Wrapper around Tesseract with confidence scoring, preprocessing (Otsu, deskew, 300 DPI),
and optional table reconstruction from bounding boxes. Confidence scores are used
selectively: words below threshold are flagged for human review, not discarded.
"""
import os
import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# On local, set tesseract_cmd if environment variable is provided
tesseract_path = os.environ.get("TESSERACT_CMD")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Confidence below this is flagged for human review (not discarded)
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 60


@dataclass
class OCRResult:
    """OCR recognition result."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    word_confidences: List[float]
    low_confidence_words: List[Tuple[str, float]] = field(default_factory=list)  # (word, conf) for conf < threshold
    words_with_bboxes: List[Tuple[str, Tuple[int, int, int, int], float]] = field(default_factory=list)  # (text, bbox, conf)


class TesseractOCR:
    """
    Tesseract OCR with financial document optimization.

    Preprocessing (when use_strong_preprocessing=True): Otsu binarisation,
    deskew, 300 DPI normalisation before passing to Tesseract.
    Table mode: psm 6 (uniform block) or 11 (sparse) for table-friendly layout.
    Confidence: words below low_confidence_threshold are flagged for human review, not discarded.
    """
    
    def __init__(
        self,
        lang: str = "eng",
        psm: int = 3,  # Fully automatic; use 6 for uniform block, 11 for sparse text (tables)
        oem: int = 3,  # Default OCR Engine Mode (LSTM)
        tesseract_cmd: Optional[str] = None,
        use_strong_preprocessing: bool = True,  # Otsu + deskew + 300 DPI
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize Tesseract OCR.

        Args:
            lang: Language (eng, chi_sim, etc.)
            psm: Page Segmentation Mode (3=auto, 6=block, 11=sparse text)
            oem: OCR Engine Mode
            tesseract_cmd: Path to tesseract executable
            use_strong_preprocessing: Apply binarise + deskew + 300 DPI before OCR
            low_confidence_threshold: Flag words with conf < this for human review
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.use_strong_preprocessing = use_strong_preprocessing
        self.low_confidence_threshold = low_confidence_threshold
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.config = self._build_config()
        
    def _build_config(self) -> str:
        """Build Tesseract configuration string"""
        config_parts = [
            f"--psm {self.psm}",
            f"--oem {self.oem}",
            "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$%-:() ",
        ]
        return " ".join(config_parts)
    
    def recognize(
        self, 
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> OCRResult:
        """
        Recognize text in image or ROI
        
        Args:
            image: Grayscale or BGR image
            bbox: Optional bounding box (x, y, w, h)
            
        Returns:
            OCRResult with text and confidence
        """
        # Extract ROI if bbox provided
        if bbox:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
            bbox = (0, 0, image.shape[1], image.shape[0])
        
        preprocessed = self._preprocess_roi(roi)
        data = pytesseract.image_to_data(
            preprocessed,
            lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT,
        )
        if data["text"] and not (data["text"][0] or "").strip():
            print("Warning: OCR returned empty text for image")

        text_parts = []
        word_confidences = []
        words_with_bboxes: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        low_confidence_words: List[Tuple[str, float]] = []

        for i, conf in enumerate(data["conf"]):
            if conf > 0:
                word = (data["text"][i] or "").strip()
                if not word:
                    continue
                c = float(conf)
                text_parts.append(word)
                word_confidences.append(c)
                x = data["left"][i] + (bbox[0] if bbox != (0, 0, roi.shape[1], roi.shape[0]) else 0)
                y = data["top"][i] + (bbox[1] if bbox != (0, 0, roi.shape[1], roi.shape[0]) else 0)
                w, h = data["width"][i], data["height"][i]
                word_bbox = (x, y, w, h)
                words_with_bboxes.append((word, word_bbox, c))
                if c < self.low_confidence_threshold:
                    low_confidence_words.append((word, c))

        full_text = " ".join(text_parts)
        if word_confidences:
            weighted_confs = []
            for word, conf in zip(text_parts, word_confidences):
                weight = 2.0 if any(c.isdigit() for c in word) else 1.0
                weighted_confs.append(conf * weight)
            avg_confidence = sum(weighted_confs) / sum(
                2.0 if any(c.isdigit() for c in word) else 1.0 for word in text_parts
            )
        else:
            avg_confidence = 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bbox=bbox,
            word_confidences=word_confidences,
            low_confidence_words=low_confidence_words,
            words_with_bboxes=words_with_bboxes,
        )
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI: strong pipeline (Otsu + deskew + 300 DPI) or legacy adaptive threshold.
        """
        if self.use_strong_preprocessing:
            try:
                from ocr_pipeline.preprocessing.document_preprocessor import preprocess_for_ocr
                return preprocess_for_ocr(roi, binarise=True, deskew=True, normalise_dpi_value=True)
            except Exception:
                pass
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    def recognize_multiple(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[OCRResult]:
        """Recognize text in multiple bounding boxes. Returns list of OCRResult."""
        return [self.recognize(image, box) for box in boxes]

    def image_to_data_raw(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Return raw image_to_data dict (for table reconstruction or custom processing)."""
        roi = image
        if bbox:
            x, y, w, h = bbox
            roi = image[y : y + h, x : x + w]
        preprocessed = self._preprocess_roi(roi)
        return pytesseract.image_to_data(
            preprocessed, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
        )

    @staticmethod
    def reconstruct_table_from_data(
        data: Dict[str, Any],
        row_tolerance: int = 8,
    ) -> List[List[str]]:
        """
        Reconstruct table rows from image_to_data() output: group by approximate y, sort by x.

        Args:
            data: Dict from pytesseract.image_to_data(..., output_type=Output.DICT)
            row_tolerance: Max pixel difference in y to treat as same row.

        Returns:
            List of rows, each row is list of cell strings (left-to-right).
        """
        entries = []
        for i in range(len(data.get("text", []))):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            conf = data.get("conf", [0] * (i + 1))[i] if i < len(data.get("conf", [])) else 0
            if conf < 0:
                continue
            x = data["left"][i]
            y = data["top"][i]
            entries.append((x, y, word))
        if not entries:
            return []
        entries.sort(key=lambda e: (e[1], e[0]))
        rows: List[List[Tuple[int, int, str]]] = []
        for x, y, word in entries:
            if not rows:
                rows.append([(x, y, word)])
                continue
            last_y = rows[-1][0][1]
            if abs(y - last_y) <= row_tolerance:
                rows[-1].append((x, y, word))
            else:
                rows.append([(x, y, word)])
        for i in range(len(rows)):
            rows[i].sort(key=lambda e: e[0])
        return [[cell[2] for cell in row] for row in rows]

    def recognize_with_table(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        table_psm: int = 6,
        row_tolerance: int = 8,
    ) -> Tuple[OCRResult, List[List[str]]]:
        """
        Recognize and reconstruct table: use psm 6 (uniform block) or 11 (sparse), then group by y.

        Returns:
            (OCRResult, table_rows)
        """
        prev_psm = self.psm
        self.psm = table_psm
        self.config = self._build_config()
        try:
            data = self.image_to_data_raw(image, bbox=bbox)
            result = self.recognize(image, bbox)
            table_rows = self.reconstruct_table_from_data(data, row_tolerance=row_tolerance)
            return result, table_rows
        finally:
            self.psm = prev_psm
            self.config = self._build_config()

    def to_standard_output(self, result: OCRResult) -> "StandardOCROutput":
        """Convert OCRResult to model-agnostic StandardOCROutput (text, bbox, confidence per word)."""
        from .ocr_schema import StandardOCROutput, OCRWord
        words = [
            OCRWord(text=text, bbox=box, confidence=conf)
            for text, box, conf in result.words_with_bboxes
        ]
        low = [
            OCRWord(text=w, bbox=(0, 0, 0, 0), confidence=c)
            for w, c in result.low_confidence_words
        ]
        return StandardOCROutput(
            words=words,
            full_text=result.text,
            confidence=result.confidence,
            low_confidence_words=low,
        )


# Example usage
if __name__ == "__main__":
    # Initialize OCR
    ocr = TesseractOCR()
    
    # Test on sample image
    image = cv2.imread("data/images/invoice_001.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Recognize full page
    result = ocr.recognize(image)
    
    print(f"Recognized text ({result.confidence:.1f}% confidence):")
    print(result.text)
    print(f"\nWord-level confidences: {len(result.word_confidences)} words")