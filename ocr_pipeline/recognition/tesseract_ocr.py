"""
Tesseract OCR Wrapper

Wrapper around Tesseract with confidence scoring
and financial document optimization
"""
from email.mime import text
import os
import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# On local, set tesseract_cmd if environment variable is provided
tesseract_path = os.environ.get("TESSERACT_CMD")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

@dataclass
class OCRResult:
    """OCR recognition result"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    word_confidences: List[float]


class TesseractOCR:
    """
    Tesseract OCR with financial document optimization
    
    Performance:
    - Speed: ~200ms per document
    - Cost: $0 (free)
    - Accuracy: 76-89% (depends on quality)
    """
    
    def __init__(
        self,
        lang: str = "eng",
        psm: int = 3,  # Fully automatic page segmentation
        oem: int = 3,  # Default OCR Engine Mode (LSTM)
        tesseract_cmd: Optional[str] = None,
    ):
        """
        Initialize Tesseract OCR
        
        Args:
            lang: Language (eng, chi_sim, etc.)
            psm: Page Segmentation Mode
            oem: OCR Engine Mode
            tesseract_cmd: Path to tesseract executable
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Financial document configuration
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
        
        # Preprocess ROI
        preprocessed = self._preprocess_roi(roi)
        
        # Run Tesseract with detailed output
        data = pytesseract.image_to_data(
            preprocessed,
            lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT
        )
        if not data['text'][0].strip():
            print(f"Warning: OCR returned empty text for {preprocessed}")

        # Extract text and confidences
        text_parts = []
        word_confidences = []
        
        for i, conf in enumerate(data['conf']):
            if conf > 0:  # Valid detection
                word = data['text'][i]
                if word.strip():  # Non-empty
                    text_parts.append(word)
                    word_confidences.append(float(conf))
        
        # Combine text
        full_text = " ".join(text_parts)
        
        # Calculate weighted average confidence
        if word_confidences:
            # Weight numbers higher for financial docs
            weighted_confs = []
            for word, conf in zip(text_parts, word_confidences):
                weight = 2.0 if any(c.isdigit() for c in word) else 1.0
                weighted_confs.append(conf * weight)
            
            avg_confidence = sum(weighted_confs) / sum(
                2.0 if any(c.isdigit() for c in word) else 1.0 
                for word in text_parts
            )
        else:
            avg_confidence = 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bbox=bbox,
            word_confidences=word_confidences,
        )
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI for better OCR accuracy
        
        Args:
            roi: Region of interest
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Adaptive thresholding (better for varying lighting)
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def recognize_multiple(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[OCRResult]:
        """
        Recognize text in multiple bounding boxes
        
        Args:
            image: Full image
            boxes: List of (x, y, w, h) boxes
            
        Returns:
            List of OCRResult
        """
        results = []
        
        for box in boxes:
            result = self.recognize(image, box)
            results.append(result)
        
        return results


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