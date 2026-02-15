"""
Image Quality Assessment Module

Evaluates document image quality using multiple metrics:
- Brightness, Contrast, Sharpness, Resolution
- Routes to appropriate OCR engine based on quality
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Image quality assessment results"""
    brightness: float  # 0-1 (optimal: 0.4-0.7)
    contrast: float    # 0-1 (optimal: >0.3)
    sharpness: float   # 0-1 (optimal: >0.5)
    resolution_dpi: int
    overall_score: float  # 0-1 weighted average
    recommended_engine: str  # "tesseract" or "paddleocr"


class ImageQualityAssessor:
    """
    Assess image quality and recommend OCR engine
    
    Quality Thresholds:
    - High (>0.7): Use Tesseract (fast, cheap)
    - Medium (0.4-0.7): Use PaddleOCR (robust)
    - Low (<0.4): Reject or flag for preprocessing
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.3, 0.8),
        min_contrast: float = 0.25,
        min_sharpness: float = 0.4,
        min_dpi: int = 150,
    ):
        self.brightness_range = brightness_range
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness
        self.min_dpi = min_dpi
        
    def assess(self, image: np.ndarray) -> QualityMetrics:
        """
        Assess image quality across multiple dimensions
        
        Args:
            image: BGR image from cv2.imread()
            
        Returns:
            QualityMetrics with scores and recommendation
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate individual metrics
        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)
        sharpness = self._calculate_sharpness(gray)
        dpi = self._estimate_dpi(image)
        
        # Normalize scores to 0-1
        brightness_score = self._normalize_brightness(brightness)
        contrast_score = min(contrast / 100.0, 1.0)  # Normalize std dev
        sharpness_score = min(sharpness / 1000.0, 1.0)  # Normalize Laplacian variance
        dpi_score = min(dpi / 300.0, 1.0)  # 300 DPI is ideal
        
        # Weighted average (sharpness most important for OCR)
        overall_score = (
            0.2 * brightness_score +
            0.25 * contrast_score +
            0.35 * sharpness_score +
            0.2 * dpi_score
        )
        
        # Recommend engine based on quality
        if overall_score >= 0.7:
            recommended_engine = "tesseract"
        elif overall_score >= 0.4:
            recommended_engine = "paddleocr"
        else:
            recommended_engine = "reject"  # Too low quality
            
        return QualityMetrics(
            brightness=brightness_score,
            contrast=contrast_score,
            sharpness=sharpness_score,
            resolution_dpi=dpi,
            overall_score=overall_score,
            recommended_engine=recommended_engine,
        )
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate mean pixel intensity (0-255)"""
        return np.mean(gray)
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate contrast as standard deviation of pixel intensities"""
        return np.std(gray)
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """
        Calculate sharpness using Laplacian variance
        Higher variance = sharper image
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def _estimate_dpi(self, image: np.ndarray) -> int:
        """
        Estimate DPI based on image dimensions
        Assumes A4 document (8.27 x 11.69 inches)
        """
        height, width = image.shape[:2]
        
        # A4 in portrait: 8.27" wide
        estimated_dpi = width / 8.27
        
        return int(estimated_dpi)
    
    def _normalize_brightness(self, brightness: float) -> float:
        """
        Normalize brightness to 0-1 score
        Optimal range: 100-180 (out of 255)
        """
        optimal_min, optimal_max = 100, 180
        
        if brightness < optimal_min:
            # Too dark
            score = brightness / optimal_min
        elif brightness > optimal_max:
            # Too bright
            score = 1.0 - ((brightness - optimal_max) / (255 - optimal_max))
        else:
            # In optimal range
            score = 1.0
            
        return max(0.0, min(1.0, score))
    
    def should_preprocess(self, metrics: QualityMetrics) -> Dict[str, bool]:
        """
        Recommend preprocessing steps based on quality metrics
        
        Returns:
            Dict of preprocessing flags
        """
        return {
            "denoise": metrics.sharpness < 0.5,
            "enhance_contrast": metrics.contrast < 0.3,
            "adjust_brightness": metrics.brightness < 0.4 or metrics.brightness > 0.8,
            "upscale": metrics.resolution_dpi < self.min_dpi,
        }
    
    def preprocess_image(self, image: np.ndarray, metrics: QualityMetrics) -> np.ndarray:
        """
        Apply preprocessing based on quality assessment
        
        Args:
            image: Original image
            metrics: Quality assessment results
            
        Returns:
            Preprocessed image
        """
        preprocessed = image.copy()
        recommendations = self.should_preprocess(metrics)
        
        # Convert to grayscale if needed
        if len(preprocessed.shape) == 3:
            gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        else:
            gray = preprocessed
        
        # Denoise
        if recommendations["denoise"]:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast
        if recommendations["enhance_contrast"]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Adjust brightness
        if recommendations["adjust_brightness"]:
            # Simple gamma correction
            if metrics.brightness < 0.4:
                # Too dark - lighten
                gamma = 1.5
            else:
                # Too bright - darken
                gamma = 0.7
                
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in range(256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
        
        # Upscale if needed
        if recommendations["upscale"]:
            scale_factor = self.min_dpi / metrics.resolution_dpi
            new_width = int(gray.shape[1] * scale_factor)
            new_height = int(gray.shape[0] * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), 
                            interpolation=cv2.INTER_CUBIC)
        
        return gray


# Example usage
if __name__ == "__main__":
    # Test on sample image
    image = cv2.imread("data/images/sample_invoice.jpg")
    
    assessor = ImageQualityAssessor()
    metrics = assessor.assess(image)
    
    print(f"Quality Assessment:")
    print(f"  Brightness: {metrics.brightness:.2f}")
    print(f"  Contrast: {metrics.contrast:.2f}")
    print(f"  Sharpness: {metrics.sharpness:.2f}")
    print(f"  DPI: {metrics.resolution_dpi}")
    print(f"  Overall: {metrics.overall_score:.2f}")
    print(f"  Recommended: {metrics.recommended_engine}")
    
    if metrics.overall_score < 0.7:
        print("\nPreprocessing recommended:")
        recommendations = assessor.should_preprocess(metrics)
        for step, needed in recommendations.items():
            if needed:
                print(f"  - {step}")