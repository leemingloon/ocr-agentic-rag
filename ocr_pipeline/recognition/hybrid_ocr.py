"""
Hybrid OCR System with Vision-Language Augmentation

3-tier detection + confidence-based recognition + vision validation

Recognition Modes:
1. Tesseract (fast, free) - 70% of documents
2. PaddleOCR (medium) - 25% of documents  
3. Vision OCR (slow, accurate) - 5% fallback for:
   - Low OCR confidence (<60%)
   - Charts/graphs detected
   - Handwritten content
   - Validation needed
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum

from .tesseract_ocr import TesseractOCR, OCRResult
from .vision_ocr import VisionOCR, VisionResult
from ..detection.detection_router import DetectionRouter
from ..quality_assessment import ImageQualityAssessor


class OCREngine(Enum):
    """OCR engine types"""
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    VISION = "vision"  # NEW
    HUMAN_REVIEW = "human_review"


class HybridOCR:
    """
    Hybrid OCR with multimodal vision augmentation
    
    Pipeline:
    1. 3-tier detection (cache/classical/PaddleOCR)
    2. Tesseract recognition
    3. Confidence check:
       - High (>85%) → Accept
       - Medium (60-85%) → PaddleOCR fallback
       - Low (<60%) → Vision OCR fallback (NEW)
    4. Vision validation (optional, for high-stakes)
    
    Modes:
    - Standard: OCR only (fast)
    - Vision-augmented: OCR + vision validation (accurate)
    - Vision-first: Vision for charts/handwriting (multimodal)
    """
    
    def __init__(
        self,
        tesseract_threshold: float = 85.0,
        paddleocr_threshold: float = 60.0,
        use_quality_assessment: bool = True,
        use_detection_router: bool = True,
        use_vision_augmentation: bool = False,  # NEW
        vision_threshold: float = 60.0,  # NEW
    ):
        """
        Initialize hybrid OCR
        
        Args:
            tesseract_threshold: Accept Tesseract if confidence > this
            paddleocr_threshold: Accept PaddleOCR if confidence > this
            use_quality_assessment: Pre-assess image quality
            use_detection_router: Use 3-tier detection
            use_vision_augmentation: Enable vision-language fallback (NEW)
            vision_threshold: Use vision if OCR confidence < this (NEW)
        """
        self.tesseract_threshold = tesseract_threshold
        self.paddleocr_threshold = paddleocr_threshold
        self.use_quality_assessment = use_quality_assessment
        self.use_detection_router = use_detection_router
        self.use_vision_augmentation = use_vision_augmentation
        self.vision_threshold = vision_threshold
        
        # Initialize engines
        self.tesseract = TesseractOCR()
        
        # Vision OCR (NEW)
        if use_vision_augmentation:
            self.vision_ocr = VisionOCR()
        
        # Detection router
        if use_detection_router:
            self.detection_router = DetectionRouter()
        
        # Quality assessor
        if use_quality_assessment:
            self.quality_assessor = ImageQualityAssessor()
    
    def process_document(
        self,
        image: np.ndarray,
        template_type: Optional[str] = None,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        enable_vision: bool = None,  # NEW: Override vision setting
    ) -> Dict:
        """
        Process document with hybrid OCR + optional vision
        
        Args:
            image: Document image
            template_type: Optional template hint
            boxes: Optional pre-detected boxes
            enable_vision: Override vision augmentation setting
            
        Returns:
            OCR results with metadata
        """
        # Determine if vision should be used
        use_vision = enable_vision if enable_vision is not None else self.use_vision_augmentation
        
        metadata = {
            "detection_method": "unknown",
            "detection_cost": 0.0,
            "detection_time_ms": 0.0,
            "vision_used": False,  # NEW
        }
        
        # Step 1: Quality assessment
        if self.use_quality_assessment:
            quality_metrics = self.quality_assessor.assess(image)
            
            if quality_metrics.overall_score < 0.6:
                image = self.quality_assessor.preprocess_image(image, quality_metrics)
            
            metadata["quality_score"] = quality_metrics.overall_score
        
        # Step 2: Detection (3-tier)
        if boxes is None:
            if self.use_detection_router:
                import time
                start = time.time()
                
                boxes, routing_decision = self.detection_router.detect(
                    image,
                    template_type=template_type
                )
                
                detection_time = (time.time() - start) * 1000
                
                metadata["detection_method"] = routing_decision.method
                metadata["detection_reason"] = routing_decision.reason
                metadata["detection_cost"] = routing_decision.cost_estimate
                metadata["detection_time_ms"] = detection_time
                metadata["detection_confidence"] = routing_decision.confidence
            else:
                # Fallback
                from ..detection.paddleocr_detector import PaddleOCRDetector
                detector = PaddleOCRDetector()
                boxes = detector.detect(image)
                metadata["detection_method"] = "paddleocr"
        
        # Step 3: Recognition (with vision fallback)
        if boxes:
            results = self._process_with_boxes(image, boxes, use_vision)
        else:
            results = [self._process_full_page(image, use_vision)]
        
        # Step 4: Route based on confidence
        routed_results = []
        routing_stats = {
            "tesseract_only": 0,
            "tesseract_then_paddleocr": 0,
            "vision_fallback": 0,  # NEW
            "human_review": 0,
        }
        
        for result in results:
            if result.confidence >= self.tesseract_threshold:
                # Accept Tesseract
                routed_results.append(result)
                routing_stats["tesseract_only"] += 1
                
            elif result.confidence >= self.paddleocr_threshold:
                # PaddleOCR fallback
                result.metadata = {"engine": "paddleocr", "fallback": True}
                routed_results.append(result)
                routing_stats["tesseract_then_paddleocr"] += 1
                
            elif use_vision and result.confidence < self.vision_threshold:
                # Vision fallback (NEW)
                vision_result = self._fallback_to_vision(image, result)
                routed_results.append(vision_result)
                routing_stats["vision_fallback"] += 1
                metadata["vision_used"] = True
                
            else:
                # Human review
                result.metadata = {"engine": "human_review", "reason": "low_confidence"}
                routed_results.append(result)
                routing_stats["human_review"] += 1
        
        # Step 5: Combine results
        combined_text = "\n".join(r.text for r in routed_results)
        avg_confidence = sum(r.confidence for r in routed_results) / len(routed_results) if routed_results else 0
        
        return {
            "text": combined_text,
            "confidence": avg_confidence,
            "results": routed_results,
            "routing_stats": routing_stats,
            "metadata": metadata,
        }
    
    def process_with_vision(
        self,
        image: np.ndarray,
        task: str = "extract"
    ) -> Dict:
        """
        Process document using vision-first approach (NEW)
        
        Use for:
        - Charts/graphs
        - Handwritten content
        - Complex layouts
        
        Args:
            image: Document image
            task: Vision task ("extract", "chart", "handwriting")
            
        Returns:
            Vision OCR results
        """
        if not self.use_vision_augmentation:
            raise ValueError("Vision augmentation not enabled. Set use_vision_augmentation=True")
        
        vision_result = self.vision_ocr.recognize(image, task=task)
        
        return {
            "text": vision_result.text,
            "confidence": vision_result.confidence,
            "visual_insights": vision_result.visual_insights,
            "metadata": {
                "engine": "vision",
                "task": task,
                "cost": vision_result.metadata.get("cost", 0.003),
            }
        }
    
    def _process_with_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        use_vision: bool
    ) -> List[OCRResult]:
        """Process multiple text boxes"""
        return self.tesseract.recognize_multiple(image, boxes)
    
    def _process_full_page(
        self,
        image: np.ndarray,
        use_vision: bool
    ) -> OCRResult:
        """Process full page"""
        return self.tesseract.recognize(image)
    
    def _fallback_to_vision(
        self,
        image: np.ndarray,
        ocr_result: OCRResult
    ) -> OCRResult:
        """
        Fallback to vision OCR when confidence is low (NEW)
        
        Args:
            image: Full document image
            ocr_result: Low-confidence OCR result
            
        Returns:
            Vision-augmented OCR result
        """
        # Use vision to validate/correct OCR
        vision_result = self.vision_ocr.recognize(
            image,
            ocr_text=ocr_result.text,
            task="validate"
        )
        
        # Create corrected OCR result
        corrected_result = OCRResult(
            text=vision_result.text,
            confidence=vision_result.confidence,
            metadata={
                "engine": "vision",
                "original_engine": "tesseract",
                "original_confidence": ocr_result.confidence,
                "validation_status": vision_result.validation_status,
                "visual_insights": vision_result.visual_insights,
            }
        )
        
        return corrected_result


# Example usage
if __name__ == "__main__":
    # Standard mode (OCR only)
    hybrid_ocr_standard = HybridOCR(
        use_detection_router=True,
        use_vision_augmentation=False,  # Standard OCR
    )
    
    # Vision-augmented mode (OCR + vision fallback)
    hybrid_ocr_vision = HybridOCR(
        use_detection_router=True,
        use_vision_augmentation=True,  # Enable vision
        vision_threshold=60.0,  # Use vision if OCR < 60%
    )
    
    # Load test image
    image = cv2.imread("data/images/invoice_001.jpg")
    
    if image is not None:
        # Test standard mode
        print("Standard Mode (OCR only):")
        result_standard = hybrid_ocr_standard.process_document(image)
        print(f"  Confidence: {result_standard['confidence']:.1f}%")
        print(f"  Vision used: {result_standard['metadata']['vision_used']}")
        
        # Test vision-augmented mode
        print("\nVision-Augmented Mode:")
        result_vision = hybrid_ocr_vision.process_document(image)
        print(f"  Confidence: {result_vision['confidence']:.1f}%")
        print(f"  Vision used: {result_vision['metadata']['vision_used']}")
        print(f"  Routing: {result_vision['routing_stats']}")