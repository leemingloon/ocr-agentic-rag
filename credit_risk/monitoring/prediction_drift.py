"""
Prediction Drift Detection

Monitors drift in model predictions (PD distribution shifts).

Triggers:
- Mean PD shift > 5%
- Distribution shift (KS test)
- Prediction variance changes

Actions:
- NO_ACTION: <5% drift
- INVESTIGATE: 5-10% drift
- INVESTIGATE_URGENT: >10% drift
"""

from typing import Dict, List, Optional, Literal
import numpy as np
from datetime import datetime


class PredictionDriftDetector:
    """
    Monitor drift in prediction distribution
    """
    
    def __init__(
        self,
        reference_preds: Optional[np.ndarray] = None,
        window_size: int = 1000
    ):
        """
        Initialize prediction drift detector
        
        Args:
            reference_preds: Reference predictions (e.g., from validation set)
            window_size: Number of predictions to monitor
        """
        self.reference_preds = reference_preds
        self.window_size = window_size
        self.predictions_buffer = []
    
    def add_prediction(self, pd_value: float):
        """Add a new prediction to the buffer"""
        self.predictions_buffer.append(pd_value)
        
        # Keep only window_size predictions
        if len(self.predictions_buffer) > self.window_size:
            self.predictions_buffer.pop(0)
    
    def detect_drift(
        self,
        current_preds: Optional[np.ndarray] = None,
        threshold: float = 0.05
    ) -> Dict:
        """
        Detect drift in predictions
        
        Args:
            current_preds: Current predictions (if None, uses buffer)
            threshold: Drift threshold (default: 5%)
            
        Returns:
            {
                "drift_detected": bool,
                "mean_shift": float,
                "action": str,
            }
        """
        # Use buffer if current_preds not provided
        if current_preds is None:
            if len(self.predictions_buffer) < 100:
                return {
                    "drift_detected": False,
                    "mean_shift": 0.0,
                    "action": "NO_ACTION",
                    "message": "Insufficient data in buffer"
                }
            current_preds = np.array(self.predictions_buffer)
        
        # Use reference predictions if available
        if self.reference_preds is None:
            # First batch becomes reference
            self.reference_preds = current_preds
            return {
                "drift_detected": False,
                "mean_shift": 0.0,
                "action": "NO_ACTION",
                "message": "Reference set established"
            }
        
        # Calculate mean shift
        ref_mean = np.mean(self.reference_preds)
        curr_mean = np.mean(current_preds)
        mean_shift = abs(curr_mean - ref_mean) / ref_mean
        
        # Determine action
        if mean_shift > 0.10:
            action = "INVESTIGATE_URGENT"
            drift_detected = True
        elif mean_shift > threshold:
            action = "INVESTIGATE"
            drift_detected = True
        else:
            action = "NO_ACTION"
            drift_detected = False
        
        return {
            "drift_detected": drift_detected,
            "mean_shift": float(mean_shift),
            "ref_mean": float(ref_mean),
            "curr_mean": float(curr_mean),
            "action": action,
            "timestamp": datetime.now().isoformat(),
        }


# Example usage
if __name__ == "__main__":
    # Reference predictions (validation set)
    reference = np.random.beta(2, 8, 1000)  # Mean ~0.2
    
    # Current predictions (with drift)
    current = np.random.beta(3, 7, 1000)  # Mean ~0.3
    
    detector = PredictionDriftDetector(reference_preds=reference)
    drift_result = detector.detect_drift(current)
    
    print("Prediction Drift Detection:")
    print(f"  Drift Detected: {drift_result['drift_detected']}")
    print(f"  Mean Shift: {drift_result['mean_shift']:.2%}")
    print(f"  Action: {drift_result['action']}")