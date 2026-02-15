"""
Trend Engine - Detect Deteriorating Financial Trends

Analyzes time series of financial ratios to detect deterioration signals.

Signals:
- Increasing Debt/EBITDA (QoQ)
- Decreasing Interest Coverage
- Declining Revenue (YoY)
- Shrinking EBITDA Margin
- Deteriorating Working Capital
"""

import pandas as pd
from typing import Dict, List, Optional


class TrendEngine:
    """
    Detect deteriorating trends in financial metrics
    """
    
    def __init__(self):
        # Thresholds for trend detection
        self.thresholds = {
            "debt_ebitda_increase": 0.2,  # QoQ increase threshold
            "interest_coverage_decrease": 0.5,  # QoQ decrease threshold
            "revenue_decline": 0.05,  # YoY decline (5%)
            "ebitda_margin_decline": 0.02,  # 2pp decline
            "working_capital_decline": 0.10,  # 10% decline
        }
    
    def detect_deterioration(
        self,
        ratios_ts: pd.DataFrame,
        lookback_periods: int = 4
    ) -> Dict:
        """
        Detect deteriorating trends
        
        Args:
            ratios_ts: Time series DataFrame with columns like 'debt_to_ebitda', 'revenue', etc.
                      Indexed by time period (e.g., quarters)
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dict of detected signals with severity levels
        """
        if len(ratios_ts) < 2:
            return {}
        
        signals = {}
        
        # Debt/EBITDA trend
        if "debt_to_ebitda" in ratios_ts.columns:
            latest = ratios_ts["debt_to_ebitda"].iloc[-1]
            previous = ratios_ts["debt_to_ebitda"].iloc[-2]
            delta = latest - previous
            
            if delta > self.thresholds["debt_ebitda_increase"]:
                signals["debt_deterioration"] = {
                    "severity": "HIGH" if delta > 0.5 else "MEDIUM",
                    "delta": float(delta),
                    "latest": float(latest),
                    "previous": float(previous),
                }
        
        # Interest Coverage trend
        if "interest_coverage" in ratios_ts.columns:
            latest = ratios_ts["interest_coverage"].iloc[-1]
            previous = ratios_ts["interest_coverage"].iloc[-2]
            delta = latest - previous
            
            if delta < -self.thresholds["interest_coverage_decrease"]:
                signals["interest_coverage_decline"] = {
                    "severity": "HIGH" if delta < -1.0 else "MEDIUM",
                    "delta": float(delta),
                    "latest": float(latest),
                    "previous": float(previous),
                }
        
        # Revenue trend (if available)
        if "revenue" in ratios_ts.columns and len(ratios_ts) >= 4:
            latest_yoy = ratios_ts["revenue"].iloc[-1] / ratios_ts["revenue"].iloc[-4] - 1
            
            if latest_yoy < -self.thresholds["revenue_decline"]:
                signals["revenue_decline"] = {
                    "severity": "HIGH" if latest_yoy < -0.10 else "MEDIUM",
                    "yoy_change": float(latest_yoy),
                }
        
        return signals
    
    def calculate_trend_score(self, signals: Dict) -> float:
        """
        Calculate overall trend score (0=severe deterioration, 1=no issues)
        
        Args:
            signals: Dict from detect_deterioration()
            
        Returns:
            Trend score (0-1)
        """
        if not signals:
            return 1.0  # No deterioration
        
        # Count severity levels
        high_severity = sum(1 for s in signals.values() if s.get("severity") == "HIGH")
        medium_severity = sum(1 for s in signals.values() if s.get("severity") == "MEDIUM")
        
        # Score: penalize HIGH more than MEDIUM
        penalty = (high_severity * 0.3) + (medium_severity * 0.15)
        
        return max(0.0, 1.0 - penalty)
    
    def format_signals(self, signals: Dict) -> List[str]:
        """
        Format signals for display
        
        Returns:
            List of human-readable signal descriptions
        """
        formatted = []
        
        for signal_name, signal_data in signals.items():
            severity = signal_data.get("severity", "UNKNOWN")
            
            if signal_name == "debt_deterioration":
                formatted.append(
                    f"[{severity}] Debt/EBITDA increased by {signal_data['delta']:.2f}x QoQ"
                )
            elif signal_name == "interest_coverage_decline":
                formatted.append(
                    f"[{severity}] Interest Coverage decreased by {abs(signal_data['delta']):.2f}x QoQ"
                )
            elif signal_name == "revenue_decline":
                formatted.append(
                    f"[{severity}] Revenue declined by {abs(signal_data['yoy_change'])*100:.1f}% YoY"
                )
        
        return formatted


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create sample time series
    dates = pd.date_range("2024-01-01", periods=4, freq="Q")
    
    ratios_ts = pd.DataFrame({
        "debt_to_ebitda": [2.5, 2.8, 3.2, 3.6],  # Increasing trend
        "interest_coverage": [5.0, 4.5, 4.0, 3.2],  # Decreasing trend
        "revenue": [10.0, 10.5, 10.3, 9.8],  # Slight decline
    }, index=dates)
    
    engine = TrendEngine()
    signals = engine.detect_deterioration(ratios_ts)
    
    print("Detected Signals:")
    for signal in engine.format_signals(signals):
        print(f"  {signal}")
    
    trend_score = engine.calculate_trend_score(signals)
    print(f"\nTrend Score: {trend_score:.2f} (0=severe, 1=healthy)")