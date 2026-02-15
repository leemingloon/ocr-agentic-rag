"""
Financial Ratio Builder

Extracts key financial ratios from OCR-parsed financial statements.

Ratios Computed:
1. Debt/EBITDA - Leverage indicator
2. Current Ratio - Liquidity
3. Quick Ratio - Liquidity (excludes inventory)
4. Interest Coverage - Ability to service debt
5. Debt/Equity - Capital structure

Industry Thresholds (Investment Grade):
- Debt/EBITDA < 3.0x
- Current Ratio > 1.5
- Quick Ratio > 1.0
- Interest Coverage > 3.0x
- Debt/Equity < 1.0

Usage:
    builder = RatioBuilder()
    
    financials = {
        "revenue": 10000000,
        "ebitda": 2000000,
        "total_debt": 5000000,
        "current_assets": 3000000,
        "current_liabilities": 2000000,
        "inventory": 500000,
        "interest_expense": 400000,
        "equity": 6000000,
    }
    
    ratios = builder.extract_ratios(financials)
    
    print(ratios)
    # {
    #     "debt_to_ebitda": 2.5,
    #     "current_ratio": 1.5,
    #     "quick_ratio": 1.25,
    #     "interest_coverage": 5.0,
    #     "debt_to_equity": 0.83,
    # }
"""

from typing import Dict, Optional
import numpy as np


class RatioBuilder:
    """
    Financial ratio calculator
    
    Converts raw financial statement line items into credit-relevant ratios.
    """
    
    def __init__(self):
        """Initialize ratio builder"""
        # Industry benchmark thresholds (investment grade)
        self.thresholds = {
            "debt_to_ebitda": {"max": 3.0, "direction": "lower_better"},
            "current_ratio": {"min": 1.5, "direction": "higher_better"},
            "quick_ratio": {"min": 1.0, "direction": "higher_better"},
            "interest_coverage": {"min": 3.0, "direction": "higher_better"},
            "debt_to_equity": {"max": 1.0, "direction": "lower_better"},
        }
    
    def extract_ratios(self, financials: Dict[str, float]) -> Dict[str, float]:
        """
        Extract all key financial ratios
        
        Args:
            financials: Dictionary with financial statement items
                Required keys: revenue, ebitda, total_debt, current_assets,
                              current_liabilities, interest_expense, equity
                Optional keys: inventory, depreciation, ebit
                
        Returns:
            Dictionary of financial ratios
        """
        ratios = {}
        
        # 1. Debt/EBITDA (Leverage)
        if "total_debt" in financials and "ebitda" in financials:
            ratios["debt_to_ebitda"] = self._safe_divide(
                financials["total_debt"],
                financials["ebitda"]
            )
        
        # 2. Current Ratio (Liquidity)
        if "current_assets" in financials and "current_liabilities" in financials:
            ratios["current_ratio"] = self._safe_divide(
                financials["current_assets"],
                financials["current_liabilities"]
            )
        
        # 3. Quick Ratio (Liquidity, excluding inventory)
        if all(k in financials for k in ["current_assets", "current_liabilities", "inventory"]):
            quick_assets = financials["current_assets"] - financials["inventory"]
            ratios["quick_ratio"] = self._safe_divide(
                quick_assets,
                financials["current_liabilities"]
            )
        elif all(k in financials for k in ["current_assets", "current_liabilities"]):
            # If inventory not available, assume 30% of current assets
            quick_assets = financials["current_assets"] * 0.7
            ratios["quick_ratio"] = self._safe_divide(
                quick_assets,
                financials["current_liabilities"]
            )
        
        # 4. Interest Coverage (Debt serviceability)
        if "ebitda" in financials and "interest_expense" in financials:
            ratios["interest_coverage"] = self._safe_divide(
                financials["ebitda"],
                financials["interest_expense"]
            )
        elif "ebit" in financials and "interest_expense" in financials:
            # Use EBIT if EBITDA not available
            ratios["interest_coverage"] = self._safe_divide(
                financials["ebit"],
                financials["interest_expense"]
            )
        
        # 5. Debt/Equity (Capital structure)
        if "total_debt" in financials and "equity" in financials:
            ratios["debt_to_equity"] = self._safe_divide(
                financials["total_debt"],
                financials["equity"]
            )
        
        # 6. Additional ratios (if data available)
        
        # Revenue growth (if historical revenue available)
        if "revenue" in financials and "revenue_prev_year" in financials:
            ratios["revenue_growth"] = self._safe_divide(
                financials["revenue"] - financials["revenue_prev_year"],
                financials["revenue_prev_year"]
            )
        
        # EBITDA margin
        if "ebitda" in financials and "revenue" in financials:
            ratios["ebitda_margin"] = self._safe_divide(
                financials["ebitda"],
                financials["revenue"]
            )
        
        # Total Asset Turnover
        if "revenue" in financials and "total_assets" in financials:
            ratios["asset_turnover"] = self._safe_divide(
                financials["revenue"],
                financials["total_assets"]
            )
        
        return ratios
    
    def assess_credit_quality(self, ratios: Dict[str, float]) -> Dict:
        """
        Assess credit quality based on ratios vs thresholds
        
        Args:
            ratios: Dictionary of financial ratios
            
        Returns:
            {
                "overall_score": float (0-1),
                "flags": List[str],
                "assessment": str ("strong", "adequate", "weak")
            }
        """
        flags = []
        scores = []
        
        for ratio_name, ratio_value in ratios.items():
            if ratio_name not in self.thresholds:
                continue
            
            threshold = self.thresholds[ratio_name]
            
            # Check if ratio breaches threshold
            if "max" in threshold and ratio_value > threshold["max"]:
                flags.append(f"{ratio_name} exceeds threshold: {ratio_value:.2f} > {threshold['max']}")
                scores.append(0.0)
            elif "min" in threshold and ratio_value < threshold["min"]:
                flags.append(f"{ratio_name} below threshold: {ratio_value:.2f} < {threshold['min']}")
                scores.append(0.0)
            else:
                scores.append(1.0)
        
        # Overall score (% of ratios meeting thresholds)
        overall_score = np.mean(scores) if scores else 0.5
        
        # Assessment
        if overall_score >= 0.8:
            assessment = "strong"
        elif overall_score >= 0.6:
            assessment = "adequate"
        else:
            assessment = "weak"
        
        return {
            "overall_score": overall_score,
            "flags": flags,
            "assessment": assessment,
            "ratios_checked": len(scores),
        }
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safe division with handling for zero/negative denominators
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            
        Returns:
            Division result, or np.nan if invalid
        """
        if denominator == 0:
            return np.nan
        
        # For ratios like debt/EBITDA, negative EBITDA is concerning
        if denominator < 0:
            return np.inf  # Flag as problematic
        
        return numerator / denominator
    
    def format_ratios_for_display(self, ratios: Dict[str, float]) -> str:
        """
        Format ratios for human-readable display
        
        Args:
            ratios: Dictionary of financial ratios
            
        Returns:
            Formatted string
        """
        lines = ["Financial Ratios:"]
        lines.append("-" * 50)
        
        ratio_display_names = {
            "debt_to_ebitda": "Debt/EBITDA",
            "current_ratio": "Current Ratio",
            "quick_ratio": "Quick Ratio",
            "interest_coverage": "Interest Coverage",
            "debt_to_equity": "Debt/Equity",
            "revenue_growth": "Revenue Growth",
            "ebitda_margin": "EBITDA Margin",
            "asset_turnover": "Asset Turnover",
        }
        
        for ratio_name, ratio_value in ratios.items():
            display_name = ratio_display_names.get(ratio_name, ratio_name)
            
            # Format based on type
            if ratio_name in ["revenue_growth", "ebitda_margin"]:
                # Show as percentage
                formatted = f"{ratio_value:.1%}"
            else:
                # Show as decimal
                formatted = f"{ratio_value:.2f}x" if not np.isnan(ratio_value) else "N/A"
            
            # Add threshold indicator
            if ratio_name in self.thresholds:
                threshold = self.thresholds[ratio_name]
                if "max" in threshold:
                    indicator = "✓" if ratio_value <= threshold["max"] else "✗"
                    lines.append(f"  {display_name}: {formatted} {indicator} (max: {threshold['max']:.1f}x)")
                elif "min" in threshold:
                    indicator = "✓" if ratio_value >= threshold["min"] else "✗"
                    lines.append(f"  {display_name}: {formatted} {indicator} (min: {threshold['min']:.1f}x)")
            else:
                lines.append(f"  {display_name}: {formatted}")
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Sample financial data (ABC Corp)
    financials = {
        "revenue": 10000000,       # $10M
        "ebitda": 2000000,         # $2M
        "total_debt": 5000000,     # $5M
        "current_assets": 3000000, # $3M
        "current_liabilities": 2000000, # $2M
        "inventory": 500000,       # $500K
        "interest_expense": 400000, # $400K
        "equity": 6000000,         # $6M
    }
    
    # Build ratios
    builder = RatioBuilder()
    ratios = builder.extract_ratios(financials)
    
    print("Extracted Ratios:")
    print(ratios)
    
    print("\n" + builder.format_ratios_for_display(ratios))
    
    # Assess credit quality
    assessment = builder.assess_credit_quality(ratios)
    print("\n\nCredit Quality Assessment:")
    print(f"  Overall Score: {assessment['overall_score']:.1%}")
    print(f"  Assessment: {assessment['assessment'].upper()}")
    print(f"  Ratios Checked: {assessment['ratios_checked']}")
    
    if assessment['flags']:
        print("\n  Flags:")
        for flag in assessment['flags']:
            print(f"    • {flag}")
    else:
        print("\n  ✓ All ratios within acceptable thresholds")