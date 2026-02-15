"""
Counterfactual Analysis Engine

Answers "what-if" questions for credit risk scenarios.

Example Scenarios:
1. "What if Debt/EBITDA increases to 4.0x?"
2. "What if Interest Coverage drops to 2.0x?"
3. "What if news sentiment deteriorates to -0.5?"

Evaluation: Synthetic perturbations (1,000 scenarios)

Usage:
    analyzer = CounterfactualAnalyzer(pd_model)
    
    baseline = {
        "debt_to_ebitda": 3.0,
        "interest_coverage": 4.5,
    }
    
    result = analyzer.what_if(baseline, {"debt_to_ebitda": 4.0})
    
    print(f"Baseline PD: {result['baseline_pd']:.2%}")
    print(f"New PD: {result['new_pd']:.2%}")
    print(f"Delta: {result['delta_pd']:.2%}")
    print(f"Sensitivity: {result['sensitivity']:.3f}")
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .pd_model import PDModel


class CounterfactualAnalyzer:
    """
    Counterfactual "what-if" analysis for credit risk
    
    Supports:
    - Single variable perturbations
    - Multi-variable perturbations
    - Sensitivity analysis
    - Covenant stress testing
    """
    
    def __init__(self, pd_model: PDModel):
        """
        Initialize counterfactual analyzer
        
        Args:
            pd_model: Trained PD model
        """
        self.pd_model = pd_model
    
    def what_if(
        self,
        baseline_features: Dict[str, float],
        perturbations: Dict[str, float]
    ) -> Dict:
        """
        Run counterfactual analysis
        
        Args:
            baseline_features: Current feature values
            perturbations: Features to change (e.g., {"debt_to_ebitda": 4.0})
            
        Returns:
            {
                "baseline_pd": float,
                "new_pd": float,
                "delta_pd": float,
                "sensitivity": float,
                "perturbations": Dict,
            }
        """
        # Baseline PD
        baseline_pd = self.pd_model.predict_pd(baseline_features)
        
        # Create perturbed features
        perturbed_features = baseline_features.copy()
        perturbed_features.update(perturbations)
        
        # New PD
        new_pd = self.pd_model.predict_pd(perturbed_features)
        
        # Delta
        delta_pd = new_pd - baseline_pd
        
        # Sensitivity (PD change per unit change in perturbed variable)
        # For simplicity, use first perturbation
        first_var = list(perturbations.keys())[0]
        var_change = perturbations[first_var] - baseline_features.get(first_var, 0)
        
        sensitivity = delta_pd / var_change if var_change != 0 else 0
        
        return {
            "baseline_pd": baseline_pd,
            "new_pd": new_pd,
            "delta_pd": delta_pd,
            "sensitivity": sensitivity,
            "perturbations": perturbations,
        }
    
    def covenant_stress_test(
        self,
        baseline_features: Dict[str, float],
        covenant: str,
        threshold: float
    ) -> Dict:
        """
        Test impact of covenant breach
        
        Args:
            baseline_features: Current features
            covenant: Covenant metric (e.g., "debt_to_ebitda")
            threshold: Covenant threshold (e.g., 3.5)
            
        Returns:
            Analysis of covenant breach scenario
        """
        # Current value
        current_value = baseline_features.get(covenant, 0)
        
        # Scenario: breach threshold
        breach_scenario = self.what_if(
            baseline_features,
            {covenant: threshold}
        )
        
        # Scenario: exceed threshold by 20%
        severe_breach = self.what_if(
            baseline_features,
            {covenant: threshold * 1.2}
        )
        
        return {
            "covenant": covenant,
            "current_value": current_value,
            "threshold": threshold,
            "headroom": threshold - current_value,
            "at_threshold": breach_scenario,
            "severe_breach": severe_breach,
        }
    
    def sensitivity_analysis(
        self,
        baseline_features: Dict[str, float],
        variable: str,
        min_value: float,
        max_value: float,
        steps: int = 10
    ) -> pd.DataFrame:
        """
        Sensitivity analysis across a range
        
        Args:
            baseline_features: Baseline features
            variable: Variable to vary
            min_value: Minimum value
            max_value: Maximum value
            steps: Number of steps
            
        Returns:
            DataFrame with variable values and corresponding PDs
        """
        values = np.linspace(min_value, max_value, steps)
        pds = []
        
        for value in values:
            result = self.what_if(baseline_features, {variable: value})
            pds.append(result["new_pd"])
        
        return pd.DataFrame({
            variable: values,
            "pd": pds,
        })
    
    def multi_variable_scenarios(
        self,
        baseline_features: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict:
        """
        Analyze multiple scenarios
        
        Args:
            baseline_features: Baseline features
            scenarios: Dict of scenario name -> perturbations
                Example: {
                    "mild_stress": {"debt_to_ebitda": 3.5},
                    "severe_stress": {"debt_to_ebitda": 4.5, "interest_coverage": 2.0}
                }
                
        Returns:
            Dict of scenario results
        """
        results = {}
        
        for scenario_name, perturbations in scenarios.items():
            results[scenario_name] = self.what_if(baseline_features, perturbations)
        
        return results
    
    def find_threshold(
        self,
        baseline_features: Dict[str, float],
        variable: str,
        target_pd: float,
        initial_guess: float,
        tolerance: float = 0.01,
        max_iterations: int = 20
    ) -> Optional[float]:
        """
        Find variable value that produces target PD
        
        Example: "What Debt/EBITDA leads to 10% PD?"
        
        Args:
            baseline_features: Baseline features
            variable: Variable to adjust
            target_pd: Target PD
            initial_guess: Starting value
            tolerance: Convergence tolerance
            max_iterations: Max iterations
            
        Returns:
            Variable value that achieves target PD, or None if not found
        """
        current_value = initial_guess
        
        for i in range(max_iterations):
            # Test current value
            result = self.what_if(baseline_features, {variable: current_value})
            current_pd = result["new_pd"]
            
            # Check convergence
            if abs(current_pd - target_pd) < tolerance:
                return current_value
            
            # Adjust (simple binary search approach)
            if current_pd < target_pd:
                # Increase variable (assuming positive relationship)
                current_value *= 1.1
            else:
                # Decrease variable
                current_value *= 0.9
        
        return None  # Did not converge


# Example usage
if __name__ == "__main__":
    # Create dummy PD model
    pd_model = PDModel()
    
    analyzer = CounterfactualAnalyzer(pd_model)
    
    # Baseline features
    baseline = {
        "debt_to_ebitda": 3.0,
        "interest_coverage": 4.5,
        "current_ratio": 1.8,
        "news_sentiment": 0.1,
    }
    
    print("Counterfactual Analysis Examples")
    print("=" * 60)
    
    # Example 1: Single variable perturbation
    print("\n1. What if Debt/EBITDA increases to 4.0x?")
    result = analyzer.what_if(baseline, {"debt_to_ebitda": 4.0})
    print(f"   Baseline PD: {result['baseline_pd']:.2%}")
    print(f"   New PD: {result['new_pd']:.2%}")
    print(f"   Delta: {result['delta_pd']:+.2%}")
    print(f"   Sensitivity: {result['sensitivity']:.3f} (PD per 1.0x leverage)")
    
    # Example 2: Covenant stress test
    print("\n2. Covenant Stress Test (Debt/EBITDA < 3.5x)")
    covenant_result = analyzer.covenant_stress_test(
        baseline,
        covenant="debt_to_ebitda",
        threshold=3.5
    )
    print(f"   Current: {covenant_result['current_value']:.2f}x")
    print(f"   Threshold: {covenant_result['threshold']:.2f}x")
    print(f"   Headroom: {covenant_result['headroom']:.2f}x")
    print(f"   PD at threshold: {covenant_result['at_threshold']['new_pd']:.2%}")
    print(f"   PD at severe breach: {covenant_result['severe_breach']['new_pd']:.2%}")
    
    # Example 3: Multi-variable scenarios
    print("\n3. Multiple Scenarios")
    scenarios = {
        "base_case": {},
        "mild_stress": {"debt_to_ebitda": 3.5},
        "severe_stress": {"debt_to_ebitda": 4.0, "interest_coverage": 3.0},
        "crisis": {"debt_to_ebitda": 5.0, "interest_coverage": 2.0, "news_sentiment": -0.5},
    }
    
    scenario_results = analyzer.multi_variable_scenarios(baseline, scenarios)
    
    for scenario_name, result in scenario_results.items():
        if result:
            print(f"   {scenario_name}: PD = {result['new_pd']:.2%}")
        else:
            print(f"   {scenario_name}: PD = {pd_model.predict_pd(baseline):.2%}")