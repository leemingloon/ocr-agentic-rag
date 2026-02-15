"""
End-to-End Full Test Suite

Orchestrates all E2E tests:
1. Functional (baseline)
2. Robustness
3. Bias & Fairness
4. Adversarial
5. Load

Generates comprehensive production readiness report.
"""

from typing import Dict
import json
from pathlib import Path

from .e2e_functional_eval import EndToEndFunctionalEvaluator
from .e2e_robustness_test import EndToEndRobustnessTest
# from .e2e_bias_test import EndToEndBiasTest  # Would implement
# from .e2e_adversarial_test import EndToEndAdversarialTest  # Would implement
# from .e2e_load_test import EndToEndLoadTest  # Would implement


class EndToEndFullSuite:
    """
    Complete E2E test suite orchestrator
    
    Runs all E2E tests and generates production readiness report:
    - Functional baseline
    - Robustness
    - Bias & fairness
    - Adversarial
    - Load
    """
    
    def __init__(
        self,
        functional_evaluator: EndToEndFunctionalEvaluator,
        output_dir: str = "data/evaluation/results"
    ):
        """
        Initialize full suite
        
        Args:
            functional_evaluator: Functional evaluator (baseline)
            output_dir: Output directory for results
        """
        self.functional_evaluator = functional_evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test modules
        self.robustness_test = EndToEndRobustnessTest(functional_evaluator)
        # self.bias_test = EndToEndBiasTest(functional_evaluator)
        # self.adversarial_test = EndToEndAdversarialTest(functional_evaluator)
        # self.load_test = EndToEndLoadTest(functional_evaluator)
    
    def run_full_suite(self, sample_size: int = 100) -> Dict:
        """
        Run complete E2E test suite
        
        Args:
            sample_size: Number of samples per test
            
        Returns:
            Complete test results
        """
        print("=" * 70)
        print("End-to-End Full Test Suite")
        print("=" * 70)
        
        results = {
            "suite": "E2E Full Suite",
            "sample_size": sample_size,
            "tests": {},
        }
        
        # Test 1: Functional (Baseline)
        print("\n" + "=" * 70)
        print("Test 1/5: Functional Evaluation (Baseline)")
        print("=" * 70)
        
        functional_results = self.functional_evaluator.evaluate(sample_size=sample_size)
        results["tests"]["functional"] = functional_results
        
        if "metrics" in functional_results:
            print(f"\n✓ Functional Baseline Established:")
            print(f"  Overall Fidelity: {functional_results['metrics']['overall_fidelity']:.2%}")
            print(f"  Answer Accuracy: {functional_results['metrics']['answer_accuracy']:.2%}")
            print(f"  Cost per Query: ${functional_results['metrics']['avg_cost_per_query']:.6f}")
        
        # Test 2: Robustness
        print("\n" + "=" * 70)
        print("Test 2/5: Robustness Test")
        print("=" * 70)
        
        robustness_results = self.robustness_test.test(
            sample_size=sample_size,
            establish_baseline=False  # Already established
        )
        results["tests"]["robustness"] = robustness_results
        
        if "summary" in robustness_results:
            print(f"\n✓ Robustness Test Complete:")
            print(f"  Avg Degradation: {robustness_results['summary']['avg_degradation_pct']:.1f}%")
            print(f"  Max Degradation: {robustness_results['summary']['max_degradation_pct']:.1f}%")
            print(f"  All Acceptable: {robustness_results['summary']['all_acceptable']}")
        
        # Test 3-5: (Would implement similarly)
        print("\n⚠ Tests 3-5 (Bias, Adversarial, Load) - Not yet implemented")
        
        # Generate production readiness report
        results["production_readiness"] = self._generate_readiness_report(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_readiness_report(self, results: Dict) -> Dict:
        """Generate production readiness assessment"""
        report = {
            "tests_run": len(results["tests"]),
            "tests_passed": 0,
            "overall_ready": False,
        }
        
        # Check functional
        if "functional" in results["tests"]:
            functional = results["tests"]["functional"].get("metrics", {})
            report["functional"] = {
                "fidelity": functional.get("overall_fidelity", 0),
                "accuracy": functional.get("answer_accuracy", 0),
                "passed": functional.get("answer_accuracy", 0) >= 0.85,
            }
            if report["functional"]["passed"]:
                report["tests_passed"] += 1
        
        # Check robustness
        if "robustness" in results["tests"]:
            robustness = results["tests"]["robustness"].get("summary", {})
            report["robustness"] = {
                "avg_degradation_pct": robustness.get("avg_degradation_pct", 0),
                "passed": robustness.get("all_acceptable", False),
            }
            if report["robustness"]["passed"]:
                report["tests_passed"] += 1
        
        # Overall
        report["overall_ready"] = report["tests_passed"] >= 4  # Need 4/5 tests passing
        
        return report
    
    def _save_results(self, results: Dict):
        """Save results to file"""
        output_file = self.output_dir / "e2e_full_suite_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    from ocr_pipeline.recognition.hybrid_ocr import HybridOCR
    from rag_system.retrieval import HybridRetriever
    from rag_system.reranking import BGEReranker
    from rag_system.agentic.orchestrator import AgenticRAG
    from rag_system.multimodal_rag import MultimodalRAG
    
    # Initialize components
    ocr_system = HybridOCR(use_detection_router=True, use_vision_augmentation=True)
    retriever = HybridRetriever()
    reranker = BGEReranker()
    agent = AgenticRAG(retriever=retriever, reranker=reranker)
    multimodal_rag = MultimodalRAG(retriever=retriever, reranker=reranker)
    
    # Initialize functional evaluator (baseline)
    functional_evaluator = EndToEndFunctionalEvaluator(
        ocr_system=ocr_system,
        retriever=retriever,
        agent=agent,
        multimodal_rag=multimodal_rag
    )
    
    # Initialize full suite
    full_suite = EndToEndFullSuite(functional_evaluator)
    
    # Run all tests
    results = full_suite.run_full_suite(sample_size=10)
    
    print("\n" + "=" * 70)
    print("Production Readiness Summary")
    print("=" * 70)
    print(f"  Tests Run: {results['production_readiness']['tests_run']}")
    print(f"  Tests Passed: {results['production_readiness']['tests_passed']}")
    print(f"  Production Ready: {results['production_readiness']['overall_ready']}")
