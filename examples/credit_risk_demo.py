"""
Credit Risk Pipeline Demo

Demonstrates standalone credit risk functionality.

Modes:
- local: 1 sample (fast demo)
- sagemaker: Batch processing on AWS (600 samples)
- production: Full evaluation (3.7M samples)

Usage:
    # Local demo (1 sample)
    python examples/05_credit_risk_demo.py
    
    # SageMaker batch
    python examples/05_credit_risk_demo.py --mode sagemaker --s3-bucket my-bucket
"""

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from credit_risk.pipeline import CreditRiskPipeline
from credit_risk.feature_engineering.ratio_builder import RatioBuilder
from credit_risk.models.counterfactual import CounterfactualAnalyzer


def run_demo(mode: str = "local", s3_bucket: str = None):
    """Run credit risk demo"""
    
    print("=" * 70)
    print(f"Credit Risk Pipeline Demo - {mode.upper()} Mode")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = CreditRiskPipeline(
        mode=mode,
        s3_bucket=s3_bucket
    )
    
    # Sample borrower data
    borrower = "ABC Corp"
    
    # Mock OCR output (from upstream OCR pipeline)
    ocr_output = {
        "text": """
        ABC Corp Financial Statement Q3 2024
        
        Revenue: $10,000,000
        EBITDA: $2,000,000
        Total Debt: $7,000,000
        Interest Expense: $500,000
        Current Assets: $3,000,000
        Current Liabilities: $2,000,000
        Inventory: $500,000
        Equity: $5,000,000
        """
    }
    
    # Mock news articles
    news_articles = [
        "ABC Corp announced disappointing Q3 results with revenue missing expectations.",
        "The CFO cited weak demand and pricing pressures in the core market.",
        "Analysts are concerned about rising leverage and liquidity position.",
    ]
    
    print("\n" + "-" * 70)
    print("Processing Borrower: ABC Corp")
    print("-" * 70)
    
    # Run pipeline
    result = pipeline.process(
        borrower=borrower,
        ocr_output=ocr_output,
        news_articles=news_articles,
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n1. Default Probability (PD):")
    print(f"   {result['pd']:.2%}")
    
    print(f"\n2. Top Risk Drivers:")
    for i, driver in enumerate(result['drivers'], 1):
        print(f"   {i}. {driver}")
    
    print(f"\n3. Key Financial Ratios:")
    for ratio, value in result['features'].items():
        if ratio in ['debt_to_ebitda', 'interest_coverage', 'current_ratio']:
            print(f"   {ratio}: {value:.2f}")
    
    print(f"\n4. Counterfactual Scenarios:")
    for scenario, cf_result in result['counterfactuals'].items():
        print(f"   {scenario}: PD {cf_result['baseline_pd']:.2%} → {cf_result['new_pd']:.2%}")
    
    print(f"\n5. Risk Memo:")
    print("-" * 70)
    print(result['risk_memo'][:500] + "...")
    
    print(f"\n6. Drift Detection:")
    print(f"   Drift Detected: {result['drift_detected']}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    
    # Save results if SageMaker mode
    if mode == "sagemaker" and s3_bucket:
        print(f"\n✓ Results would be saved to s3://{s3_bucket}/results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="local", choices=["local", "sagemaker", "production"])
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket for SageMaker mode")
    
    args = parser.parse_args()
    
    run_demo(mode=args.mode, s3_bucket=args.s3_bucket)