"""
Main E2E Runner Script

Single entry point for running the complete pipeline in any mode.

Modes:
- local: Fast demo (1-12 samples)
- sagemaker: AWS free tier batch (600 samples)
- production: Full evaluation (3.7M samples)

Usage:
    # Quick demo (1 sample)
    python run_e2e.py
    
    # Local evaluation (80 samples)
    python run_e2e.py --eval
    
    # SageMaker batch
    python run_e2e.py --mode sagemaker --s3-bucket my-bucket --eval
    
    # Production
    python run_e2e.py --mode production --eval
"""

import argparse
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Verify API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("⚠️  WARNING: ANTHROPIC_API_KEY not set!")
    print("Create a .env file with: ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import examples
from examples.full_e2e_demo import run_full_e2e_demo
from evaluation import credit_risk_eval


def run_demo(mode: str, s3_bucket: str = None):
    """Run single sample demo"""
    print("\n" + "="*70)
    print("Running Single Sample Demo")
    print("="*70)
    
    from examples.full_e2e_demo import run_full_e2e_demo
    run_full_e2e_demo(mode=mode, s3_bucket=s3_bucket)


def run_evaluation(mode: str, s3_bucket: str = None):
    """Run full evaluation suite"""
    print("\n" + "="*70)
    print(f"Running Full Evaluation Suite - {mode.upper()} Mode")
    print("="*70)
    
    from evaluation.credit_risk_eval import CreditRiskEvaluator
    
    evaluator = CreditRiskEvaluator(mode=mode)
    
    start_time = time.time()
    results = evaluator.run_full_evaluation()
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    
    if "summary" in results:
        summary = results["summary"]
        print(f"\nMode: {summary['mode']}")
        print(f"Tests Run: {summary['total_tests']}")
        print(f"Tests Passed: {summary['passed']}/{summary['total_tests']}")
        print(f"Production Ready: {summary['production_ready']}")
    
    print(f"\nElapsed Time: {elapsed_time:.1f} seconds")
    
    # Save results
    output_file = f"evaluation_results_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # SageMaker: Upload to S3
    if mode == "sagemaker" and s3_bucket:
        try:
            import boto3
            s3 = boto3.client('s3')
            s3.upload_file(output_file, s3_bucket, f"results/{output_file}")
            print(f"✓ Results uploaded to: s3://{s3_bucket}/results/{output_file}")
        except Exception as e:
            print(f"⚠ Could not upload to S3: {e}")

def run_evaluation_sagemaker(s3_bucket: str):
    """Run evaluation with S3 integration"""
    
    print("="*60)
    print("SageMaker Evaluation with S3 Integration")
    print("="*60)
    
    # Step 1: Ensure reference data in S3
    from credit_risk.monitoring.data_drift import DataDriftDetector
    
    detector = DataDriftDetector(
        mode="sagemaker",
        s3_bucket=s3_bucket,
        s3_reference_key="data/reference/training_data.csv"
    )
    
    # Check if reference data exists, upload if not
    try:
        # Try to load (will auto-download from S3 if exists)
        print("\n1. Loading reference data from S3...")
        ref_data = detector.reference_data
        print(f"✓ Reference data loaded: {len(ref_data)} samples")
    except:
        # Doesn't exist, upload it
        print("\n1. Reference data not found, uploading...")
        
        # Load local training data
        training_df = pd.read_csv(
            'data/credit_risk/lending_club/lending_club.csv',
            nrows=10000  # Sample for demo
        )
        
        detector_upload = DataDriftDetector(mode="sagemaker", s3_bucket=s3_bucket)
        detector_upload.upload_reference_data_to_s3(
            training_df,
            "data/reference/training_data.csv"
        )
        print("✓ Reference data uploaded")
    
    # Step 2: Run evaluation
    print("\n2. Running credit risk evaluation...")
    from evaluation.credit_risk_eval import CreditRiskEvaluator
    
    evaluator = CreditRiskEvaluator(mode="sagemaker")
    results = evaluator.run_full_evaluation()
    
    print(f"\n✓ Evaluation complete: {results['summary']['passed']}/{results['summary']['total_tests']} tests passed")
    
    # Step 3: Results are auto-saved to S3 by components
    print("\n3. Results saved to S3:")
    print(f"   - Risk memos: s3://{s3_bucket}/risk_memos/")
    print(f"   - Drift reports: s3://{s3_bucket}/drift_reports/")
    print(f"   - Models: s3://{s3_bucket}/models/")

def main():
    parser = argparse.ArgumentParser(
        description="Run E2E OCR→RAG→Vision→Credit Risk Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-run test (no API costs, mock LLM responses)
  python run_e2e.py --dry-run
  
  # Local mode with real API calls
  python run_e2e.py --mode local
  
  # Local evaluation (80 samples, real API)
  python run_e2e.py --eval
  
  # SageMaker first-run test (no API costs)
  python run_e2e.py --mode sagemaker --s3-bucket my-bucket --dry-run
  
  # SageMaker evaluation (600 samples, real API)
  python run_e2e.py --mode sagemaker --s3-bucket my-bucket --eval
        """
    )
    
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "sagemaker", "production"],
        help="Execution mode (default: local)"
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run full evaluation (vs single demo)"
    )
    
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="S3 bucket for SageMaker mode"
    )
    
    # NEW: Dry-run mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without API calls (no costs, uses mock responses)"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for dry-run mode
    if args.dry_run:
        os.environ["DRY_RUN_MODE"] = "true"
        print("🧪 DRY-RUN MODE: No API calls will be made (testing pipeline only)")
        print("="*70)
    
    # Validation
    if args.mode == "sagemaker" and not args.s3_bucket:
        print("ERROR: --s3-bucket required for SageMaker mode")
        print("Example: python run_e2e.py --mode sagemaker --s3-bucket my-bucket")
        return
    
    # Run
    if args.mode == "sagemaker":
        run_evaluation_sagemaker(s3_bucket=args.s3_bucket)
    else:
        from examples.full_e2e_demo import run_full_e2e_demo
        run_full_e2e_demo(mode=args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()