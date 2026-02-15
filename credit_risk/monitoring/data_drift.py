"""
Data Drift Detection

Monitors input feature distribution for shifts that indicate model degradation.

Method: Kolmogorov-Smirnov (KS) test
Threshold: p-value < 0.05 → Significant drift
Action: >10% of features drifted → Retrain required

Evaluation Dataset: Credit Card Default (UCI) - 30K samples with temporal data

SageMaker Integration:
- Load reference data from S3
- Save drift reports to S3
- Automatic local caching
- Batch drift detection

Usage:
    # Local mode (reference data from local file)
    detector = DataDriftDetector(
        reference_data=reference_df,
        mode="local"
    )
    
    # SageMaker mode (reference data from S3)
    detector = DataDriftDetector(
        mode="sagemaker",
        s3_bucket="my-bucket",
        s3_reference_key="data/reference/training_data.csv"
    )
    
    # Detect drift
    drift_result = detector.detect_drift(current_data)
    
    if drift_result["drift_detected"]:
        print("⚠ Drift detected! Retrain model.")
        
    # Save report to S3 (SageMaker mode)
    detector.save_drift_report_to_s3(drift_result, "drift_report_20260215.json")
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal
from scipy.stats import ks_2samp
from pathlib import Path
import json
from datetime import datetime

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("Warning: boto3 not available (S3 disabled)")


class DataDriftDetector:
    """
    Detect distribution shifts in input features
    
    Uses KS test to compare current data vs reference (training) data.
    
    SageMaker Features:
    - S3 reference data loading
    - S3 report saving
    - Automatic caching
    - Batch processing
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        mode: Literal["local", "sagemaker", "production"] = "local",
        s3_bucket: Optional[str] = None,
        s3_reference_key: Optional[str] = None,
        s3_reports_prefix: str = "drift_reports/",
        local_cache_dir: str = "data/drift_cache"
    ):
        """
        Initialize drift detector
        
        Args:
            reference_data: Training data distribution (baseline)
            mode: Execution mode (local/sagemaker/production)
            s3_bucket: S3 bucket for data/reports (SageMaker mode)
            s3_reference_key: S3 key for reference data CSV
            s3_reports_prefix: S3 prefix for drift reports
            local_cache_dir: Local directory for caching reference data
        """
        self.mode = mode
        self.s3_bucket = s3_bucket
        self.s3_reference_key = s3_reference_key
        self.s3_reports_prefix = s3_reports_prefix
        self.local_cache_dir = Path(local_cache_dir)
        
        # Create cache directory
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if available
        if mode == "sagemaker" and BOTO3_AVAILABLE and s3_bucket:
            self.s3_client = boto3.client('s3')
            print(f"✓ S3 client initialized: s3://{s3_bucket}")
        else:
            self.s3_client = None
        
        # Load reference data
        if reference_data is not None:
            # Use provided reference data
            self.reference_data = reference_data
            print(f"✓ Reference data loaded: {len(reference_data)} samples, {len(reference_data.columns)} features")
        elif mode == "sagemaker" and s3_reference_key:
            # Load from S3
            self.reference_data = self._load_reference_from_s3()
        else:
            self.reference_data = None
            print("⚠ No reference data provided")
        
        # Extract feature names
        self.feature_names = list(self.reference_data.columns) if self.reference_data is not None else []
    
    def _load_reference_from_s3(self) -> pd.DataFrame:
        """Load reference data from S3 with local caching"""
        if not self.s3_client or not self.s3_reference_key:
            raise ValueError("S3 client or reference key not configured")
        
        # Check local cache first
        cache_file = self.local_cache_dir / "reference_data.csv"
        
        if cache_file.exists():
            print(f"✓ Loading reference data from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # Download from S3
        print(f"⚠ Downloading reference data from S3: s3://{self.s3_bucket}/{self.s3_reference_key}")
        
        try:
            self.s3_client.download_file(
                self.s3_bucket,
                self.s3_reference_key,
                str(cache_file)
            )
            
            print(f"✓ Reference data downloaded and cached")
            
            # Load and return
            reference_df = pd.read_csv(cache_file)
            
            print(f"✓ Reference data loaded: {len(reference_df)} samples, {len(reference_df.columns)} features")
            
            return reference_df
        
        except Exception as e:
            print(f"✗ Error downloading reference data from S3: {e}")
            raise
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict:
        """
        Detect drift in current batch vs reference
        
        Args:
            current_data: Current batch of data
            alpha: Significance level (default: 0.05)
            
        Returns:
            {
                "drift_detected": bool,
                "drifted_features": List[str],
                "feature_results": Dict,
                "action": str,
                "timestamp": str,
            }
        """
        if self.reference_data is None:
            raise ValueError("No reference data available. Provide reference_data or load from S3")
        
        drift_results = {}
        
        for col in self.feature_names:
            if col not in current_data.columns:
                print(f"⚠ Feature '{col}' missing in current data, skipping")
                continue
            
            # KS test
            statistic, p_value = ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            drift_results[col] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drifted": p_value < alpha,
            }
        
        # Aggregate
        drifted_features = [
            col for col, result in drift_results.items()
            if result["drifted"]
        ]
        
        drift_detected = len(drifted_features) > 0.10 * len(drift_results)
        
        # Determine action
        if drift_detected:
            action = "RETRAIN_REQUIRED"
        elif len(drifted_features) > 0:
            action = "MONITOR_CLOSELY"
        else:
            action = "NO_ACTION"
        
        return {
            "drift_detected": drift_detected,
            "drifted_features": drifted_features,
            "num_drifted": len(drifted_features),
            "total_features": len(drift_results),
            "drift_percentage": len(drifted_features) / len(drift_results) * 100,
            "feature_results": drift_results,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "alpha": alpha,
            "reference_samples": len(self.reference_data),
            "current_samples": len(current_data),
        }
    
    def format_drift_report(self, drift_result: Dict) -> str:
        """Format drift detection results"""
        lines = ["Data Drift Detection Report"]
        lines.append("=" * 60)
        lines.append(f"\nTimestamp: {drift_result.get('timestamp', 'N/A')}")
        
        lines.append(f"\nDrift Detected: {drift_result['drift_detected']}")
        lines.append(f"Features Drifted: {drift_result['num_drifted']} / {drift_result['total_features']} ({drift_result['drift_percentage']:.1f}%)")
        lines.append(f"Action: {drift_result['action']}")
        
        lines.append(f"\nReference Samples: {drift_result.get('reference_samples', 'N/A')}")
        lines.append(f"Current Samples: {drift_result.get('current_samples', 'N/A')}")
        
        if drift_result['drifted_features']:
            lines.append("\nDrifted Features:")
            for feat in drift_result['drifted_features']:
                result = drift_result['feature_results'][feat]
                lines.append(f"  • {feat}: KS={result['statistic']:.3f}, p={result['p_value']:.4f}")
        else:
            lines.append("\n✓ No drifted features detected")
        
        return "\n".join(lines)
    
    # ========================================
    # S3 OPERATIONS (SAGEMAKER)
    # ========================================
    
    def save_drift_report_to_s3(
        self,
        drift_result: Dict,
        report_name: Optional[str] = None
    ):
        """
        Save drift report to S3
        
        Args:
            drift_result: Drift detection results
            report_name: Report filename (e.g., "drift_report_20260215.json")
                        If None, auto-generates timestamp-based name
        """
        if not self.s3_client:
            raise ValueError("S3 client not initialized. Set mode='sagemaker' and provide s3_bucket")
        
        # Generate report name if not provided
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"drift_report_{timestamp}.json"
        
        # Save to local file first
        local_file = self.local_cache_dir / report_name
        
        with open(local_file, 'w') as f:
            json.dump(drift_result, f, indent=2)
        
        # Upload to S3
        s3_key = f"{self.s3_reports_prefix}{report_name}"
        
        try:
            self.s3_client.upload_file(
                str(local_file),
                self.s3_bucket,
                s3_key
            )
            
            print(f"✓ Drift report saved to S3: s3://{self.s3_bucket}/{s3_key}")
            
        except Exception as e:
            print(f"✗ Error uploading drift report to S3: {e}")
            raise
    
    def load_drift_report_from_s3(self, report_name: str) -> Dict:
        """
        Load drift report from S3
        
        Args:
            report_name: Report filename
            
        Returns:
            Drift detection results
        """
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        s3_key = f"{self.s3_reports_prefix}{report_name}"
        local_file = self.local_cache_dir / report_name
        
        try:
            self.s3_client.download_file(
                self.s3_bucket,
                s3_key,
                str(local_file)
            )
            
            with open(local_file, 'r') as f:
                drift_result = json.load(f)
            
            print(f"✓ Drift report loaded from S3: s3://{self.s3_bucket}/{s3_key}")
            
            return drift_result
        
        except Exception as e:
            print(f"✗ Error loading drift report from S3: {e}")
            raise
    
    def list_drift_reports_in_s3(self) -> List[Dict]:
        """
        List all drift reports in S3
        
        Returns:
            List of report info dicts
        """
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_reports_prefix
            )
            
            if 'Contents' not in response:
                return []
            
            reports = []
            for obj in response['Contents']:
                key = obj['Key']
                
                # Only include .json files
                if not key.endswith('.json'):
                    continue
                
                reports.append({
                    "s3_key": key,
                    "filename": key.replace(self.s3_reports_prefix, ''),
                    "size_kb": obj['Size'] / 1024,
                    "last_modified": obj['LastModified'].isoformat(),
                })
            
            return sorted(reports, key=lambda x: x['last_modified'], reverse=True)
        
        except Exception as e:
            print(f"✗ Error listing drift reports in S3: {e}")
            return []
    
    def upload_reference_data_to_s3(
        self,
        reference_data: pd.DataFrame,
        s3_key: Optional[str] = None
    ):
        """
        Upload reference data to S3 (for initialization)
        
        Args:
            reference_data: Reference dataset
            s3_key: S3 key for reference data (e.g., "data/reference/training_data.csv")
        """
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        if s3_key is None:
            s3_key = "data/reference/training_data.csv"
        
        # Save to local file first
        local_file = self.local_cache_dir / "reference_data_upload.csv"
        reference_data.to_csv(local_file, index=False)
        
        try:
            self.s3_client.upload_file(
                str(local_file),
                self.s3_bucket,
                s3_key
            )
            
            print(f"✓ Reference data uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            print(f"  {len(reference_data)} samples, {len(reference_data.columns)} features")
            
        except Exception as e:
            print(f"✗ Error uploading reference data to S3: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # ========================================
    # LOCAL MODE
    # ========================================
    print("=" * 60)
    print("LOCAL MODE")
    print("=" * 60)
    
    # Reference data (training)
    np.random.seed(42)
    reference = pd.DataFrame({
        "debt_to_ebitda": np.random.normal(3.0, 0.5, 1000),
        "interest_coverage": np.random.normal(4.0, 1.0, 1000),
        "current_ratio": np.random.normal(1.5, 0.3, 1000),
    })
    
    # Current data (with drift in debt_to_ebitda)
    current = pd.DataFrame({
        "debt_to_ebitda": np.random.normal(3.5, 0.5, 100),  # Mean shifted
        "interest_coverage": np.random.normal(4.0, 1.0, 100),
        "current_ratio": np.random.normal(1.5, 0.3, 100),
    })
    
    detector = DataDriftDetector(reference_data=reference, mode="local")
    drift_result = detector.detect_drift(current)
    
    print("\n" + detector.format_drift_report(drift_result))
    
    # ========================================
    # SAGEMAKER MODE
    # ========================================
    print("\n\n" + "=" * 60)
    print("SAGEMAKER MODE")
    print("=" * 60)
    
    if BOTO3_AVAILABLE:
        # Step 1: Upload reference data to S3 (one-time setup)
        print("\n1. Uploading reference data to S3...")
        try:
            detector_setup = DataDriftDetector(
                mode="sagemaker",
                s3_bucket="my-sagemaker-bucket",  # Replace with your bucket
            )
            
            detector_setup.upload_reference_data_to_s3(
                reference_data=reference,
                s3_key="data/reference/training_data.csv"
            )
        except Exception as e:
            print(f"⚠ S3 upload failed (bucket may not exist): {e}")
        
        # Step 2: Initialize detector with S3 reference data
        print("\n2. Initializing detector with S3 reference data...")
        try:
            detector_sm = DataDriftDetector(
                mode="sagemaker",
                s3_bucket="my-sagemaker-bucket",
                s3_reference_key="data/reference/training_data.csv"
            )
            
            # Detect drift
            drift_result_sm = detector_sm.detect_drift(current)
            
            print("\n" + detector_sm.format_drift_report(drift_result_sm))
            
            # Step 3: Save drift report to S3
            print("\n3. Saving drift report to S3...")
            detector_sm.save_drift_report_to_s3(
                drift_result_sm,
                report_name="drift_report_20260215.json"
            )
            
            # Step 4: List all drift reports
            print("\n4. Listing drift reports in S3...")
            reports = detector_sm.list_drift_reports_in_s3()
            
            if reports:
                print(f"\nFound {len(reports)} drift reports:")
                for r in reports:
                    print(f"  • {r['filename']} ({r['size_kb']:.2f} KB) - {r['last_modified']}")
            else:
                print("No drift reports found in S3")
            
            # Step 5: Load a drift report from S3
            print("\n5. Loading drift report from S3...")
            loaded_report = detector_sm.load_drift_report_from_s3("drift_report_20260215.json")
            print(f"Loaded report: {loaded_report['num_drifted']} drifted features")
            
        except Exception as e:
            print(f"⚠ SageMaker demo failed: {e}")
    else:
        print("⚠ boto3 not available - skipping SageMaker demo")