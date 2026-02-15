# credit_risk/pipeline.py

"""
Credit Risk Pipeline - End-to-End Orchestration

Integrates:
- OCR pipeline (upstream)
- RAG system (upstream)  
- Multimodal vision (upstream)
- Credit risk models (this pipeline)

Output:
- PD score
- Risk drivers
- LLM-generated risk memo
- Early warning signals

Evaluation:
- Full suite: 8 datasets, 3.7M samples
- Local mode: 80 samples total (10 per dataset)
- SageMaker mode: 600 samples total (50-100 per dataset)
"""

import os
import boto3
from typing import Literal
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal
from pathlib import Path

from .feature_engineering.ratio_builder import RatioBuilder
from .feature_engineering.trend_engine import TrendEngine
from .feature_engineering.nlp_signals import NLPSignalExtractor
from .models.pd_model import PDModel
from .models.counterfactual import CounterfactualAnalyzer
from .governance.risk_memo_generator import RiskMemoGenerator
from .monitoring.data_drift import DataDriftDetector
from .monitoring.prediction_drift import PredictionDriftDetector


class CreditRiskPipeline:
    """
    End-to-end credit risk pipeline
    
    Modes:
    - production: Full datasets (3.7M samples)
    - sagemaker: AWS free tier (600 samples)
    - local: Quick test (80 samples)
    
    Usage:
        # Production mode
        pipeline = CreditRiskPipeline(mode="production")
        
        # Local mode (fast)
        pipeline = CreditRiskPipeline(mode="local")
        
        result = pipeline.process(
            ocr_output=ocr_result,
            rag_context=rag_result,
            vision_charts=vision_result
        )
        
        print(result["pd"])           # 0.08 (8% default probability)
        print(result["risk_memo"])    # LLM-generated memo
        print(result["drivers"])      # ["debt_to_ebitda", "news_sentiment"]
    """
    
    def __init__(
        self,
        pd_model_path: Optional[str] = None,
        reference_data_path: Optional[str] = None,
        mode: Literal["production", "sagemaker", "local"] = "local",
        s3_bucket: Optional[str] = None,  # NEW for SageMaker
    ):
        """
        Initialize credit risk pipeline
        
        Args:
            pd_model_path: Path to trained PD model
            reference_data_path: Path to reference data for drift detection
            mode: Execution mode (production/sagemaker/local)
            s3_bucket: S3 bucket for SageMaker (e.g., 'my-sagemaker-bucket')
        """
        self.mode = mode
        self.s3_bucket = s3_bucket

        # SageMaker-specific setup
        if mode == "sagemaker":
            self._setup_sagemaker()
        
        # Feature engineering
        self.ratio_builder = RatioBuilder()
        self.trend_engine = TrendEngine()
        self.nlp_extractor = NLPSignalExtractor()
        
        # Models
        self.pd_model = PDModel()
        if pd_model_path:
            self.pd_model.load(pd_model_path)
        
        self.counterfactual = CounterfactualAnalyzer(self.pd_model)
        
        # Governance
        self.risk_memo_generator = RiskMemoGenerator()
        
        # Monitoring
        if reference_data_path:
            reference_data = pd.read_csv(reference_data_path)
            self.data_drift_detector = DataDriftDetector(reference_data)
            self.prediction_drift_detector = PredictionDriftDetector()
        else:
            self.data_drift_detector = None
            self.prediction_drift_detector = None
        
        # Evaluation sample sizes based on mode
        from . import EVALUATION_DATASETS
        self.eval_config = EVALUATION_DATASETS
    
    def _setup_sagemaker(self):
        """Setup SageMaker environment"""
        # Check if running on SageMaker
        if "SM_CHANNEL_TRAINING" in os.environ:
            print("✓ Running on SageMaker")
            self.data_dir = Path(os.environ["SM_CHANNEL_TRAINING"])
        else:
            print("⚠ SageMaker mode enabled but not running on SageMaker instance")
            self.data_dir = Path("data/credit_risk")
        
        # Initialize S3 client (for saving results)
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                print(f"✓ S3 bucket configured: {self.s3_bucket}")
            except Exception as e:
                print(f"⚠ Could not initialize S3 client: {e}")
                self.s3_client = None
        else:
            self.s3_client = None

    def get_sample_size(self, dataset_name: str, tier: str = "tier1") -> int:
        """
        Get appropriate sample size for current mode
        
        Args:
            dataset_name: Name of dataset
            tier: "tier1" or "tier2"
            
        Returns:
            Sample size for current mode
        """
        dataset_config = self.eval_config[tier].get(dataset_name, {})
        
        if self.mode == "production":
            return dataset_config.get("samples", 0)
        elif self.mode == "sagemaker":
            return dataset_config.get("sagemaker", 100)
        else:  # local
            return dataset_config.get("local", 10)
    
    def process(
        self,
        borrower: str,
        ocr_output: Dict,
        rag_context: Optional[Dict] = None,
        vision_charts: Optional[Dict] = None,
        news_articles: Optional[List[str]] = None,
        historical_financials: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Process borrower data through credit risk pipeline
        
        Args:
            borrower: Borrower name/ID
            ocr_output: OCR-extracted financial statements
            rag_context: RAG-retrieved context (optional)
            vision_charts: Vision-extracted charts (optional)
            news_articles: News articles for NLP sentiment (optional)
            historical_financials: Historical financials for trend analysis (optional)
            
        Returns:
            {
                "borrower": str,
                "pd": float,
                "drivers": List[str],
                "risk_memo": str,
                "features": Dict,
                "counterfactuals": Dict,
                "drift_detected": bool,
                "mode": str,  # Added for tracking
            }
        """
        # Step 1: Extract financial data from OCR
        financials = self._extract_financials(ocr_output)
        
        # Step 2: Feature engineering
        features = self._build_features(
            financials,
            news_articles,
            historical_financials,
            vision_charts
        )
        
        # Step 3: Drift detection (if enabled)
        drift_result = None
        if self.data_drift_detector:
            drift_result = self.data_drift_detector.detect_drift(
                pd.DataFrame([features])
            )
        
        # Step 4: PD prediction
        pd_score = self.pd_model.predict_pd(features)
        
        # Step 5: Identify risk drivers (SHAP)
        drivers = self.pd_model.get_top_drivers(features, top_n=5)
        
        # Step 6: Counterfactual analysis
        counterfactuals = self._run_counterfactuals(features)
        
        # Step 7: Generate risk memo
        risk_memo = self.risk_memo_generator.generate_memo(
            borrower=borrower,
            features=features,
            pd=pd_score,
            drivers=drivers,
        )
        
        # Step 8: Prediction drift (if enabled)
        if self.prediction_drift_detector:
            self.prediction_drift_detector.log_prediction(pd_score)
        
        return {
            "borrower": borrower,
            "pd": pd_score,
            "drivers": drivers,
            "risk_memo": risk_memo,
            "features": features,
            "counterfactuals": counterfactuals,
            "drift_detected": drift_result.get("drift_detected", False) if drift_result else False,
            "drift_details": drift_result if drift_result else None,
            "mode": self.mode,  # Track which mode was used
        }
    
    def _extract_financials(self, ocr_output: Dict) -> Dict:
        """Extract structured financials from OCR output"""
        # Parse OCR text to extract financial metrics
        text = ocr_output.get("text", "")
        
        # Simple regex-based extraction (would use more sophisticated NER in production)
        import re
        
        financials = {}
        
        # Revenue
        revenue_match = re.search(r'revenue[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if revenue_match:
            financials["revenue"] = float(revenue_match.group(1).replace(',', ''))
        
        # EBITDA
        ebitda_match = re.search(r'ebitda[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if ebitda_match:
            financials["ebitda"] = float(ebitda_match.group(1).replace(',', ''))
        
        # Total Debt
        debt_match = re.search(r'total debt[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if debt_match:
            financials["total_debt"] = float(debt_match.group(1).replace(',', ''))
        
        # Interest Expense
        interest_match = re.search(r'interest expense[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if interest_match:
            financials["interest_expense"] = float(interest_match.group(1).replace(',', ''))
        
        # Current Assets
        current_assets_match = re.search(r'current assets[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if current_assets_match:
            financials["current_assets"] = float(current_assets_match.group(1).replace(',', ''))
        
        # Current Liabilities
        current_liab_match = re.search(r'current liabilities[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if current_liab_match:
            financials["current_liabilities"] = float(current_liab_match.group(1).replace(',', ''))
        
        # Inventory
        inventory_match = re.search(r'inventory[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if inventory_match:
            financials["inventory"] = float(inventory_match.group(1).replace(',', ''))
        
        # Equity
        equity_match = re.search(r'equity[:\s]+\$?([\d,]+)', text, re.IGNORECASE)
        if equity_match:
            financials["equity"] = float(equity_match.group(1).replace(',', ''))
        
        return financials
    
    def _build_features(
        self,
        financials: Dict,
        news_articles: Optional[List[str]],
        historical_financials: Optional[pd.DataFrame],
        vision_charts: Optional[Dict],
    ) -> Dict:
        """Build feature vector for PD model"""
        features = {}
        
        # 1. Financial ratios
        if financials:
            ratios = self.ratio_builder.extract_ratios(financials)
            features.update(ratios)
        
        # 2. Trend signals
        if historical_financials is not None:
            trends = self.trend_engine.detect_deterioration(historical_financials)
            features.update(trends)
        
        # 3. NLP sentiment
        if news_articles:
            nlp_signals = self.nlp_extractor.extract_signals(news_articles)
            features["news_sentiment"] = nlp_signals["news_sentiment"]["score"]
        else:
            features["news_sentiment"] = 0.0  # Neutral
        
        # 4. Chart-derived features (if available)
        if vision_charts:
            # Extract features from charts (e.g., revenue trend slope)
            features["chart_trend"] = vision_charts.get("trend", 0.0)
        
        return features
    
    def _run_counterfactuals(self, baseline_features: Dict) -> Dict:
        """Run counterfactual analysis"""
        counterfactuals = {}
        
        # Scenario 1: Debt/EBITDA increases to 4.0x
        if "debt_to_ebitda" in baseline_features:
            cf_debt = self.counterfactual.what_if(
                baseline_features,
                {"debt_to_ebitda": 4.0}
            )
            counterfactuals["debt_to_ebitda_4x"] = cf_debt
        
        # Scenario 2: Interest coverage drops to 2.0x
        if "interest_coverage" in baseline_features:
            cf_coverage = self.counterfactual.what_if(
                baseline_features,
                {"interest_coverage": 2.0}
            )
            counterfactuals["interest_coverage_2x"] = cf_coverage
        
        # Scenario 3: News sentiment deteriorates to -0.5
        if "news_sentiment" in baseline_features:
            cf_sentiment = self.counterfactual.what_if(
                baseline_features,
                {"news_sentiment": -0.5}
            )
            counterfactuals["news_sentiment_negative"] = cf_sentiment
        
        return counterfactuals


# Example usage
if __name__ == "__main__":
    # Local mode (fast testing)
    pipeline_local = CreditRiskPipeline(mode="local")
    print(f"Local mode sample sizes:")
    print(f"  Lending Club: {pipeline_local.get_sample_size('lending_club')} samples")
    print(f"  FinanceBench: {pipeline_local.get_sample_size('financebench')} samples")
    
    # SageMaker mode (AWS free tier)
    pipeline_sagemaker = CreditRiskPipeline(mode="sagemaker")
    print(f"\nSageMaker mode sample sizes:")
    print(f"  Lending Club: {pipeline_sagemaker.get_sample_size('lending_club')} samples")
    print(f"  FinanceBench: {pipeline_sagemaker.get_sample_size('financebench')} samples")
    
    # Production mode (full evaluation)
    pipeline_prod = CreditRiskPipeline(mode="production")
    print(f"\nProduction mode sample sizes:")
    print(f"  Lending Club: {pipeline_prod.get_sample_size('lending_club')} samples")
    print(f"  FinanceBench: {pipeline_prod.get_sample_size('financebench')} samples")