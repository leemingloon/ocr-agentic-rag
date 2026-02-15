"""
Credit Risk Deterioration Pipeline

Converts unstructured borrower data → structured risk features → early warning signals

Pipeline:
1. OCR → Extract financial statements
2. RAG → Retrieve relevant context
3. Vision → Extract charts/graphs
4. Feature Engineering → Build risk features
5. PD Model → Predict default probability
6. Risk Memo → Generate LLM-based memo

Evaluation Datasets (Tier 1 + Tier 2):

Tier 1 (Must-Have):
- Lending Club (2.9M loans) - PD model
- FiQA Sentiment (1,173 samples) - NLP signals
- FinanceBench (10,231 Q&A) - Risk memo Q&A validation
- ECTSum (2,425 summaries) - Risk memo summarization validation
- Credit Card Default UCI (30K) - Drift detection

Tier 2 (Nice-to-Have):
- Freddie Mac Single-Family Loan (500K+) - Migration/PD
- Home Credit Default Risk (300K) - Feature engineering
- Synthetic counterfactuals (1,000) - What-if analysis

Total: 8 datasets, ~3.7M samples (full evaluation)
Local/SageMaker: Subset sampling for resource constraints
"""

from .feature_engineering.ratio_builder import RatioBuilder
from .feature_engineering.trend_engine import TrendEngine
from .feature_engineering.nlp_signals import NLPSignalExtractor
from .models.pd_model import PDModel
from .models.counterfactual import CounterfactualAnalyzer
from .governance.risk_memo_generator import RiskMemoGenerator
from .governance.prompt_registry import PromptRegistry
from .governance.safety_filter import SafetyFilter
from .monitoring.data_drift import DataDriftDetector
from .monitoring.prediction_drift import PredictionDriftDetector

__version__ = "1.0.0"

# Evaluation dataset configurations
EVALUATION_DATASETS = {
    "tier1": {
        "lending_club": {"samples": 2900000, "local": 10, "sagemaker": 100},
        "fiqa_sentiment": {"samples": 1173, "local": 10, "sagemaker": 100},
        "financebench": {"samples": 10231, "local": 10, "sagemaker": 100},
        "ectsum": {"samples": 2425, "local": 10, "sagemaker": 50},
        "credit_card_uci": {"samples": 30000, "local": 10, "sagemaker": 100},
    },
    "tier2": {
        "freddie_mac": {"samples": 500000, "local": 10, "sagemaker": 100},
        "home_credit": {"samples": 307511, "local": 10, "sagemaker": 100},
        "counterfactual_synthetic": {"samples": 1000, "local": 10, "sagemaker": 50},
    }
}

__all__ = [
    "RatioBuilder",
    "TrendEngine",
    "NLPSignalExtractor",
    "PDModel",
    "CounterfactualAnalyzer",
    "RiskMemoGenerator",
    "PromptRegistry",
    "SafetyFilter",
    "DataDriftDetector",
    "PredictionDriftDetector",
    "CreditRiskPipeline",
    "EVALUATION_DATASETS",
]