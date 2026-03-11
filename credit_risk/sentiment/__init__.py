"""
FinBERT sentiment pipeline with failure-mode fixes.

Modular components (each toggleable via SentimentPipelineConfig):
- Negation handling (spaCy dependency parsing, polarity flip in scope)
- Conditional sentiment (confidence penalty on conditional clauses)
- Numeric comparison context (X vs expected Y → deviation-based signal)
- Hedged language (hedge intensity → neutral band widening)
- Entity-specific sentiment (ABSA via spaCy NER + context windows)

Output: {sentence_sentiment, entity_sentiments, confidence, flags}
Downstream: sentiment_score, sentiment_confidence, sentiment_flags for PD feature pipeline.
"""

from credit_risk.sentiment.config import SentimentPipelineConfig
from credit_risk.sentiment.pipeline import SentimentPipeline

__all__ = ["SentimentPipelineConfig", "SentimentPipeline"]
