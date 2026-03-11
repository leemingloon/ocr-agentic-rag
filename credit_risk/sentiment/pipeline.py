"""
Unified FinBERT sentiment pipeline with failure-mode fixes.

Accepts raw financial text; returns structured output:
  sentence_sentiment, entity_sentiments, confidence, flags (negation, conditional, numeric_comparison, hedged).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

from credit_risk.sentiment.config import SentimentPipelineConfig
from credit_risk.sentiment.negation import NegationHandler, flip_label_in_scope
from credit_risk.sentiment.conditional import ConditionalDetector
from credit_risk.sentiment.numeric_comparison import NumericComparisonExtractor
from credit_risk.sentiment.hedging import HedgeScorer
from credit_risk.sentiment.absa import ABSALayer


def _softmax(x):
    import math
    e = [math.exp(v - max(x)) for v in x]
    return [v / sum(e) for v in e]


class SentimentPipeline:
    """
    Single entry point for sentiment with optional failure-mode fixes.
    Output: { sentence_sentiment, entity_sentiments, confidence, flags }.
    """

    def __init__(self, config: Optional[SentimentPipelineConfig] = None):
        self.config = config or SentimentPipelineConfig()
        self._tokenizer = None
        self._model = None
        self._negation = NegationHandler() if self.config.use_negation_handling else None
        self._conditional = ConditionalDetector() if self.config.use_conditional_detection else None
        self._numeric = NumericComparisonExtractor(
            underperform_pct=self.config.numeric_underperform_pct
        ) if self.config.use_numeric_comparison else None
        self._hedge = HedgeScorer() if self.config.use_hedge_adjustment else None
        self._absa = None  # set after model load so we can pass sentiment_fn

    def _ensure_model(self):
        if self._model is not None:
            return
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers and torch required for SentimentPipeline")
        model_path = self.config.model_path or self.config.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path or self.config.model_name
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3
        )
        self._model.eval()
        if self.config.use_entity_sentiment:
            self._absa = ABSALayer(
                context_window=self.config.entity_context_window,
                sentiment_fn=self._predict_single,
            )

    def _predict_single(self, text: str) -> Dict[str, Any]:
        """Raw FinBERT prediction for one string. Returns { label, score, logits }."""
        self._ensure_model()
        enc = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self._model(**enc)
        logits = out.logits[0].tolist()
        probs = _softmax(logits)
        idx = max(range(3), key=lambda i: logits[i])
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        label = id2label.get(idx, "neutral")
        score = float(probs[idx])
        return {"label": label, "score": score, "logits": logits, "probs": probs}

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Full pipeline: negation preprocessing, FinBERT, conditional/hedge/numeric adjustments,
        optional ABSA. Returns structured output for downstream (PD features).
        """
        if not text or not str(text).strip():
            return {
                "sentence_sentiment": "neutral",
                "entity_sentiments": [],
                "confidence": 0.5,
                "flags": {},
            }
        text = str(text).strip()
        flags = {}

        # 1) Negation: preprocess input to reduce misclassification
        run_text = text
        if self._negation and self._negation.has_negation(text):
            flags["negation"] = True
            run_text = self._negation.preprocess_for_sentiment(text)
        else:
            flags["negation"] = False

        self._ensure_model()
        raw = self._predict_single(run_text)
        label = raw["label"]
        confidence = raw["score"]
        probs = raw.get("probs", [0.33, 0.34, 0.33])
        id2label = {0: "negative", 1: "neutral", 2: "positive"}

        # 2) Negation: we already preprocessed run_text; only flip label if we did NOT preprocess
        # (e.g. when preprocessing had no effect) — then flip model output in scope
        if self._negation and flags.get("negation") and run_text == text and label != "neutral":
            label = flip_label_in_scope(label)

        # 3) Conditional: low confidence → neutral
        if self._conditional and self._conditional.is_conditional(text):
            flags["conditional"] = True
            if confidence < self.config.conditional_confidence_threshold:
                label = "neutral"
                confidence = min(confidence, 0.5)
        else:
            flags["conditional"] = False

        # 4) Numeric comparison: underperform → push toward negative
        if self._numeric:
            num_info = self._numeric.extract(text)
            if num_info:
                flags["numeric_comparison"] = True
                if num_info.get("is_underperform"):
                    # Override or blend: set to negative if current is positive
                    if label == "positive":
                        label = "negative"
                        confidence = max(0.6, confidence)  # signal we're confident it's bad
            else:
                flags["numeric_comparison"] = False
        else:
            flags["numeric_comparison"] = False

        # 5) Hedge: widen neutral band — require higher confidence for pos/neg
        if self._hedge:
            intensity = self._hedge.intensity(text)
            flags["hedged"] = intensity > 0
            if intensity > 0.3:
                # Require higher confidence to keep pos/neg
                threshold = 0.5 + intensity * self.config.hedge_neutral_band_scale * 0.1
                if label != "neutral" and confidence < min(threshold, 0.75):
                    label = "neutral"
                    confidence = 0.5
        else:
            flags["hedged"] = False

        # 6) Entity-level sentiment (ABSA)
        entity_sentiments = []
        if self._absa and self._absa.available:
            entity_sentiments = self._absa.run_entity_sentiment(text)

        # Map label to numeric score for PD pipeline: negative -1..0, neutral 0, positive 0..1
        score_map = {"negative": -0.5, "neutral": 0.0, "positive": 0.5}
        sentiment_score = score_map.get(label, 0.0)
        if label == "negative":
            sentiment_score = -max(confidence * 0.5, 0.2)
        elif label == "positive":
            sentiment_score = max(confidence * 0.5, 0.2)

        return {
            "sentence_sentiment": label,
            "entity_sentiments": entity_sentiments,
            "confidence": round(confidence, 4),
            "flags": flags,
            "sentiment_score": sentiment_score,
            "sentiment_confidence": confidence,
            "sentiment_flags": flags,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Run predict on each text. For large batches, consider batching FinBERT forward passes."""
        return [self.predict(t) for t in texts]
