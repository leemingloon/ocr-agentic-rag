"""Config for FinBERT sentiment pipeline — toggles for each failure-mode fix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SentimentPipelineConfig:
    """Toggles and parameters for the sentiment pipeline."""

    # Failure-mode fixes (each can be toggled off)
    use_negation_handling: bool = True
    use_conditional_detection: bool = True
    use_numeric_comparison: bool = True
    use_hedge_adjustment: bool = True
    use_entity_sentiment: bool = True

    # Conditional: reclassify as neutral if confidence < threshold on conditional sentences
    conditional_confidence_threshold: float = 0.75

    # Numeric comparison: treat as negative if deviation < -pct (e.g. -5%)
    numeric_underperform_pct: float = 5.0

    # Hedge: require higher confidence for pos/neg when hedge intensity is high
    hedge_neutral_band_scale: float = 1.5  # scale neutral band by this when strong hedge

    # ABSA: context window around entity (tokens)
    entity_context_window: int = 3

    # Model
    model_name: str = "ProsusAI/finbert"
    model_path: str | None = None  # local path for fine-tuned (e.g. models/sentiment/finbert_tuned_v1)
    max_length: int = 128

    def to_dict(self) -> dict:
        return {
            "use_negation_handling": self.use_negation_handling,
            "use_conditional_detection": self.use_conditional_detection,
            "use_numeric_comparison": self.use_numeric_comparison,
            "use_hedge_adjustment": self.use_hedge_adjustment,
            "use_entity_sentiment": self.use_entity_sentiment,
            "conditional_confidence_threshold": self.conditional_confidence_threshold,
            "numeric_underperform_pct": self.numeric_underperform_pct,
            "hedge_neutral_band_scale": self.hedge_neutral_band_scale,
            "entity_context_window": self.entity_context_window,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "max_length": self.max_length,
        }
