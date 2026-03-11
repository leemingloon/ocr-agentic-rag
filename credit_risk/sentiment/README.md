# FinBERT Sentiment Pipeline — Failure-Mode Fixes

Modular fixes for documented FinBERT failure modes (negation, conditional, numeric comparison, hedged language, entity-specific sentiment). Each fix is toggleable via `SentimentPipelineConfig`.

## Components

| Fix | Module | Description |
|-----|--------|-------------|
| **Negation** | `negation.py` | spaCy dependency parsing (`neg`), regex fallback; preprocess text (flip phrases) and/or flip label in scope. |
| **Conditional** | `conditional.py` | Detect if/unless/would/could/might; confidence &lt; 0.75 → reclassify as neutral. |
| **Numeric comparison** | `numeric_comparison.py` | Regex "X vs expected Y"; % deviation &lt; -5% → inject negative signal. |
| **Hedging** | `hedging.py` | Financial hedge lexicon; high hedge intensity → widen neutral band. |
| **Entity (ABSA)** | `absa.py` | spaCy NER (ORG, PERSON, PRODUCT); ±3 token context; FinBERT per entity. |

## Config (toggle each fix)

```python
from credit_risk.sentiment import SentimentPipeline, SentimentPipelineConfig

config = SentimentPipelineConfig(
    use_negation_handling=True,
    use_conditional_detection=True,
    use_numeric_comparison=True,
    use_hedge_adjustment=True,
    use_entity_sentiment=True,
    conditional_confidence_threshold=0.75,
    numeric_underperform_pct=5.0,
    model_path="models/sentiment/finbert_tuned_v1",  # or None for ProsusAI/finbert
)
pipe = SentimentPipeline(config=config)
out = pipe.predict("Revenue of $2.1B vs expected $2.4B")
# out["sentence_sentiment"], out["entity_sentiments"], out["confidence"], out["flags"]
```

## Output (downstream PD)

- `sentence_sentiment`: "positive" | "negative" | "neutral"
- `entity_sentiments`: list of `{entity, label, context, sentiment, score}`
- `confidence`: float
- `flags`: `{negation, conditional, numeric_comparison, hedged}` (bool)
- `sentiment_score`, `sentiment_confidence`, `sentiment_flags` for PD feature pipeline

## Evaluation (before/after F1)

- **Notebook 04a** (Section 6): Runs pipeline with fixes on vs off on the test set; reports F1 on **full** and on **subsets** (negation, conditional, numeric_comparison, hedged). Saves `data/proof/credit_risk_sentiment/eval_failure_modes.json`.
- **Notebook 03**: Adds failure-mode columns (`has_negation`, `is_conditional`, etc.) and optional augmented train (`train_augmented.parquet`) for 04a.

## Dependencies

- `transformers`, `torch` (required)
- `spacy` + `en_core_web_sm` (optional; improves negation/conditional/ABSA):  
  `pip install spacy && python -m spacy download en_core_web_sm`
