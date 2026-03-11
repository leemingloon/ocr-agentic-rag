"""
Data augmentation for sentiment fine-tuning: negation-augmented and conditional (neutral) examples.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd

from credit_risk.sentiment.negation import NegationHandler, NEGATION_FLIP_PHRASES
from credit_risk.sentiment.conditional import ConditionalDetector


def add_negation_augmented(df: pd.DataFrame, text_col: str = "text", label_col: str = "label", max_extra: int = 300) -> pd.DataFrame:
    """
    For sentences that don't already contain negation, create a negated version and flip label.
    E.g. "The company missed earnings" (negative) -> "The company did not miss earnings" (positive).
    Simple rule: prepend "did not " before verb phrase or "not " before adjective; flip label.
    """
    out_rows = []
    seen = set()
    n_extra = 0
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        label = str(row[label_col]).strip().lower()
        if not text or n_extra >= max_extra:
            continue
        # Skip if already has negation
        if re.search(r"\b(not|no|never|n't)\b", text, re.I):
            continue
        # Simple negated form: "X was positive" -> "X was not positive"
        negated = None
        new_label = None
        if re.search(r"\b(was|were|is|are)\s+(positive|negative|strong|weak|good|bad|disappointing|encouraging)\b", text, re.I):
            negated = re.sub(r"\b(was|were|is|are)\s+", r"\1 not ", text, count=1, flags=re.I)
            new_label = "negative" if label == "positive" else ("positive" if label == "negative" else "neutral")
        elif re.search(r"\b(missed|beat|exceeded|met)\b", text, re.I):
            negated = re.sub(r"\b(missed|beat|exceeded|met)\b", r"did not \1", text, count=1, flags=re.I)
            new_label = "positive" if label == "negative" else ("negative" if label == "positive" else "neutral")
        if negated and new_label and negated not in seen:
            seen.add(negated)
            out_rows.append({text_col: negated, label_col: new_label})
            n_extra += 1
    if not out_rows:
        return df
    extra = pd.DataFrame(out_rows)
    return pd.concat([df, extra], ignore_index=True)


def add_conditional_neutral_examples(df: pd.DataFrame, text_col: str = "text", label_col: str = "label", max_extra: int = 150) -> pd.DataFrame:
    """
    Add synthetic conditional sentences labeled neutral (conditionality dominates).
    E.g. "If the acquisition closes, this would be positive for shareholders" -> neutral.
    """
    conditional_detector = ConditionalDetector()
    # From existing conditionals in data, we could duplicate with neutral; or add templates
    templates = [
        "If the deal closes, this would be positive for shareholders.",
        "Unless the merger is approved, the outlook would be negative.",
        "Assuming the acquisition completes, results could improve.",
    ]
    out_rows = []
    for t in templates:
        out_rows.append({text_col: t, label_col: "neutral"})
    if len(out_rows) >= max_extra:
        extra = pd.DataFrame(out_rows[:max_extra])
    else:
        extra = pd.DataFrame(out_rows)
    return pd.concat([df, extra], ignore_index=True)
