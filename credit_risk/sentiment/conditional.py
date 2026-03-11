"""
Conditional sentiment: detect sentences with conditionality (if, unless, would, could, might)
and apply confidence penalty — reclassify as neutral if model confidence < threshold.
"""

from __future__ import annotations

import re
from typing import List, Tuple

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Subordinating conjunctions that introduce conditionality
CONDITIONAL_CONJ = {
    "if", "unless", "provided", "assuming", "when", "whenever",
    "in case", "as long as", "given that", "so long as",
}
# Modal verbs that signal hypothetical/uncertain
CONDITIONAL_MODALS = {"would", "could", "might", "may", "should", "can"}


def _load_nlp():
    if not SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            return spacy.load("en_core_web_trf")
        except OSError:
            return None


class ConditionalDetector:
    """Detect conditional clauses via spaCy (subordinating conjunctions) and modal verbs."""

    def __init__(self):
        self._nlp = _load_nlp()

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def is_conditional(self, text: str) -> bool:
        """Return True if sentence contains conditional/hypothetical markers."""
        if not text or not text.strip():
            return False
        # Regex fallback when spaCy not available
        text_lower = text.lower().strip()
        tokens = set(re.findall(r"\b\w+\b", text_lower))
        if tokens & CONDITIONAL_CONJ:
            return True
        if tokens & CONDITIONAL_MODALS:
            return True
        if re.search(r"\bif\s+\w+", text_lower) or re.search(r"\bwould\s+be\b", text_lower):
            return True
        if self._nlp is not None:
            doc = self._nlp(text)
            for token in doc:
                if token.lower_ in CONDITIONAL_CONJ or token.lower_ in CONDITIONAL_MODALS:
                    return True
        return False
