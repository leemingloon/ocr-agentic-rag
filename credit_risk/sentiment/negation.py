"""
Negation handling: detect negation scope via spaCy and flip polarity of sentiment tokens in scope.

Used as preprocessing: rewrite text so that "did not miss" is interpreted as positive before FinBERT.
We identify negated spans and optionally flip sentiment by post-processing or by generating
a "negation-normalized" text hint (e.g. replace "not disappointing" with a positive cue).
"""

from __future__ import annotations

import re
from typing import List, Tuple

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# Simple polarity flip hints for common negated phrases (when spaCy not available or as backup)
NEGATION_FLIP_PHRASES = [
    (re.compile(r"\bnot\s+disappointing\b", re.I), " encouraging "),
    (re.compile(r"\bdid\s+not\s+miss\b", re.I), " met "),
    (re.compile(r"\bnever\s+missed\b", re.I), " met "),
    (re.compile(r"\bno\s+longer\s+(?:negative|bad|weak|poor)\b", re.I), " improved "),
    (re.compile(r"\bnot\s+(?:bad|negative|poor|weak)\b", re.I), " positive "),
]


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


class NegationHandler:
    """
    Detect negation scope using spaCy dependency parsing (neg relation).
    Flip polarity of sentiment-bearing tokens within scope for downstream model.
    """

    def __init__(self):
        self._nlp = _load_nlp()

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def has_negation(self, text: str) -> bool:
        """Return True if sentence contains negation in a way that affects sentiment."""
        if self._nlp is None:
            return _has_negation_regex(text)
        doc = self._nlp(text)
        for token in doc:
            if token.dep_ == "neg" or token.lemma_.lower() in ("not", "no", "never", "n't"):
                return True
        return False

    def get_negation_scopes(self, text: str) -> List[Tuple[int, int]]:
        """
        Return list of (start_char, end_char) spans that are under negation scope.
        Uses spaCy: tokens with dep_ == 'neg' and their head's subtree.
        """
        if self._nlp is None:
            return []
        doc = self._nlp(text)
        scopes = []
        for token in doc:
            if token.dep_ != "neg":
                continue
            # Scope: head and its descendants (subtree)
            start = token.head.idx
            end = token.head.idx + len(token.head.text)
            for child in token.head.subtree:
                start = min(start, child.idx)
                end = max(end, child.idx + len(child.text))
            scopes.append((start, end))
        return scopes

    def preprocess_for_sentiment(self, text: str) -> str:
        """
        Return a version of text that reduces negation misinterpretation.
        Option A: replace negated phrases with polarity-flipped hints so FinBERT sees positive/negative cue.
        Used when we don't want to change the model — we change the input instead.
        """
        out = text
        for pattern, replacement in NEGATION_FLIP_PHRASES:
            out = pattern.sub(replacement, out)
        if self._nlp is not None:
            # Optional: for "X did not Y" where Y is negative, we could insert a positive cue
            # Here we only do regex-based flip; full scope-based rewrite would need sentiment lexicon
            pass
        return out


def _has_negation_regex(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(
        p.search(text) for p in [
            re.compile(r"\bnot\b"),
            re.compile(r"\bno\s+\w+"),
            re.compile(r"\bnever\b"),
            re.compile(r"\bn't\b"),
            re.compile(r"\bno\s+longer\b"),
        ]
    )


def flip_label_in_scope(label: str) -> str:
    """Flip positive <-> negative; neutral stays neutral."""
    if label == "positive":
        return "negative"
    if label == "negative":
        return "positive"
    return label
