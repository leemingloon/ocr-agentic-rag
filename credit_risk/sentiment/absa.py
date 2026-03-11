"""
Aspect-based sentiment (ABSA): entity-level sentiment via spaCy NER and context windows.
Run FinBERT on ±context_window tokens around each entity; return entity_sentiments.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Entity types we care about for financial sentiment
FINANCIAL_ENTITY_LABELS = {"ORG", "PERSON", "PRODUCT", "GPE", "MONEY"}


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


class ABSALayer:
    """
    For each financial entity (ORG, PERSON, PRODUCT) in the sentence, extract a context
    window (±context_window tokens) and run sentiment on that span. Return sentence-level
    + list of {entity, type, span, sentiment, score}.
    """

    def __init__(self, context_window: int = 3, sentiment_fn: Optional[Callable[[str], Dict[str, Any]]] = None):
        self.context_window = context_window
        self._nlp = _load_nlp()
        self._sentiment_fn = sentiment_fn

    @property
    def available(self) -> bool:
        return self._nlp is not None and self._sentiment_fn is not None

    def extract_entities_and_contexts(self, text: str) -> List[Dict[str, Any]]:
        """
        Return list of { "entity": str, "label": str, "start": int, "end": int, "context": str }.
        context = ±context_window tokens around entity.
        """
        if not self._nlp or not text or not text.strip():
            return []
        doc = self._nlp(text)
        tokens = list(doc)
        result = []
        for ent in doc.ents:
            if ent.label_ not in FINANCIAL_ENTITY_LABELS:
                continue
            # Token indices for entity
            start_t = ent.start
            end_t = ent.end
            # Context: extend by context_window tokens each side
            ctx_start = max(0, start_t - self.context_window)
            ctx_end = min(len(tokens), end_t + self.context_window)
            context_tokens = tokens[ctx_start:ctx_end]
            context_str = " ".join(t.text for t in context_tokens)
            result.append({
                "entity": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "context": context_str,
            })
        return result

    def run_entity_sentiment(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities, run sentiment_fn on each context, return list of
        { "entity", "label", "context", "sentiment", "score" }.
        """
        if not self._sentiment_fn:
            return []
        spans = self.extract_entities_and_contexts(text)
        out = []
        for s in spans:
            ctx = s.get("context") or s.get("entity", "")
            if not ctx.strip():
                continue
            res = self._sentiment_fn(ctx)
            if isinstance(res, dict):
                label = res.get("label", res.get("sentiment", "neutral"))
                score = float(res.get("score", res.get("confidence", 0.5)))
            else:
                label, score = "neutral", 0.5
            out.append({
                "entity": s["entity"],
                "label": s["label"],
                "context": s["context"],
                "sentiment": label,
                "score": score,
            })
        return out
